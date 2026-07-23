"""
Bước 1 (Phương án A) — BUILD CHÍNH THỨC corpus + QA pairs từ ViQuAD 2.0.

Tiền đề: đã chạy survey_viquad.py và ĐÃ RÀ TAY titles_inventory.csv
(hai cột is_history, is_vietnam) — chỉ giữ bài is_history == 1.

Sinh 3 file:
  1. corpus.parquet    — đoạn văn lịch sử (đã khử trùng) + split. Dùng để SINH câu hỏi + index RAG.
  2. qa_pairs.parquet  — mọi câu hỏi (answerable + unanswerable) gắn với đoạn, KÈM:
        - answer_text, answer_start  -> answer-aware QG cho ViT5 (miễn phí)
        - is_impossible              -> NHÃN VÀNG cho Evidence Match của Verifier
  3. splits_report.txt — thống kê chia split, kiểm tra chống rò rỉ.

CHỐNG RÒ RỈ: chia split theo `title` (cấp BÀI), không theo câu hỏi/đoạn.
Các đoạn cùng một bài chia sẻ thực thể -> phải nằm cùng một split, dùng chung
cho cả 3 phương pháp. RAG chỉ index train+dev, KHÔNG index test.

Chạy:
    python src/preprocess/build_corpus.py \
        --inventory data/processed/titles_inventory.csv \
        --outdir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

MIN_TOKENS = 40
YEAR_RE = re.compile(r"\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b")
SPLIT_RATIOS = (0.80, 0.10, 0.10)   # train / dev / test


# ----------------------------- Tiện ích (đồng bộ với survey) ----------------

def normalize_vi(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    return len(text.split())


def year_stats(text: str) -> tuple[int, Optional[int]]:
    years = [int(y) for y in YEAR_RE.findall(text)]
    if not years:
        return 0, None
    primary = max(set(years), key=lambda y: (years.count(y), -y))
    return len(years), primary


def make_context_id(text: str) -> str:
    return "ctx_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# Chia split cân tải theo title (thay hash mù). Xem split_balanced.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "split_balanced",
    str(Path(__file__).with_name("split_balanced.py")))
_sb = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_sb)
balanced_split_by_title = _sb.balanced_split_by_title
report_split = _sb.report_split


# ----------------------------- Trích answer span ----------------------------

def first_answer(answers: Any) -> tuple[str, Optional[int]]:
    """ViQuAD: answers = {'text': [...], 'answer_start': [...]}.
    Trả về (answer_text, answer_start) của đáp án đầu tiên, hoặc ("", None)."""
    if not isinstance(answers, dict):
        return "", None
    texts = answers.get("text") or []
    starts = answers.get("answer_start") or []
    if not texts:
        return "", None
    txt = normalize_vi(str(texts[0]))
    start = int(starts[0]) if starts else None
    return txt, start


# ----------------------------- Build ----------------------------------------

def build(rows: Iterable[dict], keep_titles: set[str],
          vn_titles: set[str]) -> tuple[list[dict], list[dict], dict]:
    """
    rows: iterable ViQuAD (title, context, question, answers, is_impossible).
    keep_titles: tập title is_history == 1 (sau khi rà tay).
    vn_titles:   tập title is_vietnam == 1.
    Trả về (corpus_records, qa_records, stats).
    """
    corpus: dict[str, dict] = {}          # context_id -> đoạn (khử trùng)
    qa: list[dict] = []
    title_sizes: dict[str, int] = defaultdict(int)   # title -> số đoạn (để chia cân tải)
    title_is_vn: dict[str, bool] = {}
    stats = {"rows_seen": 0, "rows_kept": 0, "passages": 0,
             "answerable": 0, "unanswerable": 0, "skipped_not_history": 0}

    for row in rows:
        stats["rows_seen"] += 1
        title = normalize_vi(str(row.get("title", "")))
        if title not in keep_titles:
            stats["skipped_not_history"] += 1
            continue

        context = normalize_vi(str(row.get("context", "")))
        if not context or count_tokens(context) < MIN_TOKENS:
            continue

        cid = make_context_id(context)
        is_vn = title in vn_titles
        title_is_vn[title] = is_vn

        # --- corpus (khử trùng theo context) ---
        if cid not in corpus:
            n_years, primary = year_stats(context)
            n_tok = count_tokens(context)
            title_sizes[title] += 1          # đếm đoạn duy nhất cho việc chia split
            corpus[cid] = {
                "context_id": cid,
                "title": title,
                "context": context,
                "n_tokens": n_tok,
                "n_years": n_years,
                "year_density": round(100 * n_years / n_tok, 2),
                "primary_year": primary,
                "is_vietnam": is_vn,
                "split": None,               # gán sau bằng thuật toán cân tải
            }

        # --- qa pair ---
        is_imp = bool(row.get("is_impossible", False))
        ans_text, ans_start = first_answer(row.get("answers"))
        # validate answer span nằm đúng trong context (nếu answerable)
        span_ok = False
        if not is_imp and ans_text:
            if ans_start is not None and 0 <= ans_start <= len(context):
                span_ok = context[ans_start:ans_start + len(ans_text)] == ans_text
            if not span_ok:                      # fallback: tìm chuỗi con
                span_ok = ans_text in context

        qa.append({
            "qa_id": "qa_" + hashlib.md5(
                (title + str(row.get("question", "")) + ans_text).encode()
            ).hexdigest()[:12],
            "context_id": cid,
            "title": title,
            "question": normalize_vi(str(row.get("question", ""))),
            "answer_text": ans_text,
            "answer_start": ans_start,
            "answer_span_ok": span_ok,
            "is_impossible": is_imp,
            "plausible_answer": normalize_vi(str(row.get("plausible", "") or "")),
            "is_vietnam": is_vn,
            "split": None,               # gán sau
        })
        stats["rows_kept"] += 1
        stats["unanswerable" if is_imp else "answerable"] += 1

    stats["passages"] = len(corpus)

    # --- CHIA SPLIT CÂN TẢI theo title (sau khi đã biết hết size) ---
    title2split = balanced_split_by_title(dict(title_sizes), title_is_vn)
    for c in corpus.values():
        c["split"] = title2split[c["title"]]
    for q in qa:
        q["split"] = title2split[q["title"]]

    return list(corpus.values()), qa, stats, title_sizes, title_is_vn, title2split


def load_inventory(path: str) -> tuple[set[str], set[str]]:
    """Đọc titles_inventory.csv đã rà tay. Trả về (history_titles, vn_titles)."""
    import pandas as pd
    inv = pd.read_csv(path)
    inv["title"] = inv["title"].astype(str).map(normalize_vi)
    hist = set(inv.loc[inv["is_history"] == 1, "title"])
    vn = set(inv.loc[inv["is_vietnam"] == 1, "title"]) if "is_vietnam" in inv else set()
    return hist, vn


def iter_viquad() -> Iterable[dict]:
    from datasets import load_dataset
    ds = load_dataset("taidng/UIT-ViQuAD2.0")
    for split in ds:
        for row in ds[split]:
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/processed/titles_inventory.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    hist_titles, vn_titles = load_inventory(args.inventory)
    print(f"Bài lịch sử giữ lại: {len(hist_titles)}  |  trong đó VN: {len(vn_titles)}")

    rows = iter_viquad()
    if args.limit:
        rows = (r for i, r in enumerate(rows) if i < args.limit)

    corpus, qa, stats, title_sizes, title_is_vn, title2split = build(
        rows, hist_titles, vn_titles)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cdf = pd.DataFrame(corpus)
    qdf = pd.DataFrame(qa)
    cdf.to_parquet(outdir / "corpus.parquet", index=False)
    qdf.to_parquet(outdir / "qa_pairs.parquet", index=False)

    # ---- kiểm tra chống rò rỉ ----
    leak = cdf.groupby("context_id")["split"].nunique().max()
    assert leak == 1, "RÒ RỈ: một đoạn nằm ở nhiều split!"
    title_leak = cdf.groupby("title")["split"].nunique().max()
    assert title_leak == 1, "RÒ RỈ: một bài nằm ở nhiều split!"

    # ---- báo cáo ----
    lines = []
    lines.append("=== BUILD CORPUS LỊCH SỬ (ViQuAD 2.0) ===")
    for k, v in stats.items():
        lines.append(f"  {k:20s}: {v:,}")
    lines.append("")
    lines.append(report_split(dict(title_sizes), title_is_vn, title2split))
    lines.append("  (chi tiết câu hỏi mỗi split:)")
    for sp in ["train", "dev", "test"]:
        n_qa = (qdf.split == sp).sum()
        n_imp = ((qdf.split == sp) & qdf.is_impossible).sum()
        lines.append(f"    {sp:5s}: {n_qa:5d} câu hỏi ({n_imp} không trả lời được)")
    lines.append(f"  Kiểm tra rò rỉ: đoạn={leak} split, bài={title_leak} split (phải =1) ✅")
    lines.append("")
    lines.append("=== NHÃN VÀNG CHO VERIFIER ===")
    lines.append(f"  answerable có span hợp lệ : {int(qdf.answer_span_ok.sum()):,}")
    lines.append(f"  unanswerable (is_impossible): {int(qdf.is_impossible.sum()):,}")
    lines.append("")
    lines.append("=== TIỂU MIỀN (trục phân tích độ bền) ===")
    lines.append(f"  đoạn VN     : {int(cdf.is_vietnam.sum()):,}")
    lines.append(f"  đoạn thế giới: {int((~cdf.is_vietnam).sum()):,}")

    report = "\n".join(lines)
    print(report)
    (outdir / "splits_report.txt").write_text(report, encoding="utf-8")
    print(f"\nĐã lưu: corpus.parquet ({len(cdf)} đoạn), "
          f"qa_pairs.parquet ({len(qdf)} câu), splits_report.txt")


if __name__ == "__main__":
    main()
