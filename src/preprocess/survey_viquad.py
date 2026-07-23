"""
Bước 1 (Phương án A) — KHẢO SÁT corpus từ `taidng/UIT-ViQuAD2.0`.

Mục tiêu: đếm xem miền LỊCH SỬ có đủ đoạn văn để làm corpus không (go/no-go).

Đơn vị dữ liệu ViQuAD: mỗi dòng = (title, context, question, answers, is_impossible).
Nhiều câu hỏi dùng chung một `context` -> phải KHỬ TRÙNG theo context để ra corpus.
Lọc lịch sử = lọc theo `title` (chỉ 176 bài).

Đầu ra:
  1. titles_inventory.csv  — TOÀN BỘ 176 tiêu đề + số đoạn + số câu hỏi + cờ đoán-lịch-sử.
     -> Bạn MỞ file này, sửa tay cột `is_history` (15 phút), rồi chạy build chính thức.
  2. corpus_all.parquet     — mọi đoạn văn (đã khử trùng).
  3. corpus_history.parquet — đoạn thuộc bài được đoán là lịch sử.

Chạy (máy có internet):
    pip install datasets pandas pyarrow
    python src/preprocess/survey_viquad.py --outdir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

# ----------------------------- Tham số --------------------------------------

MIN_TOKENS = 40
YEAR_RE = re.compile(r"\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b")
GO_THRESHOLD = 1500        # >= : GO trên miền lịch sử chặt
SOFT_THRESHOLD = 800       # >= : GO nhưng cần nới định nghĩa lịch sử

# Từ khoá đoán tiêu đề lịch sử — CHỈ LÀ GỢI Ý BAN ĐẦU.
# Bạn phải rà lại titles_inventory.csv và sửa tay cột is_history.
HISTORY_KEYWORDS = [
    "nhà ", "triều", "vương triều", "hoàng đế", "vua ", "chúa ",
    "chiến tranh", "chiến dịch", "khởi nghĩa", "kháng chiến", "trận ",
    "cách mạng", "bắc thuộc", "phong kiến", "độc lập", "thống nhất",
    "lý ", "trần ", "lê ", "nguyễn", "đinh ", "hồ ", "mạc ", "tây sơn",
    "ngô ", "tiền lê", "hậu lê", "quang trung", "gia long", "minh mạng",
    "hồ chí minh", "điện biên", "việt minh", "đông dương", "pháp thuộc",
    "lịch sử", "thời kỳ", "thời đại", "đế quốc", "vương quốc", "quốc gia",
]


# ----------------------------- Tiện ích -------------------------------------

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


def guess_history(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in HISTORY_KEYWORDS)


# ----------------------------- Xây corpus -----------------------------------

def build(rows: Iterable[dict]) -> tuple[dict, dict, dict]:
    """Trả về (passages, title_stats, stats).
    passages: context_id -> dict bản ghi đoạn văn (đã khử trùng).
    title_stats: title -> {n_answerable, n_unanswerable, contexts:set}.
    """
    passages: dict[str, dict] = {}
    title_stats: dict[str, dict] = defaultdict(
        lambda: {"n_answerable": 0, "n_unanswerable": 0, "contexts": set()}
    )
    stats = {"total_rows": 0, "unique_passages": 0, "too_short_passages": 0,
             "answerable": 0, "unanswerable": 0}

    for row in rows:
        stats["total_rows"] += 1
        title = normalize_vi(str(row.get("title", "")))
        context = normalize_vi(str(row.get("context", "")))
        if not context:
            continue
        is_imp = bool(row.get("is_impossible", False))

        # đếm câu hỏi theo bài
        ts = title_stats[title]
        ts["contexts"].add(make_context_id(context))
        if is_imp:
            ts["n_unanswerable"] += 1
            stats["unanswerable"] += 1
        else:
            ts["n_answerable"] += 1
            stats["answerable"] += 1

        # khử trùng đoạn văn
        cid = make_context_id(context)
        if cid not in passages:
            n_tok = count_tokens(context)
            n_years, primary = year_stats(context)
            passages[cid] = {
                "context_id": cid,
                "title": title,
                "context": context,
                "n_tokens": n_tok,
                "n_years": n_years,
                "year_density": round(100 * n_years / n_tok, 2) if n_tok else 0.0,
                "primary_year": primary,
                "is_history_guess": guess_history(title),
                "too_short": n_tok < MIN_TOKENS,
            }

    stats["unique_passages"] = len(passages)
    stats["too_short_passages"] = sum(p["too_short"] for p in passages.values())
    return passages, title_stats, stats


def iter_viquad() -> Iterable[dict]:
    """Nạp mọi split của ViQuAD 2.0. Cần internet + `pip install datasets`."""
    from datasets import load_dataset
    ds = load_dataset("taidng/UIT-ViQuAD2.0")
    for split in ds:
        for row in ds[split]:
            yield row


# ----------------------------- Báo cáo --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--limit", type=int, default=0, help="0=toàn bộ; >0 để test nhanh")
    args = ap.parse_args()

    import pandas as pd

    rows = iter_viquad()
    if args.limit:
        rows = (r for i, r in enumerate(rows) if i < args.limit)

    passages, title_stats, stats = build(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- bảng đoạn văn ---
    df = pd.DataFrame(list(passages.values()))
    df = df[~df.too_short].reset_index(drop=True)   # bỏ đoạn quá ngắn
    df.to_parquet(outdir / "corpus_all.parquet", index=False)

    df_hist = df[df.is_history_guess].reset_index(drop=True)
    df_hist.to_parquet(outdir / "corpus_history.parquet", index=False)

    # --- bảng kiểm kê tiêu đề (để sửa tay is_history) ---
    inv = []
    for title, ts in title_stats.items():
        inv.append({
            "title": title,
            "n_passages": len(ts["contexts"]),
            "n_answerable_q": ts["n_answerable"],
            "n_unanswerable_q": ts["n_unanswerable"],
            "is_history": int(guess_history(title)),   # sửa tay cột này
        })
    inv_df = pd.DataFrame(inv).sort_values("n_passages", ascending=False)
    inv_df.to_csv(outdir / "titles_inventory.csv", index=False, encoding="utf-8-sig")

    # --- in báo cáo ---
    print("=== TỔNG QUAN VIQUAD 2.0 ===")
    print(f"  Tổng dòng (câu hỏi)   : {stats['total_rows']:,}")
    print(f"  - trả lời được        : {stats['answerable']:,}")
    print(f"  - KHÔNG trả lời được  : {stats['unanswerable']:,}  (nhãn vàng cho Evidence Match)")
    print(f"  Số bài (title)        : {len(title_stats):,}")
    print(f"  Đoạn văn duy nhất     : {stats['unique_passages']:,}")
    print(f"  - bỏ vì quá ngắn      : {stats['too_short_passages']:,}")
    print(f"  Đoạn dùng được        : {len(df):,}")

    print("\n=== MIỀN LỊCH SỬ (đoán tự động theo tiêu đề) ===")
    n_hist = len(df_hist)
    print(f"  Đoạn lịch sử          : {n_hist:,}")
    print(f"  year_density trung vị : {df_hist.year_density.median():.2f}" if n_hist else "  (rỗng)")
    print(f"  n_tokens trung vị     : {df_hist.n_tokens.median():.0f}" if n_hist else "")

    print("\n=== QUYẾT ĐỊNH GO/NO-GO ===")
    if n_hist >= GO_THRESHOLD:
        print(f"  ✅ GO — {n_hist:,} >= {GO_THRESHOLD}. Đủ đoạn cho miền lịch sử chặt.")
    elif n_hist >= SOFT_THRESHOLD:
        print(f"  ⚠️  GO CÓ ĐIỀU KIỆN — {n_hist:,} trong [{SOFT_THRESHOLD}, {GO_THRESHOLD}).")
        print("     Nới định nghĩa lịch sử (thêm tiểu sử nhân vật, địa danh) khi sửa is_history.")
    else:
        print(f"  ❌ NO-GO trên miền chặt — chỉ {n_hist:,} < {SOFT_THRESHOLD}.")
        print("     Phương án: train ViT5 trên TOÀN BỘ ViQuAD, chỉ ĐÁNH GIÁ trên miền lịch sử.")

    print("\n=== VIỆC TIẾP THEO ===")
    print(f"  1. Mở {outdir/'titles_inventory.csv'} — rà 176 bài, sửa cột is_history (1/0).")
    print(f"  2. Chạy build chính thức đọc CSV đã sửa (script bước sau).")
    print("\nTop 15 bài được đoán là lịch sử (theo số đoạn):")
    top = inv_df[inv_df.is_history == 1].head(15)
    for _, r in top.iterrows():
        print(f"    {r['n_passages']:4d} đoạn | {r['title']}")


if __name__ == "__main__":
    main()
