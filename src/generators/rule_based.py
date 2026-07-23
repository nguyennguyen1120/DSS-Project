"""
Rule-Based Generator — Phương pháp 1.

Pipeline:
  1. Answer selection : tìm span đáp án trong context theo wh_type yêu cầu.
  2. Template         : chọn template câu hỏi theo wh_type + phân tích cú pháp đơn giản.
  3. Distractor       : lấy từ KB pool (cùng type, cùng bucket), fallback nới bucket.
  4. finalize_options : khử trùng, xáo trộn A/B/C/D, trả MCQItem.

Điểm mạnh: YEAR/DATE (regex chính xác 100%), không cần LLM, chạy offline.
Điểm yếu đã biết: câu hỏi cứng (template), yếu với PER/LOC nước ngoài (KB kém), 
không sinh được Nguyên nhân/Ý nghĩa (non-factoid).
Ghi nhận vào báo cáo — đây là bằng chứng cho luận điểm "cần nhiều phương pháp".

Input : (corpus_row, qa_row, wh_type, bloom, distractor_pool, entities_df)
Output: MCQItem (schema chung)
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Optional

# Thêm src/schema và src/preprocess vào sys.path theo đường dẫn tuyệt đối.
# Hoạt động bất kể bạn đứng ở thư mục nào khi chạy lệnh.
_SRC = Path(__file__).resolve().parent.parent   # files_v1/src/
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))

from mcq_schema import (
    MCQItem, Generator, Source, Request, Evidence,
    Metadata, GenerationTrace, Provenance,
    Method, Bloom, WhType, finalize_options, make_item_id,
)

# ============================ Template câu hỏi ==============================

# Template theo wh_type. {subj} = chủ đề (từ câu/tiêu đề), {ctx} = trích câu.
# Giữ đơn (3-5 template/type) để tránh câu hỏi quá máy móc.

TEMPLATES: dict[WhType, list[str]] = {
    WhType.thoi_gian: [
        "{subj} xảy ra vào năm nào?",
        "Sự kiện {subj} diễn ra vào năm nào?",
        "{subj} diễn ra vào thời gian nào?",
        "Năm nào {subj} diễn ra?",
    ],
    # template riêng khi context là tiểu sử (subj = tên người)
    "thoi_gian_person": [
        "{subj} sinh năm nào?",
        "{subj} qua đời năm nào?",
        "{subj} sống trong thế kỷ nào?",
        "Năm sinh của {subj} là năm nào?",
    ],
    WhType.nhan_vat: [
        "Ai {verb} {subj}?",
        "Người nào {verb} {subj}?",
        "{subj} do ai {verb}?",
        "Nhân vật nào liên quan đến {subj}?",
    ],
    WhType.dia_diem: [
        "{subj} diễn ra ở đâu?",
        "{subj} xảy ra tại địa điểm nào?",
        "Địa điểm nào liên quan đến {subj}?",
        "{subj} diễn ra tại đâu?",
    ],
    WhType.su_kien: [
        "Sự kiện nào xảy ra liên quan đến {subj}?",
        "{subj} có liên quan đến sự kiện nào?",
        "Điều gì đã xảy ra với {subj}?",
    ],
    WhType.nguyen_nhan: [
        "Nguyên nhân nào dẫn đến {subj}?",
        "Vì sao {subj} xảy ra?",
        "Điều gì dẫn đến {subj}?",
    ],
    WhType.y_nghia: [
        "{subj} có ý nghĩa gì?",
        "Tầm quan trọng của {subj} là gì?",
        "Kết quả của {subj} là gì?",
    ],
}

# Map wh_type -> entity type ưu tiên để chọn đáp án
WH_TO_ENTITY: dict[WhType, list[str]] = {
    WhType.thoi_gian: ["YEAR", "DATE"],
    WhType.nhan_vat:  ["PER"],
    WhType.dia_diem:  ["LOC"],
    WhType.su_kien:   ["ORG", "LOC", "PER"],
    WhType.nguyen_nhan: [],   # non-factoid: không có entity đơn
    WhType.y_nghia:   [],     # non-factoid
}

# ============================ Answer selection ==============================

def _normalize_surface(surface: str, ent_type: str) -> str:
    """Chuẩn hoá surface để hiển thị trong phương án.
    YEAR: bỏ tiền tố 'năm ' -> '1954' thay vì 'năm 1954'.
    """
    if ent_type in ("YEAR", "DATE"):
        surface = re.sub(r"^năm\s+", "", surface.strip())
    return surface.strip()


def select_answer(
    context: str,
    wh_type: WhType,
    entities: list[dict],
    rng: random.Random,
) -> Optional[dict]:
    """Chọn entity phù hợp làm đáp án đúng.
    Ưu tiên entity có answer_start (từ QA gold), rồi mới tìm trong context.
    Trả về dict entity hoặc None nếu không tìm được.
    """
    want_types = WH_TO_ENTITY.get(wh_type, [])
    if not want_types:
        return None   # non-factoid: rule-based bỏ qua

    candidates = [e for e in entities if e.get("type") in want_types]
    if not candidates:
        return None

    # ưu tiên entity xuất hiện trong câu đầu (thường là câu chủ đề)
    first_sent_end = context.find(".") + 1
    first_sent_end = first_sent_end if first_sent_end > 0 else len(context)
    first_sent_ents = [e for e in candidates
                       if (e.get("char_start") or 0) < first_sent_end]
    pool = first_sent_ents if first_sent_ents else candidates
    return rng.choice(pool)


# ============================ Evidence sentence =============================

def find_evidence(context: str, answer_text: str) -> tuple[str, bool]:
    """Tìm câu trong context chứa answer_text.
    Trả về (câu, found_in_context).
    """
    for sent in re.split(r"(?<=[.!?])\s+", context):
        if answer_text in sent:
            return sent.strip(), True
    # fallback: lấy câu đầu
    first = re.split(r"(?<=[.!?])\s+", context)[0].strip()
    return first, False


# ============================ Sinh câu hỏi (template) ======================

def _extract_subject(context: str, answer_text: str, title: str) -> tuple[str, str]:
    """Trích chủ đề {subj} và động từ {verb} đơn giản cho template.
    Heuristic: dùng tiêu đề bài là chủ đề chính, động từ từ câu chứa đáp án.
    """
    subj = title or context[:40].split(",")[0].strip()
    # tìm động từ đơn giản trong câu chứa đáp án
    ev_sent, _ = find_evidence(context, answer_text)
    verbs = re.findall(r"\b(lãnh đạo|chỉ huy|sáng lập|thành lập|chinh phục"
                       r"|cai trị|kiểm soát|đặt ra|ban hành|ký kết)\b", ev_sent)
    verb = verbs[0] if verbs else "thực hiện"
    return subj, verb


def build_question(
    wh_type: WhType,
    subj: str,
    verb: str,
    rng: random.Random,
    is_person_context: bool = False,
) -> str:
    # nếu hỏi thời gian nhưng chủ đề là người -> dùng template tiểu sử
    if wh_type == WhType.thoi_gian and is_person_context:
        templates = TEMPLATES.get("thoi_gian_person",
                                  TEMPLATES[WhType.thoi_gian])
    else:
        templates = TEMPLATES.get(wh_type, ["{subj} là gì?"])
    tmpl = rng.choice(templates)
    q = tmpl.format(subj=subj, verb=verb)
    if not q.endswith("?"):
        q += "?"
    return q


# ============================ Distractor generation =========================

def get_distractors(
    answer_surface: str,
    answer_type: str,
    bucket: str,
    pool: dict,
    rng: random.Random,
    n: int = 3,
    max_expand: int = 2,
) -> list[str]:
    """Lấy N distractor từ pool, fallback nới bucket nếu thiếu.

    Chiến lược bucket cho YEAR:
      bucket gốc (cùng thế kỷ) -> lân cận (±1C) -> pool nổi tiếng
    Chiến lược cho PER/LOC/ORG:
      cùng bucket (VN/TG) -> bucket kia -> toàn bộ
    Ràng buộc: distractor khác đáp án, không trùng nhau.
    """
    answer_lower = answer_surface.lower()
    collected: list[str] = []
    seen = {answer_lower}

    def _draw_from(candidates: set[str]) -> None:
        avail = [c for c in candidates
                 if c.lower() not in seen and c.strip()]
        rng.shuffle(avail)
        for c in avail:
            if len(collected) >= n:
                break
            collected.append(c)
            seen.add(c.lower())

    # --- cùng bucket ---
    _draw_from(pool.get((answer_type, bucket), set()))

    # --- fallback: nới bucket ---
    if len(collected) < n:
        if answer_type in ("YEAR", "DATE"):
            # nới sang thế kỷ lân cận
            century_num = _parse_century(bucket)
            for delta in range(1, max_expand + 1):
                for c in [f"{century_num+delta}C", f"{century_num-delta}C"]:
                    _draw_from(pool.get((answer_type, c), set()))
                    if len(collected) >= n:
                        break
            # cuối cùng: pool năm nổi tiếng cứng (chắc chắn có)
            FAMOUS_YEARS = {"1945", "1954", "1975", "1986", "1776", "1789",
                            "1914", "1918", "1939", "1945", "1066", "1453"}
            extra = [y for y in FAMOUS_YEARS
                     if y.lower() != answer_lower and y not in seen]
            rng.shuffle(extra)
            for y in extra:
                if len(collected) >= n:
                    break
                collected.append(y)
                seen.add(y)
        else:
            # nới sang bucket kia (VN <-> TG)
            alt_bucket = "VN" if bucket == "TG" else "TG"
            _draw_from(pool.get((answer_type, alt_bucket), set()))
            # rồi thử type lân cận (LOC thay LOC, PER thay PER)
            for alt_type in ["LOC", "PER", "ORG"]:
                if alt_type != answer_type:
                    continue
                for bkt in [bucket, alt_bucket]:
                    _draw_from(pool.get((alt_type, bkt), set()))

    return [_normalize_surface(d, answer_type) for d in collected[:n]]


def _parse_century(bucket: str) -> int:
    try:
        return int(bucket.replace("C", ""))
    except ValueError:
        return 20


# ============================ Generator chính ===============================

def generate_one(
    context: str,
    title: str,
    context_id: str,
    is_vietnam: bool,
    wh_type: WhType,
    bloom: Bloom,
    entities: list[dict],       # entities của đoạn này (từ entities.parquet)
    pool: dict,                 # distractor_pool toàn corpus
    variant: str = "template_v1",
    seed: Optional[int] = None,
) -> Optional[MCQItem]:
    """Sinh một MCQItem hoặc trả None nếu không đủ điều kiện.

    None xảy ra khi:
    - wh_type là non-factoid (nguyen_nhan, y_nghia) → rule-based bỏ qua
    - không tìm được entity phù hợp làm đáp án
    - không đủ 3 distractor dù đã fallback
    """
    rng = random.Random(seed)

    # 1. Answer selection
    ans_ent = select_answer(context, wh_type, entities, rng)
    if ans_ent is None:
        return None

    answer_type = ans_ent["type"]
    answer_text = _normalize_surface(ans_ent["surface"], answer_type)
    bucket = ans_ent.get("bucket", "TG")
    char_start = ans_ent.get("char_start")
    char_end = ans_ent.get("char_end")

    # 2. Evidence sentence
    ev_sent, found = find_evidence(context, answer_text)

    # phát hiện context tiểu sử: tiêu đề bài là tên người (có PER entity khớp title)
    is_person_context = any(
        e.get("type") == "PER" and title and
        e.get("surface", "").lower() in title.lower()
        for e in entities
    )

    # 3. Sinh câu hỏi
    subj, verb = _extract_subject(context, answer_text, title)
    question = build_question(wh_type, subj, verb, rng,
                              is_person_context=is_person_context)

    # 4. Distractor
    distractors = get_distractors(
        answer_text, answer_type, bucket, pool, rng, n=3)
    if len(distractors) < 3:
        return None    # pool quá thưa dù đã fallback

    # 5. finalize_options + build MCQItem
    try:
        opts, answer_key = finalize_options(
            correct_text=answer_text,
            distractors=distractors,
            correct_provenance=Provenance.context_span,
            distractor_provenance=Provenance.kb,
            correct_entity_type=answer_type,
            seed=rng.randint(0, 9999),
        )
    except ValueError:
        return None

    return MCQItem(
        item_id=make_item_id(context_id, question, "rule_based"),
        generator=Generator(method=Method.rule_based, variant=variant),
        source=Source(
            context_id=context_id,
            context=context,
            title=title,
            is_vietnam=is_vietnam,
        ),
        request=Request(
            bloom_requested=bloom,
            wh_type_requested=wh_type,
        ),
        question=question,
        options=opts,
        answer_key=answer_key,
        answer_text=answer_text,
        evidence=Evidence(
            sentence=ev_sent,
            char_start=char_start,
            char_end=char_end,
            found_in_context=found,
        ),
        metadata=Metadata(
            wh_type_detected=wh_type,
            topic=title,
            tags=[answer_type.lower(), bucket.lower()],
        ),
        generation_trace=GenerationTrace(
            n_llm_calls=0,
            cost_usd=0.0,
        ),
    )


# ============================ Batch runner ==================================

def generate_batch(
    corpus_df,
    entities_df,
    pool: dict,
    wh_types: list[WhType] | None = None,
    bloom: Bloom = Bloom.nhan_biet,
    split: str = "train",
    limit: int = 0,
    seed: int = 42,
) -> list[MCQItem]:
    """Chạy generator trên tập corpus. Chỉ dùng đoạn thuộc split chỉ định.
    Mỗi đoạn thử mỗi wh_type một lần -> tối đa N_types câu/đoạn.
    """
    if wh_types is None:
        wh_types = [WhType.thoi_gian, WhType.nhan_vat, WhType.dia_diem]

    # group entities theo context_id để tra nhanh
    ent_by_ctx: dict[str, list[dict]] = {}
    for _, row in entities_df.iterrows():
        cid = row["context_id"]
        ent_by_ctx.setdefault(cid, []).append(row.to_dict())

    df = corpus_df[corpus_df["split"] == split]
    if limit:
        df = df.head(limit)

    items, rng = [], random.Random(seed)
    for _, row in df.iterrows():
        ctx = row["context"]
        cid = row["context_id"]
        title = row.get("title", "")
        is_vn = bool(row.get("is_vietnam", False))
        ents = ent_by_ctx.get(cid, [])

        for wh in wh_types:
            item = generate_one(
                context=ctx, title=title, context_id=cid,
                is_vietnam=is_vn, wh_type=wh, bloom=bloom,
                entities=ents, pool=pool,
                seed=rng.randint(0, 999999),
            )
            if item:
                items.append(item)
    return items


# ============================ CLI ===========================================

def main():
    import argparse, json, pickle, time
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",    default="data/processed/corpus.parquet")
    ap.add_argument("--entities",  default="data/processed/entities.parquet")
    ap.add_argument("--pool",      default="data/processed/distractor_pool.pkl")
    ap.add_argument("--outdir",    default="data/generated")
    ap.add_argument("--split",     default="train")
    ap.add_argument("--limit",     type=int, default=100)
    ap.add_argument("--wh_types",  default="thoi_gian,nhan_vat,dia_diem")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    corpus_df  = pd.read_parquet(args.corpus)
    entities_df = pd.read_parquet(args.entities)
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    wh_types = [WhType(w.strip()) for w in args.wh_types.split(",")]
    t0 = time.time()
    items = generate_batch(
        corpus_df, entities_df, pool,
        wh_types=wh_types, split=args.split,
        limit=args.limit, seed=args.seed,
    )
    elapsed = time.time() - t0

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.outdir) / "rule_based.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    # thống kê
    tried = len(corpus_df[corpus_df["split"]==args.split].head(args.limit or 10**9)) * len(wh_types)
    print(f"\n=== RULE-BASED GENERATOR ===")
    print(f"  Đoạn thử       : {len(corpus_df[corpus_df['split']==args.split].head(args.limit or 10**9)):,}")
    print(f"  Câu hỏi sinh được: {len(items):,} / {tried:,} (VQR thô: {len(items)/tried:.0%})")
    print(f"  Thời gian       : {elapsed:.1f}s  ({elapsed/max(len(items),1)*1000:.0f}ms/câu)")
    print(f"  Chi phí API     : $0.00 (offline)")

    from collections import Counter
    by_wh = Counter(item.request.wh_type_requested for item in items)
    print("\n  Phân bố wh_type:")
    for wh, n in sorted(by_wh.items(), key=lambda x: -x[1]):
        print(f"    {wh.value:12s}: {n}")

    ev_ok = sum(1 for i in items if i.evidence.found_in_context)
    print(f"\n  Evidence found_in_context: {ev_ok}/{len(items)} "
          f"= {ev_ok/max(len(items),1):.0%}")
    print(f"\nĐã lưu {len(items)} câu hỏi -> {out_path}")
    print("\n[MẪU ĐẦU RA]")
    if items:
        i = items[0]
        print(f"  Q: {i.question}")
        for o in i.options:
            mark = " ←" if o.is_correct else ""
            print(f"  {o.label}. {o.text}{mark}")
        print(f"  Evidence: {i.evidence.sentence[:80]}...")


if __name__ == "__main__":
    main()
