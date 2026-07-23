"""
Inference ViT5 — sinh câu hỏi trên test set sau khi train xong.
Chạy sau train_vit5.py.

Chạy:
    python src/generators/infer_vit5.py \
        --qg_model models/vit5_qg \
        --dg_model models/vit5_dg \
        --data_dir data/processed \
        --out data/generated/vit5_ft.jsonl \
        --limit 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "schema"))

from mcq_schema import (
    MCQItem, Generator, Source, Request, Evidence,
    Metadata, GenerationTrace, Provenance,
    Method, Bloom, WhType, finalize_options, make_item_id,
)

MAX_SRC = 256
MAX_TGT = 64

WH_BLOOM = {
    "thoi_gian":   Bloom.nhan_biet,
    "nhan_vat":    Bloom.nhan_biet,
    "dia_diem":    Bloom.nhan_biet,
    "su_kien":     Bloom.thong_hieu,
    "nguyen_nhan": Bloom.thong_hieu,
    "y_nghia":     Bloom.van_dung,
}

YEAR_RE = re.compile(r"^\d{3,4}$")


def infer_wh_type(question: str, answer: str) -> str:
    if YEAR_RE.match(str(answer).strip()):
        return "thoi_gian"
    q = str(question).lower()
    WH = {
        "thoi_gian":   ["khi nào","năm nào","bao giờ","ngày nào"],
        "nhan_vat":    ["ai ","người nào","nhân vật","vua nào"],
        "dia_diem":    ["ở đâu","tại đâu","địa điểm"],
        "nguyen_nhan": ["vì sao","tại sao","nguyên nhân"],
        "y_nghia":     ["ý nghĩa","kết quả","hệ quả","dẫn đến"],
    }
    for wh, kws in WH.items():
        if any(k in q for k in kws):
            return wh
    return "su_kien"


def load_test_data(data_dir: str, limit: int = 0):
    import pandas as pd
    qa  = pd.read_parquet(f"{data_dir}/qa_pairs.parquet")
    ctx = pd.read_parquet(f"{data_dir}/corpus.parquet")

    qa_clean = qa[(qa.answer_span_ok == True) & (qa.is_impossible == False)].copy()

    # Drop các cột đã có trong corpus để tránh _x/_y khi merge
    drop_cols = [c for c in ["context", "title", "is_vietnam"]
                 if c in qa_clean.columns]
    if drop_cols:
        qa_clean = qa_clean.drop(columns=drop_cols)

    # merge để lấy context, title, is_vietnam
    df = qa_clean.merge(
        ctx[["context_id", "context", "title", "is_vietnam"]],
        on="context_id", how="inner"
    )

    df["wh_type"] = df.apply(
        lambda r: infer_wh_type(r["question"], r["answer_text"]), axis=1)

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if limit:
        test_df = test_df.sample(n=min(limit, len(test_df)),
                                  random_state=42).reset_index(drop=True)
    print(f"Test set: {len(test_df)} mẫu")
    return test_df


class ViT5Generator:
    def __init__(self, qg_model_dir: str, dg_model_dir: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        print("Loading tokenizer + models...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            qg_model_dir, use_fast=False, legacy=True)
        self.vocab_size = len(self.tokenizer)

        self.model_qg = AutoModelForSeq2SeqLM.from_pretrained(qg_model_dir)
        self.model_qg.resize_token_embeddings(self.vocab_size)
        self.model_qg = self.model_qg.to(self.device).eval()

        self.model_dg = AutoModelForSeq2SeqLM.from_pretrained(dg_model_dir)
        self.model_dg.resize_token_embeddings(self.vocab_size)
        self.model_dg = self.model_dg.to(self.device).eval()

        print("Models loaded.")

    def _encode(self, text: str, max_len: int):
        import torch
        ids = self.tokenizer(
            text, return_tensors="pt",
            max_length=max_len, truncation=True,
        ).input_ids
        # clamp
        ids = ids.clamp(0, self.vocab_size - 1)
        return ids.to(self.device)

    def generate_question(self, context: str, answer: str, wh_type: str) -> str:
        import torch
        src = (f"<type> {wh_type} </type> "
               f"<ans> {answer} </ans> "
               f"<ctx> {context[:500]} </ctx>")
        ids = self._encode(src, MAX_SRC)
        with torch.no_grad():
            out = self.model_qg.generate(
                ids, max_new_tokens=MAX_TGT,
                num_beams=4, no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def generate_distractors(self, question: str, answer: str,
                              context: str) -> list[str]:
        import torch
        src = (f"<q> {question} </q> "
               f"<ans> {answer} </ans> "
               f"<ctx> {context[:350]} </ctx>")
        ids = self._encode(src, MAX_SRC)
        with torch.no_grad():
            out = self.model_dg.generate(
                ids, max_new_tokens=48,
                num_beams=4, no_repeat_ngram_size=2,
                early_stopping=True,
            )
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Tách bằng | nếu có
        if "|" in raw:
            parts = [d.strip() for d in raw.split("|") if d.strip()]
        else:
            # Model chưa học format | → tách bằng câu/cụm
            # Lấy phần đầu của chuỗi (trước dấu phẩy hoặc chấm đầu tiên)
            # rồi dùng sliding window để lấy thêm
            import re
            # thử tách bằng dấu câu
            parts = [p.strip() for p in re.split(r'[,;.]\s+', raw) if p.strip()]
            if not parts:
                parts = [raw.strip()] if raw.strip() else []

        # Lọc: không trùng đáp án, không quá dài, không chứa đáp án bên trong,
        # không quá ngắn (<3 ký tự), không phải mảnh câu văn (có "và","khi","trong")
        answer_lower = answer.lower()
        FILLER = {"trong", "ngoài", "và", "khi", "để", "vì", "nhưng",
                  "với", "từ", "tại", "của", "là", "có", "không",
                  "letter", "năm", "tháng", "ngày"}
        clean = []
        seen = {answer_lower}
        for p in parts:
            pl = p.lower().strip()
            if not pl:
                continue
            if pl in seen:
                continue
            if len(p) < 4:          # quá ngắn
                continue
            # 1 từ viết thường → không phải tên riêng → loại
            if len(p.split()) == 1 and p[0].islower():
                continue
            if len(p) > 80:         # quá dài → chuỗi văn xuôi
                continue
            if answer_lower in pl:  # chứa đáp án đúng bên trong
                continue
            if pl in FILLER:        # từ lấp đầy vô nghĩa
                continue
            seen.add(pl)
            clean.append(p)

        return clean[:3]


def find_evidence(context: str, answer: str) -> tuple[str, bool]:
    for sent in re.split(r"(?<=[.!?])\s+", context):
        if answer in sent:
            return sent.strip(), True
    return context.split(".")[0].strip(), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qg_model",  default="models/vit5_qg")
    ap.add_argument("--dg_model",  default="models/vit5_dg")
    ap.add_argument("--data_dir",  default="data/processed")
    ap.add_argument("--out",       default="data/generated/vit5_ft.jsonl")
    ap.add_argument("--limit",     type=int, default=300)
    args = ap.parse_args()

    test_df = load_test_data(args.data_dir, args.limit)
    gen     = ViT5Generator(args.qg_model, args.dg_model)

    items, skipped = [], 0
    skip_reasons = {"empty_question": 0, "few_distractors": 0, "finalize_fail": 0}

    for i, row in test_df.iterrows():
        ctx    = row.get("context", row.get("context_x", ""))
        answer = row["answer_text"]
        wh     = row["wh_type"]
        cid    = row["context_id"]
        title  = row.get("title", row.get("title_x", ""))
        is_vn  = bool(row.get("is_vietnam", row.get("is_vietnam_x", False)))

        # Stage A: sinh câu hỏi
        question = gen.generate_question(ctx, answer, wh)
        if not question.strip():
            skip_reasons["empty_question"] += 1
            skipped += 1
            if skipped <= 3:
                print(f"  [DEBUG skip] empty question | ans='{answer}' wh={wh}")
            continue
        if not question.endswith("?"):
            question += "?"

        # Stage B: sinh distractor
        distractors = gen.generate_distractors(question, answer, ctx)

        # Fallback: nếu DG sinh thiếu, lấy thêm từ các entity trong context
        if len(distractors) < 3:
            import re
            # trích các cụm danh từ/số ngắn từ context làm distractor phụ
            candidates = re.findall(
                r'\b([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]'
                r'[a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+'
                r'(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]'
                r'[a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+)*)\b',
                ctx
            )
            # thêm năm từ context
            years = re.findall(r'\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b', ctx)
            candidates = list(set(candidates + years))

            FILLER_SET = {"trong", "ngoài", "và", "khi", "để", "vì",
                          "nhưng", "với", "từ", "tại", "của", "là", "có",
                          "không", "letter", "năm", "tháng", "ngày"}
            ans_lower = answer.lower()
            seen_d = {d.lower() for d in distractors} | {ans_lower}
            for c in candidates:
                if len(distractors) >= 3:
                    break
                cl = c.lower().strip()
                if (cl not in seen_d
                        and c.strip()
                        and cl != ans_lower
                        and len(c) >= 4
                        and len(c) <= 60
                        and cl not in FILLER_SET
                        and ans_lower not in cl):
                    distractors.append(c)
                    seen_d.add(cl)

        # Debug 5 câu đầu
        if i < 5:
            print(f"\n  [DEBUG {i}] Q: {question}")
            print(f"             ans: {answer}")
            print(f"             dist: {distractors}")

        # finalize 4 phương án
        try:
            opts, key = finalize_options(
                correct_text=answer,
                distractors=distractors,
                correct_provenance=Provenance.context_span,
                distractor_provenance=Provenance.generated,
                seed=i,
            )
        except ValueError as e:
            skip_reasons["finalize_fail"] += 1
            skipped += 1
            if skipped <= 5:
                print(f"  [DEBUG skip] finalize fail: {e}")
            continue

        ev_sent, found = find_evidence(ctx, answer)
        bloom = WH_BLOOM.get(wh, Bloom.nhan_biet)

        item = MCQItem(
            item_id=make_item_id(cid, question, "vit5_ft"),
            generator=Generator(
                method=Method.vit5_ft,
                variant="vit5_base_stage_ab",
                model_name="VietAI/vit5-base",
            ),
            source=Source(
                context_id=cid,
                context=ctx,
                title=title,
                is_vietnam=is_vn,
            ),
            request=Request(
                bloom_requested=bloom,
                wh_type_requested=WhType(wh),
            ),
            question=question,
            options=opts,
            answer_key=key,
            answer_text=answer,
            evidence=Evidence(
                sentence=ev_sent,
                found_in_context=found,
            ),
            metadata=Metadata(
                wh_type_detected=WhType(wh),
                bloom_predicted=bloom,
                topic=title,
                tags=[wh, "vn" if is_vn else "world"],
            ),
            generation_trace=GenerationTrace(
                cost_usd=0.0,
                n_llm_calls=0,
            ),
        )
        items.append(item)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_df)} | sinh: {len(items)} | bỏ: {skipped}",
                  flush=True)

    print(f"\nHoàn tất: {len(items)} câu | bỏ: {skipped}")
    print(f"Skip reasons: {skip_reasons}")

    # Lưu
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    from collections import Counter
    print(f"\n=== ViT5-FT GENERATOR ===")
    print(f"  Câu hỏi sinh được : {len(items)} / {len(test_df)}")
    print(f"  VQR thô           : {len(items)/max(len(test_df),1):.0%}")
    ev_ok = sum(1 for i in items if i.evidence.found_in_context)
    print(f"  Evidence in ctx   : {ev_ok}/{len(items)} = {ev_ok/max(len(items),1):.0%}")
    print(f"  Chi phí API       : $0.00 (offline)")
    print("\n  Phân bố wh_type:")
    for wh, n in Counter(
        i.request.wh_type_requested for i in items).most_common():
        print(f"    {wh.value:14s}: {n}")
    print(f"\nĐã lưu {len(items)} câu -> {args.out}")

    if items:
        it = items[0]
        print("\n[MẪU]")
        print(f"  Q: {it.question}")
        for o in it.options:
            print(f"  {o.label}. {o.text}{'  ←' if o.is_correct else ''}")


if __name__ == "__main__":
    main()
