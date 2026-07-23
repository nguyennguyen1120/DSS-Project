"""
RAG+LLM Generator — Phương pháp 2.

Pipeline:
  1. Hybrid retrieval: BM25 + FAISS -> Reciprocal Rank Fusion -> top-k context
  2. Prompt GPT-4o-mini với few-shot + JSON schema, ép trả evidence_sentence
  3. Validate output, retry nếu JSON lỗi (tối đa 2 lần)
  4. Trả MCQItem qua schema chung

Điểm mạnh: sinh được Nguyên nhân/Ý nghĩa (non-factoid), distractor thông minh,
đa dạng câu hỏi. Điểm yếu: tốn tiền API, có thể hallucinate nếu context thiếu.

Cài:
    pip install openai rank-bm25 faiss-cpu sentence-transformers
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))

from mcq_schema import (
    MCQItem, Generator, Source, Request, Evidence,
    Metadata, GenerationTrace, Provenance,
    Method, Bloom, WhType, finalize_options, make_item_id,
)

# ============================ Retriever =====================================

class HybridRetriever:
    """BM25 + FAISS kết hợp bằng Reciprocal Rank Fusion (RRF)."""

    RRF_K = 60   # hằng số RRF chuẩn

    def __init__(self, index_dir: str = "data/index",
                 embed_model: str = "bkai-foundation-models/vietnamese-bi-encoder"):
        import faiss
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        idx = Path(index_dir)
        self.faiss_index = faiss.read_index(str(idx / "faiss.index"))
        with open(idx / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.meta = pd.read_parquet(idx / "index_meta.parquet")
        self.texts = self.meta["context"].tolist()
        self.embed_model = SentenceTransformer(embed_model)
        print(f"[RAG] Index loaded: {len(self.texts)} đoạn")

    def _bm25_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        idx_scores = sorted(enumerate(scores), key=lambda x: -x[1])
        return idx_scores[:top_k]

    def _faiss_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        import numpy as np
        emb = self.embed_model.encode(
            [query], normalize_embeddings=True).astype("float32")
        scores, indices = self.faiss_index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid RRF. Trả list dict {context_id, title, context, score}."""
        fetch_k = top_k * 3
        bm25_results = self._bm25_scores(query, fetch_k)
        faiss_results = self._faiss_scores(query, fetch_k)

        # RRF: score = Σ 1/(k + rank)
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(faiss_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        top = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in top:
            row = self.meta.iloc[idx]
            results.append({
                "context_id": row["context_id"],
                "title": row["title"],
                "context": row["context"],
                "is_vietnam": bool(row.get("is_vietnam", False)),
                "rrf_score": score,
            })
        return results


# ============================ Prompt ========================================

# Few-shot examples — mỗi wh_type một example để guide LLM
FEW_SHOT = {
    WhType.thoi_gian: {
        "context": "Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954, kết thúc cuộc kháng chiến chống Pháp.",
        "output": {
            "question": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
            "correct_answer": "1954",
            "distractors": ["1945", "1975", "1965"],
            "evidence_sentence": "Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954, kết thúc cuộc kháng chiến chống Pháp.",
            "bloom_level": "nhan_biet",
        }
    },
    WhType.nhan_vat: {
        "context": "Đại tướng Võ Nguyên Giáp là người chỉ huy chiến dịch Điện Biên Phủ năm 1954.",
        "output": {
            "question": "Ai là người chỉ huy chiến dịch Điện Biên Phủ?",
            "correct_answer": "Võ Nguyên Giáp",
            "distractors": ["Hồ Chí Minh", "Nguyễn Chí Thanh", "Lê Duẩn"],
            "evidence_sentence": "Đại tướng Võ Nguyên Giáp là người chỉ huy chiến dịch Điện Biên Phủ năm 1954.",
            "bloom_level": "nhan_biet",
        }
    },
    WhType.nguyen_nhan: {
        "context": "Đế quốc La Mã sụp đổ năm 476 do áp lực từ các bộ tộc Germanic, suy yếu kinh tế và bất ổn chính trị kéo dài.",
        "output": {
            "question": "Nguyên nhân nào dẫn đến sự sụp đổ của Đế quốc La Mã?",
            "correct_answer": "Áp lực từ các bộ tộc Germanic, suy yếu kinh tế và bất ổn chính trị",
            "distractors": [
                "Bị Đế quốc Ottoman xâm lược",
                "Thiên tai và dịch bệnh quy mô lớn",
                "Nội chiến giữa các tướng lĩnh La Mã",
            ],
            "evidence_sentence": "Đế quốc La Mã sụp đổ năm 476 do áp lực từ các bộ tộc Germanic, suy yếu kinh tế và bất ổn chính trị kéo dài.",
            "bloom_level": "thong_hieu",
        }
    },
}

# Fallback example cho các wh_type chưa có few-shot
_DEFAULT_EXAMPLE = FEW_SHOT[WhType.thoi_gian]


def _get_example(wh_type: WhType) -> dict:
    return FEW_SHOT.get(wh_type, _DEFAULT_EXAMPLE)


def _wh_instruction(wh_type: WhType) -> str:
    instructions = {
        WhType.thoi_gian:   "Câu hỏi phải hỏi về MỐC THỜI GIAN (năm, ngày, thế kỷ).",
        WhType.nhan_vat:    "Câu hỏi phải hỏi về NHÂN VẬT (ai, người nào).",
        WhType.dia_diem:    "Câu hỏi phải hỏi về ĐỊA ĐIỂM (ở đâu, tại đâu).",
        WhType.su_kien:     "Câu hỏi phải hỏi về SỰ KIỆN (điều gì xảy ra).",
        WhType.nguyen_nhan: "Câu hỏi phải hỏi về NGUYÊN NHÂN (vì sao, do đâu).",
        WhType.y_nghia:     "Câu hỏi phải hỏi về Ý NGHĨA hoặc KẾT QUẢ (dẫn đến điều gì, có ý nghĩa gì).",
    }
    return instructions.get(wh_type, "")


def _bloom_instruction(bloom: Bloom) -> str:
    instructions = {
        Bloom.nhan_biet:   "Độ khó: NHẬN BIẾT — câu hỏi trực tiếp, đáp án có trong văn bản.",
        Bloom.thong_hieu:  "Độ khó: THÔNG HIỂU — câu hỏi yêu cầu diễn giải, không chỉ chép lại.",
        Bloom.van_dung:    "Độ khó: VẬN DỤNG — câu hỏi yêu cầu liên kết nhiều thông tin.",
        Bloom.van_dung_cao:"Độ khó: VẬN DỤNG CAO — câu hỏi phân tích, đánh giá.",
    }
    return instructions.get(bloom, "")


def build_prompt(
    context: str,
    wh_type: WhType,
    bloom: Bloom,
    title: str = "",
) -> str:
    example = _get_example(wh_type)
    example_json = json.dumps(example["output"], ensure_ascii=False, indent=2)

    return f"""Bạn là chuyên gia tạo câu hỏi trắc nghiệm lịch sử giáo dục.
Nhiệm vụ: Tạo 1 câu hỏi trắc nghiệm 4 phương án từ đoạn văn dưới đây.

QUY TẮC BẮT BUỘC:
1. {_wh_instruction(wh_type)}
2. {_bloom_instruction(bloom)}
3. Đáp án đúng và evidence_sentence PHẢI được trích xuất NGUYÊN VĂN từ đoạn văn.
4. 3 phương án nhiễu phải cùng loại với đáp án đúng nhưng SAI về mặt lịch sử.
5. Chỉ được dùng thông tin trong đoạn văn. KHÔNG suy diễn thêm.
6. Trả về JSON hợp lệ, KHÔNG thêm text bên ngoài JSON.

VÍ DỤ (wh_type={wh_type.value}):
Ngữ cảnh: {example['context']}
Output:
{example_json}

---
BÀI (tiêu đề: {title or 'không có'}):
{context}

Trả về JSON với đúng các trường sau:
{{
  "question": "câu hỏi",
  "correct_answer": "đáp án đúng (nguyên văn từ đoạn văn)",
  "distractors": ["nhiễu 1", "nhiễu 2", "nhiễu 3"],
  "evidence_sentence": "câu chứa bằng chứng (nguyên văn từ đoạn văn)",
  "bloom_level": "{bloom.value}"
}}"""


# ============================ LLM call ======================================

def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
    temperature: float = 0.3,
) -> tuple[Optional[dict], float, int]:
    """Gọi OpenAI, retry nếu JSON lỗi. Trả (parsed_dict, cost_usd, n_calls)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    cost = 0.0
    calls = 0
    last_err = None

    # pricing GPT-4o-mini (per 1M token, tháng 7/2025)
    PRICE_IN  = 0.15 / 1_000_000
    PRICE_OUT = 0.60 / 1_000_000

    for attempt in range(max_retries + 1):
        calls += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=600,
            )
            usage = resp.usage
            cost += usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT
            raw = resp.choices[0].message.content or ""
            # bóc ```json ... ``` nếu có
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            parsed = json.loads(raw)
            return parsed, cost, calls
        except Exception as e:
            last_err = e
            print(f"\n    [LỖI lần {attempt+1}] {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)   # exponential backoff

    return None, cost, calls


# ============================ Parse LLM output ==============================

def parse_llm_output(
    data: dict,
    context: str,
    wh_type: WhType,
    bloom: Bloom,
) -> Optional[tuple[str, str, list[str], str, bool]]:
    """Validate và trích xuất (question, correct_answer, distractors, evidence, found).
    Trả None nếu output không hợp lệ.
    """
    question = str(data.get("question", "")).strip()
    correct = str(data.get("correct_answer", "")).strip()
    distractors = [str(d).strip() for d in data.get("distractors", []) if str(d).strip()]
    evidence = str(data.get("evidence_sentence", "")).strip()

    # kiểm tra cơ bản
    if not question or not correct or len(distractors) < 3:
        return None
    if not question.endswith("?"):
        question += "?"

    # kiểm tra evidence có trong context (Evidence Match thô — Verifier sẽ kiểm kỹ hơn)
    found = evidence in context
    if not found:
        # fuzzy: thử tìm chuỗi con dài nhất
        words = evidence.split()
        for n in range(len(words), max(3, len(words)//2), -1):
            chunk = " ".join(words[:n])
            if chunk in context:
                evidence = chunk
                found = True
                break

    return question, correct, distractors[:3], evidence, found


# ============================ Generator chính ===============================

def generate_one(
    topic: str,
    wh_type: WhType,
    bloom: Bloom,
    retriever: HybridRetriever,
    top_k: int = 3,
    llm_model: str = "gpt-4o-mini",
    seed: Optional[int] = None,
) -> Optional[tuple[list[MCQItem], float, int]]:
    """Sinh câu hỏi cho một topic/wh_type.
    Retrieve top_k context, chọn context tốt nhất (rank 1), gọi LLM.
    Trả (items, total_cost, total_calls) hoặc None nếu thất bại.
    """
    import random
    rng = random.Random(seed)

    # 1. Retrieve
    query = f"{topic} {wh_type.value.replace('_', ' ')}"
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return None

    items = []
    total_cost, total_calls = 0.0, 0

    for res in results:
        context   = res["context"]
        ctx_id    = res["context_id"]
        title     = res["title"]
        is_vn     = res["is_vietnam"]

        # 2. Build prompt + gọi LLM
        t0 = time.time()
        prompt = build_prompt(context, wh_type, bloom, title)
        parsed, cost, calls = call_openai(prompt, model=llm_model)
        latency_ms = (time.time() - t0) * 1000
        total_cost += cost
        total_calls += calls

        if parsed is None:
            continue

        # 3. Parse + validate output
        result = parse_llm_output(parsed, context, wh_type, bloom)
        if result is None:
            continue

        question, correct, distractors, evidence, found = result

        # 4. finalize_options
        try:
            opts, answer_key = finalize_options(
                correct_text=correct,
                distractors=distractors,
                correct_provenance=Provenance.context_span,
                distractor_provenance=Provenance.generated,
                seed=rng.randint(0, 9999),
            )
        except ValueError:
            continue

        # 5. Build MCQItem
        item = MCQItem(
            item_id=make_item_id(ctx_id, question, "rag_llm"),
            generator=Generator(
                method=Method.rag_llm,
                variant="hybrid_rrf_gpt4o_mini",
                model_name=llm_model,
            ),
            source=Source(
                context_id=ctx_id,
                context=context,
                title=title,
                is_vietnam=is_vn,
            ),
            request=Request(
                bloom_requested=bloom,
                wh_type_requested=wh_type,
            ),
            question=question,
            options=opts,
            answer_key=answer_key,
            answer_text=correct,
            evidence=Evidence(
                sentence=evidence,
                found_in_context=found,
            ),
            metadata=Metadata(
                wh_type_detected=wh_type,
                bloom_predicted=Bloom(parsed.get("bloom_level", bloom.value))
                    if parsed.get("bloom_level") in [b.value for b in Bloom] else bloom,
                topic=title,
            ),
            generation_trace=GenerationTrace(
                latency_ms=latency_ms,
                cost_usd=cost,
                n_llm_calls=calls,
                retrieved_context_ids=[r["context_id"] for r in results],
            ),
        )
        items.append(item)

    return (items, total_cost, total_calls) if items else None


# ============================ Batch runner ==================================

def generate_batch(
    topics: list[str],
    wh_types: list[WhType],
    bloom: Bloom,
    retriever: HybridRetriever,
    llm_model: str = "gpt-4o-mini",
    top_k: int = 3,
    seed: int = 42,
) -> tuple[list[MCQItem], float, int]:
    import random
    rng = random.Random(seed)
    all_items, total_cost, total_calls = [], 0.0, 0

    for topic in topics:
        for wh in wh_types:
            result = generate_one(
                topic=topic, wh_type=wh, bloom=bloom,
                retriever=retriever, top_k=top_k,
                llm_model=llm_model,
                seed=rng.randint(0, 999999),
            )
            if result:
                items, cost, calls = result
                all_items.extend(items)
                total_cost += cost
                total_calls += calls

    return all_items, total_cost, total_calls


# ============================ CLI ===========================================

def main():
    import argparse, pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--corpus",    default="data/processed/corpus.parquet")
    ap.add_argument("--outdir",    default="data/generated")
    ap.add_argument("--topics",    default="",
                    help="Danh sách topic cách nhau bằng |. Nếu rỗng: lấy ngẫu nhiên từ corpus")
    ap.add_argument("--n_topics",  type=int, default=20,
                    help="Số topic ngẫu nhiên nếu --topics rỗng")
    ap.add_argument("--wh_types",  default="thoi_gian,nhan_vat,dia_diem,nguyen_nhan")
    ap.add_argument("--bloom",     default="nhan_biet")
    ap.add_argument("--top_k",     type=int, default=3)
    ap.add_argument("--model",     default="gpt-4o-mini")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    retriever = HybridRetriever(args.index_dir)

    # lấy topics
    if args.topics:
        topics = [t.strip() for t in args.topics.split("|")]
    else:
        df = pd.read_parquet(args.corpus)
        titles = df[df["split"].isin({"train","dev"})]["title"].unique().tolist()
        import random; random.seed(args.seed)
        topics = random.sample(titles, min(args.n_topics, len(titles)))
        print(f"Topics ngẫu nhiên ({len(topics)}): {topics[:5]}...")

    wh_types = [WhType(w.strip()) for w in args.wh_types.split(",")]
    bloom    = Bloom(args.bloom)

    print(f"\nSinh câu hỏi: {len(topics)} topic × {len(wh_types)} wh_type × top_k={args.top_k}")
    print(f"Model: {args.model} | Bloom: {bloom.value}")
    total_planned = len(topics) * len(wh_types)
    print(f"Tổng lượt gọi LLM tối đa: {total_planned * args.top_k} lần")
    print(f"Ước tính chi phí tối đa: ${total_planned * args.top_k * 0.0003:.3f}")
    print("Bắt đầu...\n")

    import random
    rng = random.Random(args.seed)
    all_items, total_cost, total_calls = [], 0.0, 0
    done = 0

    for topic in topics:
        for wh in wh_types:
            done += 1
            print(f"  [{done:3d}/{total_planned}] topic='{topic[:30]}' wh={wh.value}...",
                  end=" ", flush=True)
            result = generate_one(
                topic=topic, wh_type=wh, bloom=bloom,
                retriever=retriever, top_k=args.top_k,
                llm_model=args.model,
                seed=rng.randint(0, 999999),
            )
            if result:
                items, cost, calls = result
                all_items.extend(items)
                total_cost += cost
                total_calls += calls
                print(f"✓ {len(items)} câu | ${cost:.4f}")
            else:
                print("✗ thất bại")

    # lưu
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.outdir) / "rag_llm.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(item.model_dump_json() + "\n")

    tried = len(topics) * len(wh_types) * args.top_k
    print(f"\n=== RAG+LLM GENERATOR ===")
    print(f"  Câu hỏi sinh được : {len(all_items)} / {tried} (VQR thô: {len(all_items)/max(tried,1):.0%})")
    print(f"  Tổng chi phí API  : ${total_cost:.4f}")
    print(f"  Tổng LLM calls    : {total_calls}")
    ev_ok = sum(1 for i in all_items if i.evidence.found_in_context)
    print(f"  Evidence in ctx   : {ev_ok}/{len(all_items)} = {ev_ok/max(len(all_items),1):.0%}")
    from collections import Counter
    by_wh = Counter(i.request.wh_type_requested for i in all_items)
    print("\n  Phân bố wh_type:")
    for wh, n in sorted(by_wh.items(), key=lambda x: -x[1]):
        print(f"    {wh.value:14s}: {n}")
    print(f"\nĐã lưu {len(all_items)} câu hỏi -> {out_path}")

    if all_items:
        i = all_items[0]
        print("\n[MẪU ĐẦU RA]")
        print(f"  Q: {i.question}")
        for o in i.options:
            print(f"  {o.label}. {o.text}{' ←' if o.is_correct else ''}")
        print(f"  Evidence: {i.evidence.sentence[:80]}...")
        print(f"  Cost: ${i.generation_trace.cost_usd:.5f} | "
              f"Latency: {i.generation_trace.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
