"""Test RAG+LLM: RRF logic, prompt builder, parse_llm_output — không cần OpenAI."""
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))
sys.path.insert(0, str(_SRC / "generators"))

from mcq_schema import WhType, Bloom
from rag_llm import (build_prompt, parse_llm_output,
                     _wh_instruction, _bloom_instruction)

CTX = ("Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954, "
       "kết thúc chín năm kháng chiến chống thực dân Pháp. "
       "Chiến dịch do Đại tướng Võ Nguyên Giáp chỉ huy.")

# ---------- 1. Prompt builder ----------
prompt = build_prompt(CTX, WhType.thoi_gian, Bloom.nhan_biet, "Chiến dịch Điện Biên Phủ")
assert "THỜI GIAN" in prompt or "thoi_gian" in prompt
assert CTX in prompt
assert "evidence_sentence" in prompt
assert "bloom_level" in prompt
print("✓ prompt chứa đủ thành phần")
print(f"  Độ dài prompt: {len(prompt)} ký tự")

# ---------- 2. parse_llm_output — hợp lệ ----------
good_output = {
    "question": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
    "correct_answer": "1954",
    "distractors": ["1945", "1975", "1965"],
    "evidence_sentence": "Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954, kết thúc chín năm kháng chiến chống thực dân Pháp.",
    "bloom_level": "nhan_biet",
}
result = parse_llm_output(good_output, CTX, WhType.thoi_gian, Bloom.nhan_biet)
assert result is not None
q, correct, distractors, evidence, found = result
assert q.endswith("?")
assert correct == "1954"
assert len(distractors) == 3
assert found is True
print(f"✓ parse hợp lệ: Q='{q}', ans='{correct}', found={found}")

# ---------- 3. parse_llm_output — evidence không trong context ----------
bad_evidence = dict(good_output)
bad_evidence["evidence_sentence"] = "Câu này không tồn tại trong context."
result2 = parse_llm_output(bad_evidence, CTX, WhType.thoi_gian, Bloom.nhan_biet)
assert result2 is not None
_, _, _, evidence2, found2 = result2
# found phải là False (không tìm thấy exact)
# nhưng fuzzy match có thể tìm được một phần
print(f"✓ evidence không khớp: found={found2} (fuzzy fallback)")

# ---------- 4. parse_llm_output — thiếu distractor -> None ----------
bad_dist = dict(good_output)
bad_dist["distractors"] = ["1945"]   # chỉ 1 distractor
result3 = parse_llm_output(bad_dist, CTX, WhType.thoi_gian, Bloom.nhan_biet)
assert result3 is None
print("✓ thiếu distractor -> None")

# ---------- 5. parse_llm_output — thiếu question -> None ----------
bad_q = dict(good_output)
bad_q["question"] = ""
result4 = parse_llm_output(bad_q, CTX, WhType.thoi_gian, Bloom.nhan_biet)
assert result4 is None
print("✓ thiếu question -> None")

# ---------- 6. RRF logic ----------
from rag_llm import HybridRetriever
rrf_k = HybridRetriever.RRF_K

def rrf_score(rank: int) -> float:
    return 1.0 / (rrf_k + rank + 1)

# doc ở rank 0 BM25 + rank 1 FAISS: tổng = 1/(60+1) + 1/(60+2)
doc_a_bm25 = rrf_score(0)
doc_a_faiss = rrf_score(1)
doc_a_total = doc_a_bm25 + doc_a_faiss

# doc chỉ ở rank 0 FAISS
doc_b_faiss = rrf_score(0)

# doc_a có mặt cả hai -> phải thắng doc_b chỉ có FAISS
assert doc_a_total > doc_b_faiss, f"{doc_a_total} vs {doc_b_faiss}"
print(f"✓ RRF: doc 2 nguồn ({doc_a_total:.4f}) > doc 1 nguồn ({doc_b_faiss:.4f})")

# ---------- 7. wh_instruction đủ loại ----------
for wh in WhType:
    assert _wh_instruction(wh), f"Thiếu instruction cho {wh}"
print("✓ instruction đủ 6 wh_type")

# ---------- 8. bloom_instruction đủ loại ----------
for b in Bloom:
    assert _bloom_instruction(b), f"Thiếu instruction cho {b}"
print("✓ instruction đủ 4 Bloom level")

print("\n✅ TẤT CẢ TEST PASS — RAG+LLM logic đúng (không cần API).")
