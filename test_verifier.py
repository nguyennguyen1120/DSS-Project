"""Test verifier logic — không cần OpenAI API."""
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC / "schema"))
sys.path.insert(0, str(_SRC / "preprocess"))
sys.path.insert(0, str(_SRC / "verifier"))

from mcq_schema import (MCQItem, Generator, Source, Request, Evidence,
                         Metadata, GenerationTrace, Provenance,
                         Method, Bloom, WhType, finalize_options, make_item_id)
from verifier import (check_question_clarity, check_distractor_type,
                      check_historical_correctness, check_answer_position,
                      compute_verdict, Check, WEIGHTS)

CTX = ("Chiến thắng Điện Biên Phủ diễn ra năm 1954, kết thúc chín năm kháng "
       "chiến chống thực dân Pháp. Chiến dịch do Đại tướng Võ Nguyên Giáp chỉ huy.")

def make_item(question="Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
              answer="1954", distractors=["1945","1975","1986"]):
    opts, key = finalize_options(answer, distractors,
                                  correct_entity_type="YEAR",
                                  distractor_provenance=Provenance.kb, seed=0)
    return MCQItem(
        item_id=make_item_id("ctx_1", question, "rule_based"),
        generator=Generator(method=Method.rule_based),
        source=Source(context_id="ctx_1", context=CTX, is_vietnam=True),
        request=Request(bloom_requested=Bloom.nhan_biet,
                        wh_type_requested=WhType.thoi_gian),
        question=question, options=opts, answer_key=key,
        answer_text=answer,
        evidence=Evidence(sentence="Chiến thắng Điện Biên Phủ diễn ra năm 1954",
                          found_in_context=True),
    )

# ── 1. Question Clarity ──
item = make_item()
c = check_question_clarity(item)
assert c.passed and c.score >= 0.8, f"score={c.score}"
print(f"✓ clarity OK: score={c.score} reason='{c.reason}'")

item_short = make_item(question="Năm?")
c2 = check_question_clarity(item_short)
assert not c2.passed, f"câu quá ngắn phải fail"
print(f"✓ clarity FAIL (quá ngắn): score={c2.score}")

item_amb = make_item(question="Điều này diễn ra vào năm nào?")
c3 = check_question_clarity(item_amb)
assert c3.score < 0.9, "đại từ mơ hồ phải bị trừ điểm"
print(f"✓ clarity ambiguous pronoun: score={c3.score}")

# ── 2. Distractor Type Match ──
item = make_item()
c = check_distractor_type(item)
assert c.score > 0, f"score={c.score}"
print(f"✓ distractor type: score={c.score} reason='{c.reason}'")

# ── 3. Historical Correctness ──
kb = {"ctx_1": {"1954", "Điện Biên Phủ", "Võ Nguyên Giáp"}}
item = make_item()
c = check_historical_correctness(item, kb)
assert c.passed and c.score >= 0.85, f"score={c.score}"
print(f"✓ historical correct: score={c.score}")

# answer không trong KB → unverified, không fail
item_wrong = make_item(answer="1955")
c2 = check_historical_correctness(item_wrong, kb)
assert c2.passed, "không trong KB → unverified không phải fail"
assert c2.score < 1.0
print(f"✓ historical unverified: score={c2.score} reason='{c2.reason}'")

# ── 4. Answer Position Balance ──
pos_counts = {}
for key in ["A","B","C","D","A","B","C","D"]:  # phân bố đều
    item_pos = make_item()
    object.__setattr__(item_pos, "answer_key", key)
    c = check_answer_position(item_pos, pos_counts)
print(f"✓ position balance: score={c.score:.3f} dist={pos_counts}")

# ── 5. compute_verdict ──
checks_good = {
    "evidence_match":         Check(score=0.95, passed=True,  reason="entailment"),
    "single_correct":         Check(score=1.00, passed=True,  reason="1/4"),
    "distractor_type_match":  Check(score=0.90, passed=True,  reason="type_match=3/3"),
    "historical_correctness": Check(score=1.00, passed=True,  reason="KB match"),
    "question_clarity":       Check(score=0.90, passed=True,  reason="OK"),
    "duplicate_check":        Check(score=0.95, passed=True,  reason="sim=0.05"),
    "bloom_fidelity":         Check(score=1.00, passed=True,  reason="exact"),
    "answer_position_ok":     Check(score=0.80, passed=True,  reason="ok"),
}
score, status, violations = compute_verdict(checks_good)
assert status.value == "accepted", f"status={status}"
assert violations == []
print(f"✓ verdict accepted: score={score}")

checks_bad = {k: Check(score=0.2, passed=False, reason="fail")
              for k in WEIGHTS}
score2, status2, violations2 = compute_verdict(checks_bad)
assert status2.value == "rejected"
assert len(violations2) == 8
print(f"✓ verdict rejected: score={score2} violations={len(violations2)}")

# ── 6. Weights sum ──
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6
print(f"✓ weights sum=1.0")

print("\n✅ TẤT CẢ TEST PASS — verifier logic đúng (không cần API).")
