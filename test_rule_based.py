"""Test rule-based generator end-to-end (không cần file parquet thật)."""
import sys
sys.path.insert(0, "src/schema")
sys.path.insert(0, "src/preprocess")
sys.path.insert(0, "src/generators")

from mcq_schema import WhType, Bloom, MCQItem
from rule_based import (generate_one, select_answer, get_distractors,
                        find_evidence, build_question, _parse_century)

# ---- Mock entities cho đoạn Điện Biên Phủ ----
CTX = ("Chiến thắng Điện Biên Phủ diễn ra năm 1954, kết thúc chín năm kháng "
       "chiến chống thực dân Pháp. Chiến dịch do Đại tướng Võ Nguyên Giáp "
       "chỉ huy tại lòng chảo Điện Biên Phủ, tỉnh Lai Châu.")
TITLE = "Chiến dịch Điện Biên Phủ"
CID = "ctx_test_001"

ENTS = [
    {"surface": "1954", "type": "YEAR", "normalized": "1954",
     "bucket": "20C", "char_start": CTX.index("1954"), "char_end": CTX.index("1954")+4},
    {"surface": "Võ Nguyên Giáp", "type": "PER", "normalized": "võ_nguyên_giáp",
     "bucket": "VN", "char_start": CTX.index("Võ Nguyên Giáp"),
     "char_end": CTX.index("Võ Nguyên Giáp")+14},
    {"surface": "Điện Biên Phủ", "type": "LOC", "normalized": "điện_biên_phủ",
     "bucket": "VN", "char_start": CTX.index("Điện Biên Phủ"),
     "char_end": CTX.index("Điện Biên Phủ")+13},
]

POOL = {
    ("YEAR", "20C"): {"1945", "1954", "1975", "1986", "1960"},
    ("PER", "VN"):   {"Hồ Chí Minh", "Ngô Đình Diệm", "Trần Hưng Đạo", "Nguyễn Huệ"},
    ("LOC", "VN"):   {"Hà Nội", "Sài Gòn", "Huế", "Đà Nẵng", "Điện Biên Phủ"},
    ("LOC", "TG"):   {"Paris", "Washington", "London", "Beijing", "Moscow"},
}

import random
rng = random.Random(42)

# ---------- 1. answer selection ----------
ans_year = select_answer(CTX, WhType.thoi_gian, ENTS, rng)
assert ans_year is not None and ans_year["type"] == "YEAR", ans_year
print("✓ answer thoi_gian:", ans_year["surface"])

ans_per = select_answer(CTX, WhType.nhan_vat, ENTS, rng)
assert ans_per is not None and ans_per["type"] == "PER", ans_per
print("✓ answer nhan_vat:", ans_per["surface"])

ans_loc = select_answer(CTX, WhType.dia_diem, ENTS, rng)
assert ans_loc is not None and ans_loc["type"] == "LOC", ans_loc
print("✓ answer dia_diem:", ans_loc["surface"])

# non-factoid trả None
ans_nf = select_answer(CTX, WhType.nguyen_nhan, ENTS, rng)
assert ans_nf is None
print("✓ nguyen_nhan -> None (rule-based bỏ qua)")

# ---------- 2. evidence ----------
ev, found = find_evidence(CTX, "1954")
assert found and "1954" in ev
print(f"✓ evidence found: '{ev[:50]}...'")

# ---------- 3. distractor ----------
distractors = get_distractors("1954", "YEAR", "20C", POOL, rng, n=3)
assert len(distractors) == 3
assert "1954" not in distractors
assert len(set(distractors)) == 3
print("✓ distractor YEAR:", distractors)

# fallback: bucket hiếm -> vẫn lấy được từ FAMOUS_YEARS
dist_rare = get_distractors("938", "YEAR", "10C", POOL, rng, n=3)
assert len(dist_rare) == 3
print("✓ fallback YEAR bucket hiếm:", dist_rare)

# ---------- 4. generate_one - thoi_gian ----------
item = generate_one(
    context=CTX, title=TITLE, context_id=CID, is_vietnam=True,
    wh_type=WhType.thoi_gian, bloom=Bloom.nhan_biet,
    entities=ENTS, pool=POOL, seed=42,
)
assert item is not None
assert isinstance(item, MCQItem)
assert item.answer_text == "1954"
assert item.evidence.found_in_context is True
assert item.generator.method.value == "rule_based"
assert item.request.wh_type_requested == WhType.thoi_gian
opts_texts = [o.text for o in item.options]
assert "1954" in opts_texts
assert len(set(opts_texts)) == 4
print(f"✓ MCQItem thoi_gian: Q='{item.question}'")
print(f"  Options: {opts_texts}  key={item.answer_key}")

# ---------- 5. generate_one - nhan_vat ----------
item2 = generate_one(
    context=CTX, title=TITLE, context_id=CID, is_vietnam=True,
    wh_type=WhType.nhan_vat, bloom=Bloom.nhan_biet,
    entities=ENTS, pool=POOL, seed=7,
)
assert item2 is not None
assert item2.answer_text == "Võ Nguyên Giáp"
print(f"✓ MCQItem nhan_vat: Q='{item2.question}'")

# ---------- 6. MCQItem schema invariants đều đúng ----------
from pydantic import ValidationError
# kiểm tra tính nhất quán schema (validator không nổ)
assert item.answer_key in ["A","B","C","D"]
correct_opts = [o for o in item.options if o.is_correct]
assert len(correct_opts) == 1
assert correct_opts[0].text == item.answer_text
print("✓ schema invariants giữ vững")

# ---------- 7. non-factoid trả None ----------
item_nf = generate_one(
    context=CTX, title=TITLE, context_id=CID, is_vietnam=True,
    wh_type=WhType.nguyen_nhan, bloom=Bloom.van_dung,
    entities=ENTS, pool=POOL, seed=0,
)
assert item_nf is None
print("✓ non-factoid (nguyen_nhan) -> None")

# ---------- 8. _parse_century ----------
assert _parse_century("20C") == 20
assert _parse_century("10C") == 10
assert _parse_century("BC") == 20  # fallback
print("✓ _parse_century")

# ---------- 9. pool LOC thiếu -> fallback TG ----------
pool_vn_only = {
    ("YEAR","20C"): {"1945","1975","1986"},
    ("LOC","VN"):   {"Hà Nội","Sài Gòn","Huế","Đà Nẵng"},
    ("LOC","TG"):   {"Paris","London","Washington"},
}
item3 = generate_one(
    context=CTX, title=TITLE, context_id=CID, is_vietnam=True,
    wh_type=WhType.dia_diem, bloom=Bloom.nhan_biet,
    entities=ENTS, pool=pool_vn_only, seed=1,
)
# "Điện Biên Phủ" ở bucket VN; pool VN có 4 item, trừ đáp án còn 3 -> đủ
assert item3 is not None
print("✓ LOC distractor từ pool VN:", [o.text for o in item3.options])

print("\n✅ TẤT CẢ TEST PASS — rule-based generator đúng đầu-cuối.")
