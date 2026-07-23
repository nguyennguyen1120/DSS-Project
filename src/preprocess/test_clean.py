"""Test lớp lọc luật trên chính các lỗi thật trong ner_quality_sample.csv."""
import sys
sys.path.insert(0, "src/preprocess")
from clean_entities import clean_entity, clean_entities

# (surface, type, kỳ vọng) — None nghĩa là phải bị loại
CASES = [
    # 3. năm bị gán LOC -> loại
    ("năm 1979", "LOC", None),
    ("năm 1396", "LOC", None),
    ("năm 1923", "LOC", None),
    # 4b. rác / stopword
    ("thứ 2", "LOC", None),
    ("tờ 2", "LOC", None),
    ("Bộ", "ORG", None),
    # 4a. rác ranh giới
    (", Guam", "LOC", ("Guam", "LOC")),
    (", Xóm Than", "LOC", ("Xóm Than", "LOC")),
    # 4c. chứa động từ -> loại
    ("khinh thường Roosevelt", "LOC", None),
    ("liên danh Roosevelt", "LOC", None),
    ("Quốc hội nên Roosevelt", "LOC", None),
    ("liên quân chống Phổ", "LOC", None),
    # 2. chức danh + tên -> bóc, ép PER
    ("chủ tịch Hồ Chí Minh", "LOC", ("Hồ Chí Minh", "PER")),
    ("vua Porus", "LOC", ("Porus", "PER")),
    # 1. danh từ chung + tên -> bóc
    ("dân tộc Nga", "LOC", ("Nga", "LOC")),
    ("người Bồ Đào Nha", "LOC", ("Bồ Đào Nha", "LOC")),
    ("binh sĩ Nam Triều Tiên", "LOC", ("Nam Triều Tiên", "LOC")),
    ("tỉnh Fars", "LOC", ("Fars", "LOC")),
    ("sông Potomac", "LOC", ("Potomac", "LOC")),
    ("thủ đô Washington", "LOC", ("Washington", "LOC")),
    ("bán đảo Triều Tiên", "LOC", ("Triều Tiên", "LOC")),
    # đúng sẵn -> giữ nguyên
    ("Hoa Kỳ", "LOC", ("Hoa Kỳ", "LOC")),
    ("Marcus Aurelius", "PER", ("Marcus Aurelius", "PER")),
    ("Trung Quốc", "LOC", ("Trung Quốc", "LOC")),
    ("Augustus", "PER", ("Augustus", "PER")),
    # YEAR/DATE không đụng tới
    ("1954", "YEAR", ("1954", "YEAR")),
    ("1775-06-14", "DATE", ("1775-06-14", "DATE")),
]

ok = 0
for surface, typ, expect in CASES:
    got = clean_entity(surface, typ)
    status = "✓" if got == expect else "✗"
    if got == expect:
        ok += 1
    else:
        print(f"  {status} '{surface}' ({typ}) -> {got}  (kỳ vọng {expect})")
print(f"\n{ok}/{len(CASES)} case đúng")
assert ok == len(CASES), "Có case sai — xem ở trên"

# thống kê cải thiện trên toàn mẫu 100 dòng
import csv
rows = list(csv.DictReader(open('/mnt/user-data/uploads/ner_quality_sample.csv',
                                encoding='utf-8-sig')))
ents = [{"surface": r["surface"], "type": r["type"],
         "normalized": r["normalized"]} for r in rows]
cleaned = clean_entities(ents)
print(f"\nTrước lọc : {len(ents)} thực thể")
print(f"Sau lọc   : {len(cleaned)} thực thể (bỏ {len(ents)-len(cleaned)} rác)")
from collections import Counter
print("Type sau lọc:", dict(Counter(e["type"] for e in cleaned)))

print("\n✅ TẤT CẢ TEST PASS — lớp lọc luật bắt đúng 4 khuôn mẫu lỗi.")
