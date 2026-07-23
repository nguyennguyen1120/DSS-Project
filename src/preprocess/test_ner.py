"""Test offline logic ner_kb (giả lập underthesea, không cần cài)."""
import sys
sys.path.insert(0, "src/preprocess")
from ner_kb import (extract_temporal, extract_named, _merge_iob,
                    year_bucket, process_corpus, normalize_vi)

# ---------- Test regex thời gian ----------
t1 = "Chiến thắng Điện Biên Phủ diễn ra năm 1954. Ngô Quyền thắng năm 938."
te = extract_temporal(t1)
years = {e["normalized"] for e in te if e["type"] == "YEAR"}
assert "1954" in years and "938" in years, years
print("✓ regex năm (kể cả 3 chữ số):", sorted(years))

t2 = "Nền cộng hòa La Mã sụp đổ năm 27 trước Công nguyên dưới thời Augustus."
te2 = extract_temporal(t2)
bc = [e for e in te2 if e["type"] == "YEAR" and e["normalized"] == "-27"]
assert bc, [e["normalized"] for e in te2]
print("✓ năm TCN thành âm:", bc[0]["normalized"])

t3 = "Cách mạng nổ ra ngày 2 tháng 9 năm 1945 tại Hà Nội."
te3 = extract_temporal(t3)
dates = [e for e in te3 if e["type"] == "DATE"]
assert any(e["normalized"] == "1945-09-02" for e in dates), [e["normalized"] for e in te3]
print("✓ ngày đầy đủ:", dates[0]["normalized"])

t4 = "Sự kiện diễn ra vào thế kỉ XIX ở châu Âu."
te4 = extract_temporal(t4)
assert any(e["type"] == "CENTURY" for e in te4)
print("✓ thế kỷ:", [e["surface"] for e in te4 if e["type"] == "CENTURY"])

# ---------- Test gộp IOB ----------
tagged = [("Đại", "N", "B-NP", "O"), ("tướng", "N", "I-NP", "O"),
          ("Võ", "Np", "B-NP", "B-PER"), ("Nguyên", "Np", "B-NP", "I-PER"),
          ("Giáp", "Np", "B-NP", "I-PER"), ("chỉ", "V", "B-VP", "O"),
          ("Điện", "Np", "B-NP", "B-LOC"), ("Biên", "Np", "B-NP", "I-LOC")]
merged = _merge_iob(tagged)
assert ("Võ Nguyên Giáp", "PER") in merged, merged
assert ("Điện Biên", "LOC") in merged, merged
print("✓ gộp IOB:", merged)

# ---------- Test extract_named + tách chức danh + offset ----------
def fake_sent(text):   # tách câu giả
    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def fake_ner(sent):    # NER giả: đánh dấu vài tên đã biết
    import re
    tokens = sent.replace(".", " .").split()
    out = []
    known_per = {"Võ", "Nguyên", "Giáp", "Ngô", "Quyền"}
    known_loc = {"Nam", "Việt"}
    i = 0
    for w in tokens:
        clean = w.strip(".,")
        if clean in known_per:
            prev = out[-1][-1] if out else "O"
            tag = "I-PER" if prev in ("B-PER", "I-PER") else "B-PER"
        elif clean == "Việt":
            tag = "B-LOC"
        elif clean == "Nam" and out and out[-1][-1] in ("B-LOC", "I-LOC"):
            tag = "I-LOC"
        else:
            tag = "O"
        out.append((w, "Np", "B-NP", tag))
    return out

txt = "Đại tướng Võ Nguyên Giáp chỉ huy. Ngô Quyền lãnh đạo tại Việt Nam."
named = extract_named(txt, fake_ner, fake_sent)
surfaces = {e["surface"] for e in named}
# chức danh "Đại tướng" phải bị loại khỏi tên (nhưng fake_ner không tag nó nên ko sao)
assert "Võ Nguyên Giáp" in surfaces, surfaces
assert "Ngô Quyền" in surfaces, surfaces
# kiểm tra offset đúng
giap = [e for e in named if e["surface"] == "Võ Nguyên Giáp"][0]
assert txt[giap["char_start"]:giap["char_end"]] == "Võ Nguyên Giáp", \
    txt[giap["char_start"]:giap["char_end"]]
print("✓ extract_named + offset:", surfaces)

# ---------- Test bucket ----------
assert year_bucket(1954) == "20C"
assert year_bucket(938) == "10C"
assert year_bucket(-27) == "BC"
print("✓ bucket:", year_bucket(1954), year_bucket(938), year_bucket(-27))

# ---------- Test process_corpus end-to-end ----------
import pandas as pd
df = pd.DataFrame([
    {"context_id": "c1", "is_vietnam": True, "primary_year": 1954,
     "context": "Điện Biên Phủ năm 1954, Võ Nguyên Giáp chỉ huy tại Việt Nam."},
    {"context_id": "c2", "is_vietnam": False, "primary_year": None,
     "context": "Đế quốc La Mã sụp đổ năm 476 sau Công nguyên."},
])
ents, pool = process_corpus(df, fake_ner, fake_sent)
assert len(ents) > 0
# pool có nhóm YEAR
year_buckets = [k for k in pool if k[0] == "YEAR"]
assert year_buckets, list(pool.keys())
print("✓ process_corpus:", len(ents), "thực thể,", len(pool), "nhóm pool")
print("  các nhóm:", sorted(pool.keys()))

print("\n✅ TẤT CẢ TEST PASS — NER/KB/pool/offset/bucket đúng.")
