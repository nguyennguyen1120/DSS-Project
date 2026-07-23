"""Test offline logic build_corpus (không cần internet)."""
import sys
sys.path.insert(0, "src/preprocess")
from build_corpus import build, first_answer, make_context_id

CTX_DBP = ("Chiến thắng Điện Biên Phủ diễn ra năm 1954, kết thúc chín năm kháng "
           "chiến chống thực dân Pháp. Chiến dịch do Đại tướng Võ Nguyên Giáp chỉ "
           "huy tại lòng chảo Điện Biên Phủ, tỉnh Lai Châu, từ 13/3 đến 7/5/1954.")
CTX_ROME = ("Đế quốc La Mã là giai đoạn của nền văn minh La Mã cổ đại, đặc trưng "
            "bởi chính thể quân chủ. Thành Rome được cai trị bởi các hoàng đế sau "
            "khi nền cộng hòa sụp đổ vào năm 27 trước Công nguyên dưới thời Augustus.")
CTX_ENG = ("Tiếng Anh có bảy lớp từ chính gồm động từ, danh từ, tính từ, trạng từ, "
           "hạn định từ, giới từ và liên từ, duy trì hệ thống cách ở đại từ nhân xưng "
           "nhiều hơn các lớp từ khác trong ngữ pháp hiện đại ngày nay hiện tại.")

# answer_start của "1954" trong CTX_DBP:
start_1954 = CTX_DBP.index("1954")

mock_rows = [
    # Bài LỊCH SỬ VN — 2 câu answerable + 1 unanswerable, cùng context
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Điện Biên Phủ diễn ra năm nào?",
     "answers": {"text": ["1954"], "answer_start": [start_1954]},
     "is_impossible": False, "plausible": ""},
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Ai chỉ huy?",
     "answers": {"text": ["Võ Nguyên Giáp"],
                 "answer_start": [CTX_DBP.index("Võ Nguyên Giáp")]},
     "is_impossible": False, "plausible": ""},
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Có bao nhiêu lính Mỹ tham chiến?",
     "answers": {"text": [], "answer_start": []},
     "is_impossible": True, "plausible": "5000"},
    # Bài LỊCH SỬ thế giới
    {"title": "Đế quốc La Mã", "context": CTX_ROME,
     "question": "Nền cộng hòa La Mã sụp đổ năm nào?",
     "answers": {"text": ["27 trước Công nguyên"],
                 "answer_start": [CTX_ROME.index("27 trước")]},
     "is_impossible": False, "plausible": ""},
    # Bài KHÔNG lịch sử -> phải bị loại
    {"title": "Ngữ pháp tiếng Anh", "context": CTX_ENG,
     "question": "Mấy lớp từ?",
     "answers": {"text": ["bảy"], "answer_start": [CTX_ENG.index("bảy")]},
     "is_impossible": False, "plausible": ""},
]

# Giả lập inventory đã rà tay
keep = {"Chiến dịch Điện Biên Phủ", "Đế quốc La Mã"}   # is_history=1
vn = {"Chiến dịch Điện Biên Phủ"}                       # is_vietnam=1

corpus, qa, stats, title_sizes, title_is_vn, title2split = build(mock_rows, keep, vn)

print("=== STATS ===")
for k, v in stats.items():
    print(f"  {k:20s}: {v}")

assert stats["rows_seen"] == 5
assert stats["skipped_not_history"] == 1        # bài tiếng Anh bị loại
assert stats["rows_kept"] == 4
assert stats["passages"] == 2                    # DBP khử trùng còn 1 + La Mã 1
assert stats["answerable"] == 3
assert stats["unanswerable"] == 1

# --- answer span validate ---
qa_by_q = {q["question"]: q for q in qa}
assert qa_by_q["Điện Biên Phủ diễn ra năm nào?"]["answer_span_ok"] is True
assert qa_by_q["Điện Biên Phủ diễn ra năm nào?"]["answer_text"] == "1954"
# unanswerable: giữ plausible, không có span
unans = qa_by_q["Có bao nhiêu lính Mỹ tham chiến?"]
assert unans["is_impossible"] is True
assert unans["plausible_answer"] == "5000"
assert unans["answer_span_ok"] is False

# --- tiểu miền VN ---
cby = {c["title"]: c for c in corpus}
assert cby["Chiến dịch Điện Biên Phủ"]["is_vietnam"] is True
assert cby["Đế quốc La Mã"]["is_vietnam"] is False

# --- CHỐNG RÒ RỈ: cùng title -> cùng split (cân tải) ---
sp_dbp = {q["split"] for q in qa if q["title"] == "Chiến dịch Điện Biên Phủ"}
assert len(sp_dbp) == 1, f"RÒ RỈ: DBP ở nhiều split {sp_dbp}"
# mỗi đoạn/câu đều được gán split (không còn None)
assert all(c["split"] in {"train", "dev", "test"} for c in corpus)
assert all(q["split"] in {"train", "dev", "test"} for q in qa)
# deterministic: chạy lại build cho kết quả split giống hệt
_c2, _q2, *_ = build(mock_rows, keep, vn)
assert {c["context_id"]: c["split"] for c in corpus} == \
       {c["context_id"]: c["split"] for c in _c2}, "Chia split không deterministic"

print("\n=== CORPUS ===")
for c in corpus:
    vn_flag = "VN" if c["is_vietnam"] else "TG"
    print(f"  [{vn_flag}] {c['title']:26s} | split={c['split']:5s} | "
          f"năm {c['primary_year']} | {c['n_tokens']} token")

print("\n=== QA PAIRS ===")
for q in qa:
    kind = "IMPOSSIBLE" if q["is_impossible"] else f"ans='{q['answer_text']}'"
    print(f"  [{q['split']:5s}] {kind:24s} span_ok={q['answer_span_ok']} | {q['question']}")

print("\n✅ TẤT CẢ TEST PASS — build/khử trùng/QA/span/chống rò rỉ đúng.")
