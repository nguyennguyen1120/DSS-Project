"""Test offline logic khảo sát ViQuAD (không cần internet)."""
import sys
sys.path.insert(0, "src/preprocess")
from survey_viquad import build, guess_history, make_context_id

CTX_DBP = ("Chiến thắng Điện Biên Phủ diễn ra năm 1954, kết thúc chín năm kháng "
           "chiến chống thực dân Pháp. Chiến dịch do Đại tướng Võ Nguyên Giáp chỉ "
           "huy tại lòng chảo Điện Biên Phủ, tỉnh Lai Châu, từ 13/3 đến 7/5/1954.")
CTX_LY = ("Năm 1010, vua Lý Thái Tổ ban Chiếu dời đô, dời kinh đô từ Hoa Lư về "
          "thành Đại La và đổi tên thành Thăng Long, mở đầu thời kỳ nhà Lý phát "
          "triển rực rỡ trong lịch sử phong kiến Việt Nam.")
CTX_ENG = ("Tiếng Anh có bảy lớp từ chính: động từ, danh từ, tính từ, trạng từ, "
           "hạn định từ, giới từ và liên từ. Đại từ nhân xưng duy trì hệ thống cách "
           "hoàn chỉnh hơn các lớp từ khác trong ngữ pháp tiếng Anh hiện đại.")

# Mock: mô phỏng nhiều câu hỏi chung 1 context, có answerable + unanswerable
mock_rows = [
    # Bài lịch sử "Chiến dịch Điện Biên Phủ": 3 câu hỏi, 2 answerable + 1 unanswerable
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Điện Biên Phủ diễn ra năm nào?",
     "answers": {"text": ["1954"], "answer_start": [30]}, "is_impossible": False},
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Ai chỉ huy chiến dịch?",
     "answers": {"text": ["Võ Nguyên Giáp"], "answer_start": [70]}, "is_impossible": False},
    {"title": "Chiến dịch Điện Biên Phủ", "context": CTX_DBP,
     "question": "Chiến dịch có bao nhiêu binh sĩ Mỹ?",
     "answers": {"text": [], "answer_start": []}, "is_impossible": True},

    # Bài lịch sử "Nhà Lý": 2 câu hỏi answerable, cùng context
    {"title": "Nhà Lý", "context": CTX_LY,
     "question": "Ai ban Chiếu dời đô?",
     "answers": {"text": ["Lý Thái Tổ"], "answer_start": [13]}, "is_impossible": False},
    {"title": "Nhà Lý", "context": CTX_LY,
     "question": "Kinh đô dời về đâu?",
     "answers": {"text": ["Thăng Long"], "answer_start": [60]}, "is_impossible": False},

    # Bài KHÔNG lịch sử "Ngữ pháp tiếng Anh": 1 câu answerable
    {"title": "Ngữ pháp tiếng Anh", "context": CTX_ENG,
     "question": "Tiếng Anh có mấy lớp từ chính?",
     "answers": {"text": ["bảy"], "answer_start": [13]}, "is_impossible": False},
]

passages, title_stats, stats = build(mock_rows)

print("=== STATS ===")
for k, v in stats.items():
    print(f"  {k:20s}: {v}")

# 6 dòng, 3 context duy nhất, 4 answerable + 1 unanswerable... đợi: DBP có 2 ans +1 unans, LY 2 ans, ENG 1 ans
assert stats["total_rows"] == 6
assert stats["unique_passages"] == 3, stats["unique_passages"]
assert stats["answerable"] == 5
assert stats["unanswerable"] == 1

# khử trùng: DBP xuất hiện 3 lần nhưng chỉ 1 passage
assert len(passages) == 3

# đoán lịch sử: DBP và LY là lịch sử, ENG thì không
by_title = {p["title"]: p for p in passages.values()}
assert by_title["Chiến dịch Điện Biên Phủ"]["is_history_guess"] is True
assert by_title["Nhà Lý"]["is_history_guess"] is True
assert by_title["Ngữ pháp tiếng Anh"]["is_history_guess"] is False

# thống kê theo bài
assert title_stats["Chiến dịch Điện Biên Phủ"]["n_answerable"] == 2
assert title_stats["Chiến dịch Điện Biên Phủ"]["n_unanswerable"] == 1
assert len(title_stats["Chiến dịch Điện Biên Phủ"]["contexts"]) == 1  # cùng 1 context

# năm trích đúng
assert by_title["Nhà Lý"]["primary_year"] == 1010
assert by_title["Chiến dịch Điện Biên Phủ"]["primary_year"] == 1954

print("\n=== PASSAGES ===")
for p in passages.values():
    flag = "LỊCH SỬ" if p["is_history_guess"] else "khác"
    print(f"  [{flag:7s}] {p['title']:28s} | năm {p['primary_year']} | "
          f"{p['n_tokens']} token | density {p['year_density']}")

print("\n✅ TẤT CẢ TEST PASS — logic khảo sát/khử trùng/lọc lịch sử đúng.")
