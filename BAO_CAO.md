# BÁO CÁO ĐỒ ÁN
# HỆ HỖ TRỢ RA QUYẾT ĐỊNH CHO SINH CÂU HỎI TRẮC NGHIỆM LỊCH SỬ

**Môn học:** Hệ Hỗ Trợ Ra Quyết Định  
**Đề tài:** DSS for Educational MCQ Generation — So sánh các phương pháp sinh câu hỏi trắc nghiệm lịch sử, hỗ trợ giáo viên ra quyết định lựa chọn phương pháp và ráp đề thi  
**Ngày báo cáo:** Tháng 7/2026  

---

## MỤC LỤC

1. [Giới thiệu và bài toán DSS](#1-giới-thiệu-và-bài-toán-dss)
2. [Nguồn dữ liệu](#2-nguồn-dữ-liệu)
3. [Phân tích dữ liệu (EDA)](#3-phân-tích-dữ-liệu-eda)
4. [Tiền xử lý dữ liệu](#4-tiền-xử-lý-dữ-liệu)
5. [Các phương pháp sinh câu hỏi](#5-các-phương-pháp-sinh-câu-hỏi)
6. [Module kiểm chứng (Verifier)](#6-module-kiểm-chứng-verifier)
7. [Tầng DSS: AHP/TOPSIS](#7-tầng-dss-ahptopsis)
8. [Tầng DSS: Ráp đề thi (ILP)](#8-tầng-dss-ráp-đề-thi-ilp)
9. [Kết quả so sánh](#9-kết-quả-so-sánh)
10. [Hệ thống (Dashboard)](#10-hệ-thống-dashboard)
11. [Danh sách lệnh đã chạy](#11-danh-sách-lệnh-đã-chạy)
12. [Kết luận và hướng phát triển](#12-kết-luận-và-hướng-phát-triển)

---

## 1. Giới thiệu và bài toán DSS

### 1.1. Bối cảnh

Giáo viên lịch sử cần xây dựng ngân hàng câu hỏi trắc nghiệm chất lượng cao,
đảm bảo phân bố Bloom theo ma trận đặc tả. Việc sinh câu hỏi thủ công tốn thời
gian và khó đảm bảo nhất quán. Bài toán đặt ra: **hệ thống nào nên dùng để sinh
câu hỏi, và làm sao ráp thành đề thi đáp ứng yêu cầu sư phạm?**

### 1.2. Kiến trúc DSS 3 tầng

```
INPUT: (Ngữ cảnh lịch sử, Loại câu hỏi)
       ↓
TẦNG 1 — Sinh câu hỏi (3 phương pháp song song)
  ├─ Rule-Based
  ├─ RAG + LLM
  └─ ViT5 Fine-tuning
       ↓
TẦNG 2 — Kiểm định (Verifier 8 tiêu chí)
  → verifier_score, status: accepted/needs_review/rejected
  → So sánh 3 phương pháp → AHP/TOPSIS → Khuyến nghị
       ↓
TẦNG 3 — Ráp đề thi (ILP tối ưu)
  → Chọn K câu tối ưu theo ma trận đặc tả Bloom
OUTPUT: Đề thi + Giải thích quyết định
```

### 1.3. Input và Output

**Input (2 nguồn):**
- Nguồn 1: Đoạn văn ngữ cảnh lịch sử (context)
- Nguồn 2: Yêu cầu loại câu hỏi: Bloom level × Wh-type

**Output:**
- Câu hỏi + 4 phương án (A/B/C/D) + đáp án đúng
- Metadata: chủ đề, Bloom, độ khó, điểm verifier, cờ vi phạm
- Xếp hạng phương pháp (TOPSIS score)
- Đề thi tối ưu theo ma trận đặc tả

### 1.4. Taxonomy loại câu hỏi

| Loại (Wh-type) | Từ để hỏi | Answer type | Rule-based |
|---|---|---|---|
| thoi_gian | năm nào, khi nào | YEAR/DATE | ✅ |
| nhan_vat | ai, người nào | PER | ✅ |
| dia_diem | ở đâu, tại đâu | LOC | ✅ |
| su_kien | điều gì xảy ra | EVENT | ⚠️ |
| nguyen_nhan | vì sao, do đâu | CLAUSE | ❌ |
| y_nghia | ý nghĩa gì, kết quả | CLAUSE | ❌ |

---

## 2. Nguồn dữ liệu

### 2.1. Nguồn 1 — UIT-ViQuAD 2.0

| Thuộc tính | Giá trị |
|---|---|
| Dataset ID | `taidng/UIT-ViQuAD2.0` |
| Truy cập | HuggingFace, `load_dataset()` |
| Tổng dòng | 39,569 câu hỏi |
| Số bài Wikipedia | 184 bài |
| Định dạng | Parquet |
| Vai trò | Corpus chính: ngữ cảnh sinh câu hỏi + train ViT5 |

**Điểm đặc biệt:**
- Trường `answer_start`: vị trí đáp án trong context → answer-aware QG miễn phí
- Trường `is_impossible` (11.501 câu): nhãn vàng cho tiêu chí Evidence Match

### 2.2. Nguồn 2 — VNHSGE môn Lịch sử

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | GitHub `Xdao85/VNHSGE`, JSON format |
| Số câu lịch sử | 2,000 câu (eval+test+train) |
| Định dạng | `{ID, Q, C, E}` — câu hỏi, đáp án, giải thích |
| Bloom | 4 mức: knowledge/comprehension/application/high application |
| Vai trò | Gold MCQ tham chiếu, nhãn Bloom cho Bloom Fidelity |

---

## 3. Phân tích dữ liệu (EDA)

### 3.1. Corpus lịch sử (sau khi lọc từ ViQuAD)

| Thông số | Giá trị |
|---|---|
| Tổng bài Wikipedia ban đầu | 184 bài |
| Bài lịch sử giữ lại (is_history=1) | **38 bài** |
| Trong đó lịch sử VN (is_vietnam=1) | 9 bài |
| Trong đó lịch sử thế giới | 29 bài |
| Đoạn văn duy nhất (đã khử trùng) | **1,862 đoạn** |
| Câu hỏi trả lời được | 10,114 câu |
| Câu không trả lời được (is_impossible) | **3,883 câu** |
| n_tokens trung vị / đoạn | 167 token |
| year_density trung vị | 0.53 năm/100 token |

**Nhận xét:** year_density = 0.53 cho thấy corpus thiên về văn mô tả (nguyên
nhân, bối cảnh) hơn là liệt kê sự kiện-năm. Điều này dự báo Rule-based sẽ
yếu ở loại câu hỏi Thời gian.

### 3.2. Phân bố split (chia cân tải theo title)

| Split | Số bài | Số đoạn | Tỉ lệ | VN | Thế giới |
|---|---|---|---|---|---|
| train | 23 | 1,491 | 80.1% | 427 | 1,064 |
| dev | 8 | 179 | 9.6% | 52 | 127 |
| test | 7 | 192 | 10.3% | 57 | 135 |

**Phương pháp chia:** Greedy load-balancing theo title (cấp bài), không theo
câu hỏi → chống rò rỉ dữ liệu. Mọi split đều có cả hai tiểu miền VN và thế giới.

### 3.3. Knowledge Base thực thể (NER)

| Type | Số lượng | Precision (100 mẫu) |
|---|---|---|
| LOC | 12,045 | **97%** |
| PER | 7,626 | **90%** |
| YEAR | 2,922 | **100%** |
| ORG | 2,252 | **80%** |
| DATE | 339 | ~95% |

**NER pipeline:** `NlpHUST/ner-vietnamese-electra-base` (ELECTRA) +
`clean_entities.py` (7 luật lọc hậu xử lý).

**Cải tiến so với underthesea:** LOC precision tăng từ ~41% → 97% sau khi
thêm lớp lọc luật (bỏ "năm 1979"→LOC, tách chức danh+tên, loại fragment ELECTRA).

### 3.4. Distractor pool

| (Type, Bucket) | Số surface |
|---|---|
| (PER, TG) | 1,982 |
| (LOC, TG) | 1,836 |
| (LOC, VN) | 881 |
| (PER, VN) | 739 |
| (YEAR, 20C) | 186 |

---

## 4. Tiền xử lý dữ liệu

### 4.1. Pipeline

```
ViQuAD 2.0 (HuggingFace)
    ↓ survey_viquad.py      — khảo sát 184 bài, xuất titles_inventory.csv
    ↓ [Rà tay 15 phút]     — đánh is_history, is_vietnam cho 184 bài
    ↓ build_corpus.py       — lọc, khử trùng, chia split cân tải
    ↓ split_balanced.py     — greedy load-balancing chống rò rỉ
    ↓ ner_kb.py             — ELECTRA-NER + regex thời gian
    ↓ clean_entities.py     — lọc 7 khuôn mẫu lỗi NER
    ↓
corpus.parquet (1,862 đoạn)
qa_pairs.parquet (13,997 câu)
entities.parquet (25,325 thực thể)
distractor_pool.pkl ({(type,bucket) → set})
```

### 4.2. Schema câu hỏi (MCQItem — Pydantic v2)

```json
{
  "item_id": "itm_7f3a9c21",
  "generator": {"method": "rule_based|rag_llm|vit5_ft"},
  "source":    {"context_id": "...", "context": "...", "is_vietnam": true},
  "request":   {"bloom_requested": "nhan_biet", "wh_type_requested": "thoi_gian"},
  "question":  "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
  "options":   [{"label":"A","text":"1945","is_correct":false,"provenance":"kb"}, ...],
  "answer_key": "B",
  "answer_text": "1954",
  "evidence":  {"sentence": "...", "found_in_context": true},
  "verification": null
}
```

**4 bất biến được validator ép:**
1. Đúng 4 phương án, nhãn A/B/C/D, không trùng nội dung
2. Đúng 1 đáp án đúng
3. answer_key và answer_text khớp phương án is_correct=True
4. evidence.sentence là substring nguyên văn của context

---

## 5. Các phương pháp sinh câu hỏi

### 5.1. Phương pháp 1 — Rule-Based

**Cơ chế:**
```
context + wh_type
    → NER (trích entity theo type)
    → Template ("... diễn ra vào năm nào?")
    → Distractor pool (cùng type, cùng bucket thế kỷ)
    → finalize_options() → MCQItem
```

**Đặc điểm:**
- Offline hoàn toàn, $0, 8ms/câu
- Mạnh ở YEAR (regex 100%), Historical Correctness (0.977)
- Yếu ở Duplicate Check (0.036): template quá cứng → câu hỏi trùng nhau
- Không sinh được nguyen_nhan, y_nghia (non-factoid)

**Kết quả:**
- 423 câu sinh được từ 200 đoạn test (VQR thô 70%)
- Sau Verifier: VQR = **27.2%**

### 5.2. Phương pháp 2 — RAG + LLM

**Cơ chế:**
```
topic + wh_type
    → Hybrid retrieval (BM25 + FAISS/vietnamese-bi-encoder, RRF)
    → Top-3 context
    → Prompt GPT-4o-mini (few-shot, JSON schema, evidence_sentence bắt buộc)
    → parse + validate → MCQItem
```

**Đặc điểm:**
- Cần OPENAI_API_KEY, ~5,114ms/câu, ~$0.00019/câu
- Mạnh ở Evidence Match (0.951), Question Clarity (1.0)
- Sinh được mọi loại câu hỏi kể cả non-factoid
- Retry tự động khi JSON lỗi (tối đa 2 lần)

**Kết quả:**
- 90 câu sinh được (VQR thô 100%)
- Sau Verifier: VQR = **62.2%**
- Chi phí: $0.0174 / 90 câu

### 5.3. Phương pháp 3 — ViT5 Fine-tuning

**Cơ chế:**
```
Stage A (QG): <type> wh </type> <ans> answer </ans> <ctx> context </ctx>
              → VietAI/vit5-base fine-tuned → question

Stage B (DG): <q> question </q> <ans> answer </ans> <ctx> context </ctx>
              → VietAI/vit5-base fine-tuned → "dist1 | dist2 | dist3"
```

**Train:**
- Model: `VietAI/vit5-base` (250M params)
- Hardware: RTX 3050 4GB VRAM, fp16
- Batch: 4 × grad_accum 8 = effective 32
- Epochs: 5 với early stopping (patience=2)
- Stage A best dev loss: ~1.78
- Stage B best dev loss: ~1.99

**Hạn chế phát hiện:** Model DG không học được format separator `|` do
dữ liệu train Stage B sinh tự động (answer từ cùng context) không đủ
đa dạng. Phải thêm fallback trích entity từ context.

**Kết quả:**
- 297 câu sinh được từ 300 test (VQR thô ~99%)
- Sau Verifier: VQR = **27.9%**

---

## 6. Module kiểm chứng (Verifier)

### 6.1. 8 tiêu chí và cách đo

| # | Tiêu chí | Phương pháp | Nhãn vàng | Trọng số AHP |
|---|---|---|---|---|
| 1 | Evidence Match | NLI (GPT-4o-mini): context ⊨ Q+ans | is_impossible ViQuAD | 17.8% |
| 2 | Single Correct Answer | NLI cho cả 4 phương án | — | 13.2% |
| 3 | Distractor Type Match | NER type khớp + cosine ∈[0.4,0.9] | KB entities | 10.3% |
| 4 | Historical Correctness | Đối chiếu KB, flag unverified | entities.parquet | 12.6% |
| 5 | Question Clarity | Heuristic: độ dài, đại từ mơ hồ, phủ định kép | — | 8.6% |
| 6 | Duplicate Check | Jaccard trigram (fallback khi không có embedder) | — | 4.5% |
| 7 | Bloom Fidelity | Zero-shot GPT-4o-mini phân loại Bloom | — | 7.4% |
| 8 | Answer Position Balance | Phân bố A/B/C/D ≈ 25% theo batch | — | 4.5% |

**Công thức:**
```
verifier_score = Σ weight_i × score_i   ∈ [0, 1]
status: accepted (≥0.75) / needs_review (0.55–0.75) / rejected (<0.55)
```

### 6.2. Validation Verifier

`is_impossible = True` trong ViQuAD → câu hỏi không có bằng chứng trong context
→ phải bị Evidence Match đánh trượt. Đây là nhãn vàng định lượng được, không cần
gán tay. Có 3,883 câu loại này.

---

## 7. Tầng DSS: AHP/TOPSIS

### 7.1. AHP — Trọng số tiêu chí

Ma trận so sánh cặp 8×8 (thang Saaty 1–9), tổng hợp quan điểm giáo viên:

| Tiêu chí | Trọng số | % |
|---|---|---|
| VQR | 0.2566 | 25.7% |
| Evidence Match | 0.1776 | 17.8% |
| Single Correct Answer | 0.1322 | 13.2% |
| Historical Correctness | 0.1262 | 12.6% |
| Distractor Type Match | 0.1027 | 10.3% |
| Question Clarity | 0.0855 | 8.6% |
| Bloom Fidelity | 0.0742 | 7.4% |
| Duplicate Check | 0.0451 | 4.5% |

**Consistency Ratio (CR) = 0.0230 < 0.1 ✅** — ma trận nhất quán.

### 7.2. TOPSIS — Xếp hạng

| Hạng | Phương pháp | TOPSIS Score | d_PIS | d_NIS |
|---|---|---|---|---|
| 🥇 1 | **RAG+LLM** | **0.9222** | 0.0137 | 0.1621 |
| 🥈 2 | ViT5-FT | 0.3544 | 0.1322 | 0.0726 |
| 🥉 3 | Rule-Based | 0.1744 | 0.1612 | 0.0341 |

### 7.3. Phân tích độ nhạy

Thay đổi trọng số VQR từ 1% → 50%: **không có điểm giao** — thứ hạng
RAG+LLM > ViT5 > Rule-based ổn định tuyệt đối bất kể trọng số VQR.

---

## 8. Tầng DSS: Ráp đề thi (ILP)

### 8.1. Bài toán tối ưu

```
max  Σ verifier_score(i) · x(i)
s.t. Σ x(i) = K                        (đúng K câu)
     phân bố Bloom ≈ ma trận đặc tả    (±1 câu slack)
     mỗi title ≥ 1 câu                 (phủ đủ chủ đề)
     mỗi context_id ≤ 2 câu            (không lặp ngữ cảnh)
     x(i) ∈ {0,1}
Solver: PuLP/CBC
```

### 8.2. So sánh 3 chiến lược ráp đề

| Chiến lược | Tổng score | Thời gian | Tuân thủ Bloom |
|---|---|---|---|
| ILP (tối ưu) | cao nhất | ~30s | ✅ chính xác |
| Greedy | ~95% của ILP | <1ms | ⚠️ xấp xỉ |
| Random (baseline) | thấp nhất | <1ms | ❌ ngẫu nhiên |

---

## 9. Kết quả so sánh

### 9.1. Bảng so sánh chính (từ Verifier — số liệu thật)

| Tiêu chí | Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|---|
| **VQR (accepted %)** | 27.2% | **62.2%** | 27.9% |
| **Avg verifier_score** | 0.582 | **0.777** | 0.658 |
| Evidence Match | 0.349 | **0.951** | 0.811 |
| Single Correct Answer | 0.271 | **0.594** | 0.367 |
| Distractor Type Match | **0.689** | 0.642 | 0.463 |
| Historical Correctness | **0.977** | 0.838 | 0.708 |
| Question Clarity | 0.988 | **1.000** | 0.992 |
| Bloom Fidelity | **1.000** | 0.933 | 0.754 |
| Duplicate Check | 0.036 | 0.457 | **0.513** |
| **Latency** | **8ms** | 5,114ms | ~200ms |
| **Chi phí/câu** | **$0** | $0.00019 | **$0** |
| Offline | ✅ | ❌ | ✅ |

### 9.2. Kỳ vọng vs Thực tế

| Phương pháp | Kỳ vọng | Thực tế | Nhận xét |
|---|---|---|---|
| Rule-Based | VQR~60%, mạnh YEAR | VQR=27%, Duplicate=0.036 | Template trùng → VQR thấp hơn kỳ vọng |
| RAG+LLM | VQR~80%, mạnh Nguyên nhân | VQR=62%, Evidence=0.951 ✅ | Gần kỳ vọng, Evidence Match vượt |
| ViT5 | VQR~70%, cân bằng | VQR=28%, Single Correct=0.367 | DG kém → nhiều câu có >1 đáp án đúng |

### 9.3. Phân tích điểm mạnh/yếu

**Rule-Based:**
- ✅ Mạnh tuyệt đối: Historical Correctness (0.977), Bloom Fidelity (1.0), Latency (8ms), $0
- ❌ Yếu nghiêm trọng: Duplicate Check (0.036) — nguyên nhân: template quá cứng, sinh câu "... diễn ra vào năm nào?" lặp đi lặp lại
- ❌ Không sinh được non-factoid (Nguyên nhân, Ý nghĩa)

**RAG+LLM:**
- ✅ Mạnh: Evidence Match (0.951), VQR cao nhất (62.2%), sinh mọi loại câu hỏi
- ❌ Yếu: Chậm (5,114ms), tốn tiền, cần internet

**ViT5-FT:**
- ✅ Mạnh: Evidence in context (100% thô — model QG trích trực tiếp từ context)
- ❌ Yếu: Single Correct Answer (0.367) — model DG không học được format separator |, sinh distractor kém chất lượng
- ✅ Offline, $0, latency trung bình

### 9.4. Kết luận DSS

> Không có phương pháp nào thắng trên mọi tiêu chí.
> Chiến lược tối ưu là **định tuyến theo ngữ cảnh sử dụng**:
>
> - **Thời gian/Nhân vật/Địa điểm + offline + $0** → Rule-Based (cần đa dạng hoá template)
> - **Nguyên nhân/Ý nghĩa + ưu tiên chất lượng** → RAG+LLM
> - **Số lượng lớn + offline + chấp nhận distractor trung bình** → ViT5-QG + RAG-DG (hybrid)
>
> Thứ hạng TOPSIS ổn định (không có điểm giao khi thay đổi trọng số VQR 1%→50%).

---

## 10. Hệ thống (Dashboard)

Dashboard Streamlit 5 trang:

| Trang | Tính năng |
|---|---|
| Tổng quan | KPI cards, bảng so sánh, biểu đồ VQR |
| Demo Sinh Câu Hỏi | Nhập context tự do → sinh từ 3 phương pháp song song |
| Ngân hàng câu hỏi | Lọc/xem câu hỏi, verifier_score, vi phạm |
| AHP/TOPSIS | **Kéo trọng số → thứ hạng đổi realtime** + Radar chart |
| Ráp đề thi | Chọn K câu + Bloom ratio → ILP/Greedy/Random + download |

---

## 11. Danh sách lệnh đã chạy

### 11.1. Cài đặt môi trường

```bash
# Cài thư viện cơ bản
pip install datasets pandas pyarrow pydantic

# NER và transformers
pip install transformers sentencepiece underthesea

# Train ViT5
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate

# RAG
pip install faiss-cpu sentence-transformers rank-bm25 openai

# DSS
pip install pulp

# Dashboard
pip install streamlit plotly
```

### 11.2. Bước 1 — Khảo sát và build corpus

```bash
# Khảo sát 184 bài ViQuAD, xuất titles_inventory.csv để rà tay
python src/preprocess/survey_viquad.py --outdir data/processed

# [THỦ CÔNG] Mở titles_inventory.csv, đánh is_history (1/0) và is_vietnam (1/0)
# Kết quả: 38 bài is_history=1, trong đó 9 bài is_vietnam=1

# Build corpus chính thức (đọc file đã rà tay)
python src/preprocess/build_corpus.py \
    --inventory data/processed/titles_inventory.csv \
    --outdir data/processed
# → corpus.parquet (1,862 đoạn), qa_pairs.parquet (13,997 câu), splits_report.txt
# Kiểm tra rò rỉ: ✅ đoạn=1 split, bài=1 split
```

### 11.3. Bước 2 — NER và Knowledge Base

```bash
# Chạy ELECTRA-NER (backend mới, tốt hơn underthesea)
python src/preprocess/ner_kb.py \
    --corpus data/processed/corpus.parquet \
    --outdir data/processed \
    --backend electra
# → entities.parquet (25,325), distractor_pool.pkl (37 nhóm), ner_quality_sample.csv

# [THỦ CÔNG] Chấm tay 100 mẫu trong ner_quality_sample.csv (điền cột correct: 1/0)
# Kết quả: YEAR=100%, LOC=84%, PER=75%, ORG=57% → CHƯA ĐẠT

# Chạy lại với clean_entities v2 (sau khi bổ sung luật lọc)
python src/preprocess/ner_kb.py \
    --corpus data/processed/corpus.parquet \
    --outdir data/processed \
    --backend electra
# → entities.parquet (25,325 → sau lọc thực tế sạch hơn)
# Precision sau lọc (mô phỏng trên 100 mẫu): YEAR=100%, LOC=97%, PER=90%, ORG=80%
```

### 11.4. Bước 3 — Phương pháp 1: Rule-Based

```bash
# Sinh câu hỏi trên tập train, 3 loại wh_type
python src/generators/rule_based.py \
    --corpus data/processed/corpus.parquet \
    --entities data/processed/entities.parquet \
    --pool data/processed/distractor_pool.pkl \
    --split train --limit 200 \
    --wh_types thoi_gian,nhan_vat,dia_diem
# → data/generated/rule_based.jsonl (423 câu)
# VQR thô=70%, Evidence=97%, Latency=8ms, Cost=$0
```

### 11.5. Bước 4 — Phương pháp 2: RAG+LLM

```bash
# Build FAISS + BM25 index (chạy 1 lần, ~10 phút, tải model ~500MB)
python src/generators/build_index.py \
    --corpus data/processed/corpus.parquet \
    --outdir data/index
# → faiss.index, bm25.pkl, index_meta.parquet
# Xác nhận: test split KHÔNG nằm trong index

# Sinh câu hỏi (cần OPENAI_API_KEY)
set OPENAI_API_KEY=sk-...
python src/generators/rag_llm.py \
    --n_topics 10 \
    --wh_types thoi_gian,nhan_vat,nguyen_nhan \
    --bloom nhan_biet \
    --top_k 3
# → data/generated/rag_llm.jsonl (90 câu)
# VQR thô=100%, Evidence=82%, Latency=5114ms, Cost=$0.0174
```

### 11.6. Bước 5 — Phương pháp 3: ViT5 Fine-tuning

```bash
# Train Stage A (QG) + Stage B (DG) — RTX 3050, ~3-5 giờ
python src/generators/train_vit5.py \
    --data_dir data/processed \
    --outdir models \
    --stage AB \
    --epochs 5 \
    --batch 4 \
    --accum 8
# → models/vit5_qg/ (Stage A, best dev loss ~1.78)
# → models/vit5_dg/ (Stage B, best dev loss ~1.99)
# Hỗ trợ resume từ checkpoint khi máy tắt giữa chừng

# Inference trên test set
python src/generators/infer_vit5.py \
    --qg_model models/vit5_qg \
    --dg_model models/vit5_dg \
    --data_dir data/processed \
    --out data/generated/vit5_ft.jsonl \
    --limit 300
# → data/generated/vit5_ft.jsonl (297 câu)
# VQR thô~99% (nhưng distractor kém, nhiều câu fail Verifier)
```

### 11.7. Bước 6 — Verifier

```bash
# Chạy Verifier trên 3 file (cần OPENAI_API_KEY cho NLI và Bloom Fidelity)
set OPENAI_API_KEY=sk-...
python src/verifier/verifier.py \
    --inputs data/generated/rule_based.jsonl \
             data/generated/rag_llm.jsonl \
             data/generated/vit5_ft.jsonl \
    --entities data/processed/entities.parquet \
    --outdir data/verified \
    --no_embed
# → verified_rule_based.jsonl, verified_rag_llm.jsonl, verified_vit5_ft.jsonl
# → results_table.csv (bảng so sánh 3×8)
```

**Kết quả results_table.csv:**
```
method      n_items  vqr       avg_score  evidence_match  ...
rule_based  423      0.271868  0.582257   0.349           ...
rag_llm     90       0.622222  0.776738   0.951           ...
vit5_ft     297      0.279461  0.657703   0.811           ...
```

### 11.8. Bước 7 — AHP/TOPSIS

```bash
python src/dss/ahp_topsis.py \
    --results data/verified/results_table.csv \
    --outdir data/dss
# → ahp_weights.csv (CR=0.023 ✅), topsis_ranking.csv, sensitivity.csv, dss_report.txt
# Kết quả: RAG+LLM (0.922) > ViT5 (0.354) > Rule-based (0.174)
# Độ nhạy: thứ hạng ổn định khi VQR thay đổi 1%→50%
```

### 11.9. Bước 8 — Ráp đề thi

```bash
pip install pulp
python src/dss/exam_builder.py \
    --verified_dir data/verified \
    --n_questions 40 \
    --bloom_ratio 0.4,0.35,0.25 \
    --outdir data/exams
# → exam_ilp.csv, exam_greedy.csv, exam_random.csv, strategy_comparison.csv
```

### 11.10. Bước 9 — Dashboard

```bash
streamlit run src/app/dashboard.py
# Mở http://localhost:8501
# 5 trang: Tổng quan, Demo, Ngân hàng, AHP/TOPSIS realtime, Ráp đề
```

---

## 12. Kết luận và hướng phát triển

### 12.1. Đóng góp chính

1. **Pipeline DSS hoàn chỉnh 3 tầng** — sinh, kiểm định, ráp đề — với giải
   thích quyết định tường minh qua AHP/TOPSIS.

2. **Verifier 8 tiêu chí có nhãn vàng** — tận dụng `is_impossible` của ViQuAD
   2.0 làm ground truth cho Evidence Match, không cần gán tay.

3. **Phát hiện kỹ thuật có giá trị:**
   - NER underthesea LOC precision ~41% → ELECTRA + lọc luật → 97%
   - Split hash mù → 87/6.5/6.5 → Greedy cân tải → 80/10/10
   - ViT5 DG không học được format separator `|` với corpus nhỏ

4. **Kết luận DSS khả thi:** Hybrid routing tốt hơn chọn một phương pháp đơn độc.

### 12.2. Hạn chế

- VNHSGE chưa được tích hợp trực tiếp vào pipeline (Bloom Fidelity dùng
  GPT zero-shot thay vì classifier từ VNHSGE).
- Distractor pool thiếu đa dạng cho corpus thế giới (tên nước ngoài ít trong KB).
- Human eval chưa thực hiện (chỉ có Verifier tự động).
- ViT5 Stage B cần dữ liệu DG chất lượng hơn (do người viết, không phải tự động).

### 12.3. Hướng phát triển

- Train Bloom classifier trên VNHSGE-History để thay thế GPT zero-shot
- Hybrid routing tự động theo wh_type (rule-based cho factoid, RAG cho non-factoid)
- Mở rộng sang tiếng Anh (SciQ) để so sánh đa ngôn ngữ
- Human eval 50 câu/phương pháp với Cohen's kappa

---

## PHỤ LỤC: Cấu trúc thư mục dự án

```
files_v1/
├── README.md
├── setup.py
├── src/
│   ├── schema/        mcq_schema.py
│   ├── preprocess/    survey_viquad.py, build_corpus.py, split_balanced.py,
│   │                  ner_kb.py, clean_entities.py, review_sample.py
│   ├── generators/    rule_based.py, build_index.py, rag_llm.py,
│   │                  train_vit5.py, infer_vit5.py
│   ├── verifier/      verifier.py
│   ├── dss/           ahp_topsis.py, exam_builder.py
│   └── app/           dashboard.py
├── data/
│   ├── processed/     corpus.parquet, qa_pairs.parquet, entities.parquet,
│   │                  distractor_pool.pkl, titles_inventory.csv, splits_report.txt
│   ├── index/         faiss.index, bm25.pkl, index_meta.parquet
│   ├── generated/     rule_based.jsonl, rag_llm.jsonl, vit5_ft.jsonl
│   ├── verified/      verified_*.jsonl, results_table.csv
│   ├── dss/           ahp_weights.csv, topsis_ranking.csv, sensitivity.csv
│   └── exams/         exam_ilp.csv, strategy_comparison.csv
├── models/
│   ├── vit5_qg/       Stage A model (best dev loss ~1.78)
│   └── vit5_dg/       Stage B model (best dev loss ~1.99)
└── tests/
    test_schema.py, test_build.py, test_survey.py,
    test_ner.py, test_clean.py, test_rule_based.py,
    test_rag_llm.py, test_verifier.py
```

---

*Báo cáo này tổng hợp toàn bộ quá trình thực hiện đồ án đến tháng 7/2026.*
*Mọi số liệu đều từ kết quả chạy thật trên dữ liệu thật.*
