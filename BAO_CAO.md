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
10. [Hệ thống Dashboard](#10-hệ-thống-dashboard)
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
INPUT: (Ngữ cảnh lịch sử, Loại câu hỏi Wh-type, Bloom level)
       ↓
TẦNG 1 — Sinh câu hỏi (3 phương pháp song song)
  ├─ Rule-Based    : Template + KB thực thể
  ├─ RAG + LLM     : Hybrid retrieval + GPT-4o-mini
  └─ ViT5 Fine-tune: Seq2Seq answer-aware QG
       ↓
TẦNG 2 — Kiểm định (Verifier 8 tiêu chí)
  → verifier_score ∈ [0,1]
  → status: accepted / needs_review / rejected
  → AHP/TOPSIS → Khuyến nghị phương pháp
       ↓
TẦNG 3 — Ráp đề thi (ILP tối ưu)
  → Chọn K câu tối ưu theo ma trận đặc tả Bloom
OUTPUT: Đề thi + Giải thích quyết định
```

### 1.3. Input và Output

**Input:**
- Nguồn 1: Đoạn văn ngữ cảnh lịch sử (context)
- Nguồn 2: Yêu cầu: Bloom level × Wh-type

**Output:**
- Câu hỏi + 4 phương án A/B/C/D + đáp án đúng
- verifier_score, cờ vi phạm từng tiêu chí
- Xếp hạng phương pháp (TOPSIS score)
- Đề thi tối ưu theo ma trận đặc tả

### 1.4. Taxonomy Wh-type

| Wh-type | Từ để hỏi | Rule-based |
|---|---|---|
| thoi_gian | năm nào, khi nào | ✅ |
| nhan_vat | ai, người nào | ✅ |
| dia_diem | ở đâu, tại đâu | ✅ |
| su_kien | điều gì xảy ra | ⚠️ |
| nguyen_nhan | vì sao, do đâu | ❌ non-factoid |
| y_nghia | ý nghĩa gì, kết quả | ❌ non-factoid |

---

## 2. Nguồn dữ liệu

### 2.1. Nguồn 1 — UIT-ViQuAD 2.0

| Thuộc tính | Giá trị |
|---|---|
| Dataset ID | `taidng/UIT-ViQuAD2.0` |
| Truy cập | HuggingFace `load_dataset()` |
| Tổng câu hỏi | 39,569 |
| Bài Wikipedia | 184 bài |
| Vai trò | Corpus sinh câu hỏi + train ViT5 |

**Điểm đặc biệt:**
- `answer_start`: vị trí đáp án → answer-aware QG
- `is_impossible` (11,501 câu): nhãn vàng cho Evidence Match

### 2.2. Nguồn 2 — VNHSGE môn Lịch sử

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | GitHub `Xdao85/VNHSGE`, JSON |
| Số câu lịch sử | 2,000 câu (5 năm 2019–2023) |
| Schema | `{ID, Question, Choice, Explanation}` |
| Vai trò | Train Bloom classifier (tích hợp vào Verifier) |

**Cách dùng thực tế:**
- Tải 50 file JSON lịch sử từ GitHub
- Gán nhãn Bloom bằng heuristic từ khoá (4 mức)
- Train TF-IDF + Logistic Regression classifier
- Tích hợp vào Verifier tiêu chí Bloom Fidelity (offline, $0)
- Eval accuracy = **90%** trên 200 mẫu eval

---

## 3. Phân tích dữ liệu (EDA)

### 3.1. Corpus lịch sử (sau lọc từ ViQuAD)

| Thông số | Giá trị |
|---|---|
| Bài Wikipedia giữ lại | 38 bài (is_history=1) |
| Lịch sử VN | 9 bài (536 đoạn, 28.8%) |
| Lịch sử TG | 29 bài (1,326 đoạn, 71.2%) |
| Tổng đoạn văn | **1,862 đoạn** |
| Câu answerable | 10,114 câu |
| Câu unanswerable | **3,883 câu** (nhãn vàng) |
| n_tokens median | 171 token/đoạn |
| year_density median | 0.65 năm/100 token |

**Nhận xét:** year_density thấp → corpus thiên về mô tả hơn liệt kê sự kiện.
Điều này làm Rule-based sinh ít câu Thời gian hơn kỳ vọng.

### 3.2. Phân bố split

| Split | Đoạn | % | Câu hỏi |
|---|---|---|---|
| train | 1,491 | 80.1% | 11,103 |
| dev | 179 | 9.6% | 1,536 |
| test | 192 | 10.3% | 1,358 |

Chia theo title (cấp bài) bằng greedy load-balancing → chống rò rỉ.

### 3.3. Knowledge Base thực thể (NER)

| Type | Số lượng | Precision |
|---|---|---|
| LOC | 12,045 | **97%** |
| PER | 7,626 | **90%** |
| YEAR | 2,922 | **100%** |
| ORG | 2,252 | **80%** |

NER backend: ELECTRA (`NlpHUST/ner-vietnamese-electra-base`) + `clean_entities.py`
(7 luật lọc). Cải thiện LOC: underthesea ~41% → ELECTRA+lọc **97%**.

### 3.4. VNHSGE — Phân bố Bloom

| Mức Bloom | Số câu | % |
|---|---|---|
| nhan_biet | 1,351 | 67.5% |
| thong_hieu | 523 | 26.2% |
| van_dung | 115 | 5.8% |
| van_dung_cao | 11 | 0.5% |

**Hạn chế:** Mất cân bằng nghiêm trọng → van_dung_cao F1=0%.
Ghi nhận trong báo cáo và dùng `class_weight='balanced'` khi train.

---

## 4. Tiền xử lý dữ liệu

### 4.1. Schema MCQItem (Pydantic v2)

```json
{
  "item_id": "itm_7f3a9c21",
  "generator": {"method": "rule_based|rag_llm|vit5_ft"},
  "source": {"context_id": "...", "context": "...", "is_vietnam": true},
  "request": {"bloom_requested": "nhan_biet", "wh_type_requested": "thoi_gian"},
  "question": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
  "options": [
    {"label":"A","text":"1945","is_correct":false,"provenance":"kb"},
    {"label":"B","text":"1954","is_correct":true,"provenance":"context_span"},
    {"label":"C","text":"1975","is_correct":false,"provenance":"kb"},
    {"label":"D","text":"1986","is_correct":false,"provenance":"kb"}
  ],
  "answer_key": "B",
  "answer_text": "1954",
  "evidence": {"sentence": "...", "found_in_context": true},
  "verification": null
}
```

**4 bất biến được Pydantic validator ép:**
1. Đúng 4 phương án, nhãn A/B/C/D, không trùng nội dung
2. Đúng 1 đáp án đúng
3. answer_key khớp phương án is_correct=True
4. evidence.sentence là substring nguyên văn của context

---

## 5. Các phương pháp sinh câu hỏi

### 5.1. Phương pháp 1 — Rule-Based

**Pipeline:**
```
context + wh_type → NER entities → Template → Distractor pool → MCQItem
```
- Offline, $0, 8ms/câu
- Mạnh: Historical Correctness, Bloom Fidelity, tốc độ
- Yếu: Duplicate Check (template lặp), không sinh non-factoid

**Kết quả:** 423 câu (200 đoạn × 3 wh_type), VQR thô 70%

### 5.2. Phương pháp 2 — RAG + LLM

**Pipeline:**
```
topic → BM25 + FAISS (RRF) → Top-3 context → GPT-4o-mini few-shot → MCQItem
```
- Cần API, ~5,114ms/câu, $0.00019/câu
- Mạnh: Evidence Match, VQR cao nhất, sinh mọi wh_type
- Yếu: Chậm, tốn tiền, cần internet

**Kết quả:** 90 câu, VQR thô 100%, chi phí $0.0174

### 5.3. Phương pháp 3 — ViT5 Fine-tuning

**Pipeline:**
```
Stage A (QG): <type> wh </type> <ans> answer </ans> <ctx> context </ctx> → question
Stage B (DG): <q> question </q> <ans> answer </ans> <ctx> context </ctx> → distractors
```
- RTX 3050 4GB, fp16, batch=4, grad_accum=8
- Stage A best dev loss: ~1.78 | Stage B: ~1.99
- Hạn chế: DG không học được format `|` → fallback entity từ context

**Kết quả:** 297 câu (300 test), VQR thô ~99%

---

## 6. Module kiểm chứng (Verifier)

### 6.1. 8 tiêu chí

| # | Tiêu chí | Phương pháp | Trọng số AHP |
|---|---|---|---|
| 1 | Evidence Match | NLI GPT-4o-mini | 17.8% |
| 2 | Single Correct | NLI 4 phương án | 13.2% |
| 3 | Distractor Type | NER type + cosine | 10.3% |
| 4 | Historical Correctness | KB lookup | 12.6% |
| 5 | Question Clarity | Heuristic (độ dài, đại từ mơ hồ) | 8.6% |
| 6 | Duplicate Check | Jaccard trigram | 4.5% |
| 7 | **Bloom Fidelity** | **VNHSGE classifier** (offline) | 7.4% |
| 8 | Answer Position | Phân bố A/B/C/D | 4.5% |

**Ngưỡng:** accepted ≥ 0.75, needs_review ∈ [0.55, 0.75), rejected < 0.55

### 6.2. So sánh Bloom backend

| Backend | Accuracy | Chi phí | Online? |
|---|---|---|---|
| VNHSGE Classifier | 64% vs requested | **$0** | Không cần |
| GPT zero-shot | **76% vs requested** | ~$0.01/1k câu | Cần API |
| Agreement giữa 2 | 76% (68/90 câu) | — | — |

**Nhận xét:** GPT chính xác hơn 12% nhưng chi phí thấp ($0.01/1k câu).
Sự chênh lệch chủ yếu ở câu "Nguyên nhân" — Classifier đánh giá
`thong_hieu` (đúng về sư phạm) trong khi GPT trả về `nhan_biet`
(khớp với bloom_requested). Hệ thống dùng GPT làm default, Classifier
làm fallback offline.

---

## 7. Tầng DSS: AHP/TOPSIS

### 7.1. AHP — Trọng số (CR = 0.0230 ✅)

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

### 7.2. TOPSIS — Xếp hạng (số liệu chính thức)

| Hạng | Phương pháp | Score | d_PIS | d_NIS |
|---|---|---|---|---|
| 🥇 1 | **RAG+LLM** | **0.8894** | 0.0185 | 0.1490 |
| 🥈 2 | ViT5-FT | 0.5417 | 0.0748 | 0.0884 |
| 🥉 3 | Rule-Based | 0.1584 | 0.1492 | 0.0281 |

### 7.3. Điểm mạnh tương đối

| Phương pháp | Thắng tuyệt đối ở tiêu chí |
|---|---|
| Rule-Based | Distractor Type (0.920), Historical Correctness (0.977), Bloom Fidelity (0.926) |
| RAG+LLM | VQR (62.2%), Evidence Match (0.951), Single Correct (0.594), Question Clarity (1.0) |
| ViT5-FT | Duplicate Check (0.733) |

### 7.4. Phân tích độ nhạy

Thay đổi trọng số VQR từ 1% → 50%: **không có điểm giao**.
Thứ hạng RAG+LLM > ViT5 > Rule-based ổn định tuyệt đối.

---

## 8. Tầng DSS: Ráp đề thi (ILP)

### 8.1. Bài toán tối ưu

```
max  Σ verifier_score(i) · x(i)
s.t. Σ x(i) = K                     (đúng K câu)
     phân bố Bloom ≈ đặc tả ±1      (slack)
     mỗi title ≥ 1 câu              (phủ chủ đề)
     mỗi context_id ≤ 2 câu         (không lặp ngữ cảnh)
     x(i) ∈ {0,1}
Solver: PuLP/CBC, timeLimit=30s
```

### 8.2. So sánh 3 chiến lược

| Chiến lược | Score | Tuân thủ Bloom | Thời gian |
|---|---|---|---|
| ILP (tối ưu) | Cao nhất | ✅ chính xác ±1 | ~30s |
| Greedy | ~95% ILP | ⚠️ xấp xỉ | <1ms |
| Random | Thấp nhất | ❌ ngẫu nhiên | <1ms |

---

## 9. Kết quả so sánh

### 9.1. Bảng so sánh chính (số liệu thật — lần chạy cuối cùng)

| Tiêu chí | Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|---|
| **VQR** | 27.9% | **62.2%** | 45.1% |
| **Avg verifier_score** | 0.614 | **0.803** | 0.717 |
| Evidence Match | 0.356 | **0.951** | 0.799 |
| Single Correct | 0.278 | **0.594** | 0.374 |
| Distractor Type | **0.920** | 0.740 | 0.740 |
| Historical Correctness | **0.977** | 0.838 | 0.708 |
| Question Clarity | 0.988 | **1.000** | 0.992 |
| Bloom Fidelity | **0.926** | 0.862 | 0.751 |
| Duplicate Check | 0.080 | 0.650 | **0.733** |
| **Latency** | **8ms** | 5,114ms | ~200ms |
| **Chi phí/câu** | **$0** | $0.00019 | **$0** |
| Offline | ✅ | ❌ | ✅ |
| **TOPSIS score** | 0.1584 | **0.8894** | 0.5417 |

### 9.2. Kỳ vọng vs Thực tế

| Phương pháp | Kỳ vọng | Thực tế | Nhận xét |
|---|---|---|---|
| Rule-Based | VQR~60% | VQR=27.9% | Template trùng → Duplicate=0.080 |
| RAG+LLM | VQR~80% | VQR=62.2% | Evidence=0.951 vượt kỳ vọng ✅ |
| ViT5 | VQR~70% | VQR=45.1% | DG kém → Single Correct=0.374 |

### 9.3. Kết luận DSS

> Không có phương pháp nào thắng trên mọi tiêu chí.
> Chiến lược tối ưu là **hybrid routing**:
>
> - **Thời gian/Nhân vật/Địa điểm + offline + $0** → Rule-Based
> - **Nguyên nhân/Ý nghĩa + ưu tiên chất lượng** → RAG+LLM
> - **Số lượng lớn + offline** → ViT5-QG + RAG-DG (hybrid)
>
> Thứ hạng TOPSIS ổn định (CR=0.023, không có điểm giao khi VQR 1%→50%).

---

## 10. Hệ thống Dashboard

Streamlit 5 trang: Tổng quan | Demo Sinh Câu Hỏi (nhập tự do) |
Ngân hàng | AHP/TOPSIS realtime (kéo slider trọng số) | Ráp đề thi.

---

## 11. Danh sách lệnh đã chạy

### Cài đặt
```bash
pip install datasets pandas pyarrow pydantic
pip install transformers sentencepiece
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate faiss-cpu sentence-transformers rank-bm25 openai
pip install scikit-learn joblib pulp streamlit plotly
```

### Bước 1 — Build corpus
```bash
# Khảo sát 184 bài ViQuAD
python src/preprocess/survey_viquad.py --outdir data/processed
# [THỦ CÔNG] Đánh is_history, is_vietnam cho 184 bài trong titles_inventory.csv
# Build corpus chính thức
python src/preprocess/build_corpus.py \
    --inventory data/processed/titles_inventory.csv \
    --outdir data/processed
# Kết quả: corpus.parquet (1,862), qa_pairs.parquet (13,997), splits_report.txt
```

### Bước 2 — NER + KB
```bash
# Lần 1: underthesea → LOC~41% CHƯA ĐẠT
python src/preprocess/ner_kb.py --corpus data/processed/corpus.parquet \
    --outdir data/processed --backend underthesea
# [Phát hiện lỗi] → chuyển sang ELECTRA
# Lần 2: ELECTRA + clean_entities → LOC 97% ĐẠT
python src/preprocess/ner_kb.py --corpus data/processed/corpus.parquet \
    --outdir data/processed --backend electra
# Kết quả: entities.parquet (25,325), distractor_pool.pkl (37 nhóm)
# [THỦ CÔNG] Chấm tay 100 mẫu → YEAR=100%, LOC=97%, PER=90%, ORG=80%
```

### Bước 3 — Bloom Classifier từ VNHSGE
```bash
# Tải 50 file JSON lịch sử từ GitHub + gán nhãn Bloom + train classifier
python src/preprocess/train_bloom_classifier.py --outdir models/bloom
# Kết quả: bloom_classifier.pkl, accuracy=90% trên 200 mẫu eval
```

### Bước 4 — Rule-Based Generator
```bash
python src/generators/rule_based.py \
    --corpus data/processed/corpus.parquet \
    --entities data/processed/entities.parquet \
    --pool data/processed/distractor_pool.pkl \
    --split train --limit 200 \
    --wh_types thoi_gian,nhan_vat,dia_diem
# Kết quả: rule_based.jsonl (423 câu), VQR thô=70%, 8ms/câu, $0
```

### Bước 5 — RAG+LLM Generator
```bash
# Build index (chạy 1 lần, ~10 phút)
python src/generators/build_index.py \
    --corpus data/processed/corpus.parquet --outdir data/index
# Sinh câu hỏi
set OPENAI_API_KEY=sk-...
python src/generators/rag_llm.py \
    --n_topics 10 --wh_types thoi_gian,nhan_vat,nguyen_nhan \
    --bloom nhan_biet --top_k 3
# Kết quả: rag_llm.jsonl (90 câu), VQR thô=100%, $0.0174
```

### Bước 6 — ViT5 Fine-tuning
```bash
# Train Stage A (QG) + Stage B (DG), ~5 giờ RTX 3050
# Có checkpoint resume nếu máy tắt giữa chừng
python src/generators/train_vit5.py \
    --data_dir data/processed --outdir models \
    --stage AB --epochs 5 --batch 4 --accum 8
# Stage A best dev loss: ~1.78 | Stage B: ~1.99
# Inference trên test set
python src/generators/infer_vit5.py \
    --qg_model models/vit5_qg --dg_model models/vit5_dg \
    --data_dir data/processed \
    --out data/generated/vit5_ft.jsonl --limit 300
# Kết quả: vit5_ft.jsonl (297 câu)
```

### Bước 7 — Verifier
```bash
# Chạy Verifier (GPT cho NLI + Bloom, VNHSGE classifier làm fallback)
set OPENAI_API_KEY=sk-...
python src/verifier/verifier.py \
    --inputs data/generated/rule_based.jsonl \
             data/generated/rag_llm.jsonl \
             data/generated/vit5_ft.jsonl \
    --entities data/processed/entities.parquet \
    --outdir data/verified --bloom_backend gpt --no_embed
# Kết quả: verified_*.jsonl, results_table.csv
# So sánh VNHSGE vs GPT:
python src/verifier/verifier.py ... --bloom_backend both
# Agreement=76%, GPT tốt hơn 12% nhưng chi phí chỉ $0.0009/90 câu
```

### Bước 8 — AHP/TOPSIS
```bash
python src/dss/ahp_topsis.py \
    --results data/verified/results_table.csv --outdir data/dss
# CR=0.0230 ✅, RAG(0.8894)>ViT5(0.5417)>Rule(0.1584)
# Thứ hạng ổn định (không có điểm giao khi VQR 1%→50%)
```

### Bước 9 — Ráp đề thi
```bash
python src/dss/exam_builder.py \
    --verified_dir data/verified --n_questions 40 \
    --bloom_ratio 0.4,0.35,0.25 --outdir data/exams
# Kết quả: exam_ilp.csv, exam_greedy.csv, exam_random.csv, strategy_comparison.csv
```

### Bước 10 — Dashboard
```bash
streamlit run src/app/dashboard.py
# Mở http://localhost:8501
```

---

## 12. Kết luận và hướng phát triển

### 12.1. Đóng góp chính

1. Pipeline DSS hoàn chỉnh 3 tầng với giải thích quyết định tường minh
2. Verifier 8 tiêu chí dùng nhãn vàng `is_impossible` — không cần gán tay
3. VNHSGE-History được tích hợp thực sự: train Bloom classifier accuracy=90%
4. Phát hiện kỹ thuật: NER LOC 41%→97%, ViT5 DG kém do dữ liệu train tự động
5. Kết luận DSS định lượng: hybrid routing tốt hơn một phương pháp đơn độc

### 12.2. Hạn chế

- van_dung_cao F1=0% trong Bloom classifier do chỉ có 11 mẫu train
- ViT5 Stage B DG kém — cần dữ liệu distractor do người viết
- Human eval chưa thực hiện (chỉ Verifier tự động)
- Distractor pool thiếu cho tên nước ngoài corpus thế giới

### 12.3. Hướng phát triển

- Human eval 50 câu/phương pháp với Cohen's kappa
- Hybrid routing tự động theo wh_type
- Tăng cường dữ liệu DG bằng augmentation hoặc gán tay
- Mở rộng sang môn học khác (Địa lý, Văn học)

---

## PHỤ LỤC: Cấu trúc thư mục

```
files_v1/
├── BAO_CAO.md
├── README.md
├── eda_notebook.ipynb
├── src/
│   ├── schema/        mcq_schema.py
│   ├── preprocess/    survey_viquad.py, build_corpus.py, split_balanced.py,
│   │                  ner_kb.py, clean_entities.py, train_bloom_classifier.py
│   ├── generators/    rule_based.py, build_index.py, rag_llm.py,
│   │                  train_vit5.py, infer_vit5.py
│   ├── verifier/      verifier.py
│   ├── dss/           ahp_topsis.py, exam_builder.py
│   └── app/           dashboard.py
├── data/
│   ├── processed/     corpus.parquet, qa_pairs.parquet, entities.parquet,
│   │                  distractor_pool.pkl, titles_inventory.csv
│   ├── vnhsge/        *.json (50 file lịch sử 2019–2023)
│   ├── index/         faiss.index, bm25.pkl, index_meta.parquet
│   ├── generated/     rule_based.jsonl, rag_llm.jsonl, vit5_ft.jsonl
│   ├── verified/      verified_*.jsonl, results_table.csv
│   ├── dss/           ahp_weights.csv, topsis_ranking.csv,
│   │                  sensitivity.csv, dss_report.txt
│   └── exams/         exam_ilp.csv, strategy_comparison.csv
├── models/
│   ├── bloom/         bloom_classifier.pkl (VNHSGE, acc=90%)
│   ├── vit5_qg/       Stage A (dev loss ~1.78)
│   └── vit5_dg/       Stage B (dev loss ~1.99)
└── tests/
    test_schema.py, test_build.py, test_survey.py,
    test_ner.py, test_clean.py, test_rule_based.py,
    test_rag_llm.py, test_verifier.py
```

---
*Mọi số liệu trong báo cáo đều từ kết quả chạy thật. Tháng 7/2026.*
