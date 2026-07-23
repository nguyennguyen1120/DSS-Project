# DSS for Educational MCQ Generation — Hệ Hỗ Trợ Ra Quyết Định Sinh Câu Hỏi Trắc Nghiệm

> **Đồ án môn:** Hệ Hỗ Trợ Ra Quyết Định  
> **Đề tài:** So sánh các phương pháp sinh câu hỏi trắc nghiệm lịch sử, hỗ trợ giáo viên ra quyết định lựa chọn phương pháp và ráp đề thi

---

## Mục lục
1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Dataset](#3-dataset)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Hướng dẫn chạy từng bước](#6-hướng-dẫn-chạy-từng-bước)
7. [Kết quả hiện tại](#7-kết-quả-hiện-tại)
8. [Các bước dự kiến tiếp theo](#8-các-bước-dự-kiến-tiếp-theo)
9. [Schema câu hỏi](#9-schema-câu-hỏi)
10. [Ghi chú kỹ thuật](#10-ghi-chú-kỹ-thuật)

---

## 1. Tổng quan bài toán

### Input (2 nguồn)
| Nguồn | Mô tả | Ví dụ |
|---|---|---|
| **Ngữ cảnh** | Đoạn văn lịch sử | "Chiến thắng Điện Biên Phủ diễn ra năm 1954…" |
| **Loại câu hỏi** | `(Bloom level, Wh-type)` | `(nhan_biet, thoi_gian)` |

### Output
```
Câu hỏi: Chiến thắng Điện Biên Phủ diễn ra vào năm nào?
A. 1945    B. 1954 ←    C. 1975    D. 1986
Đáp án đúng: B
Metadata: chủ đề, loại câu hỏi, độ khó, điểm verifier
```

### Ba phương pháp so sánh
| Phương pháp | Cơ chế | Offline? | Chi phí |
|---|---|---|---|
| **Rule-Based** | Template + KB thực thể | ✅ | $0 |
| **RAG + LLM** | Hybrid retrieval + GPT-4o-mini | ❌ | ~$0.19/1k câu |
| **ViT5 Fine-tuning** | Seq2Seq answer-aware QG | ✅ | $0 |

### Ba tầng DSS
1. **Chọn phương pháp** → AHP + TOPSIS + phân tích độ nhạy
2. **Kiểm định câu hỏi** → Verifier 8 tiêu chí → `accepted/needs_review/rejected`
3. **Ráp đề** → ILP tối ưu có ràng buộc theo ma trận đặc tả Bloom

---

## 2. Kiến trúc hệ thống

```
corpus.parquet (1,862 đoạn lịch sử)
        │
        ├─► Rule-Based Generator  ──────────┐
        │   (rule_based.py)                 │
        │                                   ▼
        ├─► RAG+LLM Generator  ──────► Verifier  ──► results_table.csv
        │   (rag_llm.py)                    ▲              │
        │                                   │              ▼
        └─► ViT5-FT Generator  ────────────┘        AHP / TOPSIS
            (vit5_train_colab_v2.ipynb)                    │
                                                           ▼
                                                    DSS Dashboard
                                                    (Streamlit)
```

**Nguyên tắc so sánh công bằng:**
- Cả 3 phương pháp nhận **cùng** `(context, wh_type, bloom)` làm input
- Cùng đi qua **một** Verifier duy nhất
- RAG chỉ index `train+dev`, **không** index `test` (chống rò rỉ)

---

## 3. Dataset

### 3.1. Nguồn dữ liệu

| Dataset | Vai trò | Truy cập |
|---|---|---|
| **UIT-ViQuAD 2.0** (`taidng/UIT-ViQuAD2.0`) | Corpus chính: ngữ cảnh sinh câu hỏi + train ViT5 | HuggingFace, `load_dataset()` |
| **VNHSGE — môn Lịch sử** | Gold MCQ + nhãn Bloom (dự kiến) | GitHub |

### 3.2. Thống kê corpus lịch sử (sau khi lọc)

| Thông số | Giá trị |
|---|---|
| Số bài Wikipedia giữ lại | 38 bài (lịch sử VN + thế giới) |
| Tổng đoạn văn | **1,862 đoạn** |
| Câu trả lời được | 10,114 câu |
| Câu **không** trả lời được (`is_impossible`) | **3,883 câu** (nhãn vàng cho Evidence Match) |
| Split train/dev/test | **80.1% / 9.6% / 10.3%** |
| Tiểu miền VN | 536 đoạn (9 bài) |
| Tiểu miền thế giới | 1,326 đoạn (29 bài) |

> **Lưu ý chia split:** chia theo `title` (cấp bài), không theo câu hỏi, dùng thuật toán greedy cân tải để tránh lệch phân bố. Mọi split đều có cả hai tiểu miền VN và thế giới.

### 3.3. Phân loại câu hỏi (Wh-type taxonomy)

| Loại | Wh-type | Answer type | Rule-based xử lý được? |
|---|---|---|---|
| Thời gian | `thoi_gian` | YEAR / DATE | ✅ |
| Nhân vật | `nhan_vat` | PER | ✅ |
| Địa điểm | `dia_diem` | LOC | ✅ |
| Sự kiện | `su_kien` | EVENT | ⚠️ hạn chế |
| Nguyên nhân | `nguyen_nhan` | CLAUSE | ❌ non-factoid |
| Ý nghĩa | `y_nghia` | CLAUSE | ❌ non-factoid |

### 3.4. Knowledge Base thực thể

| Type | Số lượng (sau lọc) | Bucket |
|---|---|---|
| LOC | 12,045 | VN / TG |
| PER | 7,626 | VN / TG |
| YEAR | 2,922 | Thế kỷ (10C, 14C, 20C…) |
| ORG | 2,252 | VN / TG |
| DATE | 339 | Thế kỷ |
| CENTURY | 141 | — |

**NER pipeline:** `NlpHUST/ner-vietnamese-electra-base` + regex thời gian + `clean_entities.py`  
**Precision sau lọc (chấm tay 100 mẫu):** YEAR=100%, LOC=97%, PER=90%, ORG=80%

---

## 4. Cấu trúc thư mục

```
files_v1/
├── README.md
├── setup.py                          # pip install -e . (không bắt buộc)
│
├── src/
│   ├── schema/
│   │   └── mcq_schema.py             # Pydantic schema dùng chung 3 phương pháp
│   │
│   ├── preprocess/
│   │   ├── survey_viquad.py          # Bước 1a: khảo sát 184 bài ViQuAD
│   │   ├── build_corpus.py           # Bước 1b: lọc lịch sử + chia split cân tải
│   │   ├── split_balanced.py         # Thuật toán greedy cân tải theo title
│   │   ├── ner_kb.py                 # Bước 2: NER + xây KB + distractor pool
│   │   ├── clean_entities.py         # Lớp lọc luật 7 khuôn mẫu lỗi NER
│   │   ├── extract_corpus.py         # (cũ, dùng cho Vietnam-History-200K-Vi)
│   │   └── review_sample.py          # Lấy mẫu 50 đoạn chấm chất lượng
│   │
│   └── generators/
│       ├── rule_based.py             # Phương pháp 1: Template + KB
│       ├── build_index.py            # Xây FAISS + BM25 index (chạy 1 lần)
│       └── rag_llm.py               # Phương pháp 2: Hybrid RAG + GPT-4o-mini
│
├── data/
│   ├── processed/
│   │   ├── titles_inventory.csv      # 184 bài ViQuAD, cột is_history + is_vietnam
│   │   ├── corpus.parquet            # 1,862 đoạn lịch sử + split + is_vietnam
│   │   ├── qa_pairs.parquet          # 13,997 cặp QA + answer_start + is_impossible
│   │   ├── entities.parquet          # 25,325 thực thể đã lọc
│   │   ├── distractor_pool.pkl       # {(type, bucket) -> set(surface)}
│   │   ├── splits_report.txt         # Báo cáo chia split + kiểm tra rò rỉ
│   │   └── ner_quality_sample.csv    # 100 mẫu NER đã chấm tay
│   │
│   ├── index/                        # FAISS + BM25 (sau khi chạy build_index.py)
│   │   ├── faiss.index
│   │   ├── bm25.pkl
│   │   └── index_meta.parquet
│   │
│   └── generated/
│       ├── rule_based.jsonl          # ✅ 423 câu hỏi đã sinh
│       ├── rag_llm.jsonl             # ✅ 90 câu hỏi đã sinh
│       └── vit5_ft.jsonl             # ⏳ (sau khi train xong trên Colab)
│
├── models/                           # Sau khi train ViT5 (copy từ Drive)
│   ├── vit5_qg/                      # Stage A: Question Generation
│   └── vit5_dg/                      # Stage B: Distractor Generation
│
├── notebooks/
│   └── vit5_train_colab_v2.ipynb    # Train ViT5 trên Google Colab A100
│
└── tests/
    ├── test_schema.py
    ├── test_build.py
    ├── test_survey.py
    ├── test_ner.py
    ├── test_clean.py
    ├── test_rule_based.py
    └── test_rag_llm.py
```

---

## 5. Cài đặt môi trường

```bash
# Python 3.10+
pip install datasets pandas pyarrow pydantic
pip install underthesea transformers torch
pip install faiss-cpu sentence-transformers rank-bm25
pip install openai
```

**Biến môi trường:**
```bash
# Windows CMD
set OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

---

## 6. Hướng dẫn chạy từng bước

### Bước 1 — Khảo sát và build corpus

```bash
# 1a. Khảo sát 184 bài ViQuAD, xuất titles_inventory.csv
python src/preprocess/survey_viquad.py --outdir data/processed

# 1b. Mở data/processed/titles_inventory.csv
#     Rà tay: đánh is_history (1/0) và is_vietnam (1/0) cho 184 bài
#     Chỉ mất ~15 phút

# 1c. Build corpus chính thức (sau khi đã sửa titles_inventory.csv)
python src/preprocess/build_corpus.py \
    --inventory data/processed/titles_inventory.csv \
    --outdir data/processed
# Kết quả: corpus.parquet (1,862 đoạn), qa_pairs.parquet (13,997 câu)
# Kiểm tra: splits_report.txt — phải thấy "rò rỉ = 1 split ✅"
```

### Bước 2 — NER + Knowledge Base

```bash
# Cài thêm
pip install transformers sentencepiece

# Chạy NER với ELECTRA backend (khuyến nghị)
python src/preprocess/ner_kb.py \
    --corpus data/processed/corpus.parquet \
    --outdir data/processed \
    --backend electra
# Kết quả: entities.parquet, distractor_pool.pkl, ner_quality_sample.csv

# Chấm tay: mở ner_quality_sample.csv, điền cột correct (1/0) cho 100 dòng
# Ngưỡng: YEAR>=95%, PER/LOC>=85%
```

### Bước 3 — Phương pháp 1: Rule-Based Generator

```bash
python src/generators/rule_based.py \
    --corpus data/processed/corpus.parquet \
    --entities data/processed/entities.parquet \
    --pool data/processed/distractor_pool.pkl \
    --split train \
    --limit 200 \
    --wh_types thoi_gian,nhan_vat,dia_diem
# Kết quả: data/generated/rule_based.jsonl
```

**Kết quả đạt được:**
- VQR = 70% (423/600 lượt thử)
- Evidence found_in_context = 97%
- Latency = 8ms/câu, Chi phí = $0

### Bước 4 — Phương pháp 2: RAG+LLM Generator

```bash
# 4a. Build FAISS + BM25 index (chạy 1 lần, ~5-10 phút)
pip install faiss-cpu sentence-transformers rank-bm25
python src/generators/build_index.py \
    --corpus data/processed/corpus.parquet \
    --outdir data/index
# Kiểm tra: phải thấy "test split KHÔNG nằm trong index"

# 4b. Sinh câu hỏi (cần OPENAI_API_KEY)
set OPENAI_API_KEY=sk-...
python src/generators/rag_llm.py \
    --n_topics 10 \
    --wh_types thoi_gian,nhan_vat,nguyen_nhan \
    --bloom nhan_biet \
    --top_k 3
# Kết quả: data/generated/rag_llm.jsonl
```

**Kết quả đạt được:**
- VQR = 100% (90/90)
- Evidence found_in_context = 82%
- Latency = ~5,114ms/câu, Chi phí = $0.0174 / 90 câu

### Bước 5 — Phương pháp 3: ViT5 Fine-tuning (Google Colab)

```
1. Upload thư mục files_v1/ lên Google Drive
   (bao gồm data/processed/qa_pairs.parquet và corpus.parquet)

2. Upload vit5_train_colab_v2.ipynb lên Drive

3. Mở notebook bằng Google Colaboratory
   Runtime → Change runtime type → A100 GPU

4. Sửa dòng PROJECT_DIR trong Cell 3:
   PROJECT_DIR = '/content/drive/MyDrive/00.Ths/HHTRQD/files_v1'

5. Runtime → Restart session → Run All
   Thời gian: ~3-4 giờ (Stage A ~2h + Stage B ~1h + Inference ~30 phút)

6. Sau khi xong, copy về máy:
   Drive: files_v1/data/generated/vit5_ft.jsonl
   Drive: files_v1/models/vit5_qg/
   Drive: files_v1/models/vit5_dg/
```

### Bước 6 — Verifier (Dự kiến)

```bash
# Chạy Verifier trên cả 3 file .jsonl
python src/verifier/verifier.py \
    --inputs data/generated/rule_based.jsonl \
             data/generated/rag_llm.jsonl \
             data/generated/vit5_ft.jsonl \
    --outdir data/verified
# Kết quả: verified_rule_based.jsonl, verified_rag_llm.jsonl, verified_vit5_ft.jsonl
#          mỗi item có trường verification.verifier_score + violations[]
```

### Bước 7 — Đánh giá và DSS (Dự kiến)

```bash
# Tính 8 tiêu chí so sánh
python src/evaluation/evaluate.py --verified_dir data/verified

# AHP + TOPSIS
python src/dss/ahp_topsis.py

# Ráp đề bằng ILP
python src/dss/exam_builder.py --n_questions 40 --bloom_ratio 0.3,0.4,0.3

# Dashboard
streamlit run src/app/dashboard.py
```

---

## 7. Kết quả hiện tại

### So sánh 3 phương pháp (kết quả thô, chưa qua Verifier)

| Tiêu chí | Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|---|
| VQR (%) | **70%** | **100%** | ⏳ |
| Evidence in context | **97%** | 82% | ⏳ |
| Sinh được Nguyên nhân/Ý nghĩa | ❌ | ✅ | ✅ |
| Latency (ms/câu) | **8ms** | 5,114ms | ⏳ |
| Chi phí/câu | **$0** | $0.00019 | **$0** |
| Chạy offline | ✅ | ❌ | ✅ |

### Chất lượng KB thực thể (chấm tay 100 mẫu)

| Type | Precision | Đạt ngưỡng? |
|---|---|---|
| YEAR | 100% | ✅ (ngưỡng 95%) |
| LOC | 97% | ✅ (ngưỡng 85%) |
| PER | 90% | ✅ (ngưỡng 85%) |
| ORG | 80% | ✅ (ngưỡng 75%) |

### Tiến độ

```
✅ Bước 1: Survey + Build corpus (1,862 đoạn, split 80/10/10)
✅ Bước 2: NER + KB (25,325 thực thể, precision 90-100%)
✅ Bước 3: Rule-based generator (423 câu)
✅ Bước 4: RAG+LLM generator (90 câu)
⏳ Bước 5: ViT5 fine-tuning (đang train trên Colab)
⬜ Bước 6: Verifier (8 tiêu chí)
⬜ Bước 7: Đánh giá + AHP/TOPSIS
⬜ Bước 8: ILP ráp đề
⬜ Bước 9: Dashboard Streamlit
```

---

## 8. Các bước dự kiến tiếp theo

### Bước 6 — Verifier (8 tiêu chí kiểm chứng)

| # | Tiêu chí | Phương pháp | Nhãn vàng |
|---|---|---|---|
| 1 | **Evidence Match** | NLI (DeBERTa-MNLI): context entail câu trả lời | `is_impossible` của ViQuAD |
| 2 | **Single Correct Answer** | NLI cho cả 4 phương án: chỉ 1 entail | — |
| 3 | **Distractor Type Match** | NER type khớp + cosine ∈ [0.4, 0.9] | KB |
| 4 | **Historical Correctness** | Đối chiếu KB fact, flag `unverified` nếu thiếu | KB |
| 5 | **Question Clarity** | Heuristic (độ dài, đại từ mơ hồ) + perplexity | — |
| 6 | **Duplicate Check** | MinHash/LSH + cosine embedding ngưỡng 0.92 | — |
| 7 | **Bloom Fidelity** | Classifier train trên VNHSGE | VNHSGE |
| 8 | **Answer Position Balance** | Phân bố A/B/C/D ≈ 25% | — |

Output: `verifier_score ∈ [0,1]` + `status: accepted/needs_review/rejected`  
**VQR chính thức = #accepted / #generated** — tiêu chí so sánh chính.

### Bước 7 — AHP/TOPSIS

- **8 tiêu chí:** VQR, Evidence Match, Bloom Fidelity, Distractor Quality, BERTScore, Latency, Chi phí, Độ bền miền (VN vs thế giới)
- Khảo sát trọng số từ 3-5 giáo viên/kỹ sư → tổng hợp bằng trung bình hình học
- Bắt buộc báo cáo **Consistency Ratio (CR < 0.1)**
- Chạy thêm **SAW và VIKOR**, so thứ hạng bằng Kendall's tau
- **Phân tích độ nhạy:** trọng số Chi phí từ 5% → 50%, tìm điểm giao quyết định đổi

### Bước 8 — ILP ráp đề

```python
# Bài toán tối ưu:
# max  Σ verifier_score(i) · x(i)
# s.t. Σ x(i) = K (số câu yêu cầu)
#      phân bố Bloom khớp ma trận đặc tả (±1 câu)
#      mỗi chủ đề ≥ 1 câu
#      mỗi context_id đóng góp ≤ 2 câu
#      x(i) ∈ {0,1}
# Thư viện: PuLP
# So sánh: ILP vs Greedy vs Genetic Algorithm
```

### Bước 9 — Dashboard Streamlit

Tính năng dự kiến:
- Sinh câu hỏi theo `(context, Bloom, Wh-type)` từ 3 phương pháp
- Xem `verifier_score` + cờ vi phạm cho từng câu
- Kéo trọng số AHP → thứ hạng TOPSIS đổi realtime
- Ráp đề và xuất file theo ma trận đặc tả

---

## 9. Schema câu hỏi

Tất cả 3 phương pháp đều trả về **cùng một định dạng** (`MCQItem` — `src/schema/mcq_schema.py`):

```json
{
  "schema_version": "1.0",
  "item_id": "itm_7f3a9c21",
  "generator": {
    "method": "rule_based | rag_llm | vit5_ft",
    "variant": "template_v1",
    "model_name": null
  },
  "source": {
    "context_id": "ctx_00231",
    "context": "Chiến thắng Điện Biên Phủ diễn ra năm 1954…",
    "title": "Chiến dịch Điện Biên Phủ",
    "is_vietnam": true
  },
  "request": {
    "bloom_requested": "nhan_biet",
    "wh_type_requested": "thoi_gian"
  },
  "question": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
  "options": [
    {"label": "A", "text": "1945", "is_correct": false, "provenance": "kb"},
    {"label": "B", "text": "1954", "is_correct": true,  "provenance": "context_span"},
    {"label": "C", "text": "1975", "is_correct": false, "provenance": "kb"},
    {"label": "D", "text": "1986", "is_correct": false, "provenance": "kb"}
  ],
  "answer_key": "B",
  "answer_text": "1954",
  "evidence": {
    "sentence": "Chiến thắng Điện Biên Phủ diễn ra năm 1954…",
    "found_in_context": true
  },
  "metadata": {
    "wh_type_detected": "thoi_gian",
    "bloom_predicted": "nhan_biet",
    "difficulty_band": "de",
    "topic": "Chiến dịch Điện Biên Phủ"
  },
  "generation_trace": {
    "latency_ms": 8.2,
    "cost_usd": 0.0,
    "n_llm_calls": 0
  },
  "verification": null
}
```

**4 bất biến được ép bởi Pydantic validator:**
1. Đúng 4 phương án, nhãn A/B/C/D, không trùng nội dung
2. Đúng 1 đáp án đúng (3 distractor)
3. `answer_key` và `answer_text` khớp phương án `is_correct=True`
4. `evidence.sentence` phải là substring nguyên văn của `context` khi `found_in_context=True`

---

## 10. Ghi chú kỹ thuật

### Chống rò rỉ dữ liệu (quan trọng)
- Split **theo `title` (cấp bài)**, không theo câu hỏi
- RAG chỉ index `train+dev`, **tuyệt đối không index `test`**
- `splits_report.txt` tự kiểm tra: `rò rỉ = 1 split ✅`
- Cùng một `titles_inventory.csv` và `corpus.parquet` cho cả 3 phương pháp

### Vấn đề đã gặp và cách xử lý
| Vấn đề | Giải pháp |
|---|---|
| `Vietnam-History-200K-Vi` chỉ còn 176 đoạn sau khử trùng | Chuyển sang ViQuAD 2.0 |
| NER `underthesea` LOC precision ~41% | Đổi sang `NlpHUST/ner-vietnamese-electra-base` + `clean_entities.py` |
| Split hash mù lệch 87/6.5/6.5 | Thuật toán greedy cân tải (`split_balanced.py`) |
| ViT5 `EncoderDecoderCache` lỗi trên Colab mới | Training loop thủ công (bỏ `Seq2SeqTrainer`) |
| `input_ids > vocab_size` → CUDA assert | `resize_token_embeddings(len(tokenizer))` + clamp input_ids |
| `năm 1954` xuất hiện trong phương án thay vì `1954` | `_normalize_surface()` bỏ tiền tố "năm " |

### Test suite
Tất cả module có test độc lập, chạy offline không cần GPU hay API:
```bash
python test_schema.py      # Schema + 4 bất biến
python test_build.py       # Corpus build + chống rò rỉ
python test_survey.py      # Khảo sát ViQuAD
python test_ner.py         # NER + KB + bucket
python test_clean.py       # Lớp lọc 7 khuôn mẫu lỗi
python test_rule_based.py  # Rule-based generator end-to-end
python test_rag_llm.py     # RAG logic (không cần OpenAI API)
```
