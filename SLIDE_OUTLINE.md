# ĐỀ CƯƠNG SLIDE BÁO CÁO CUỐI KỲ
# DSS for Educational MCQ Generation

**Môn học:** Hệ Hỗ Trợ Ra Quyết Định  
**Số slide dự kiến:** 30–35 slide  
**Thời gian trình bày:** 15–20 phút  

---

## PHẦN 1: MỞ ĐẦU (4 slide)

---

### Slide 1 — Trang bìa

```
[LOGO TRƯỜNG]

HỆ HỖ TRỢ RA QUYẾT ĐỊNH
SINH CÂU HỎI TRẮC NGHIỆM LỊCH SỬ

DSS for Educational MCQ Generation

[Tên nhóm] | [Tên giảng viên] | Tháng 7/2026
```

---

### Slide 2 — Bối cảnh và động lực

**Tiêu đề:** Tại sao cần hệ thống này?

**Nội dung:**

- ❌ **Thách thức hiện tại:**
  - Giáo viên tốn 2–3 giờ để soạn 40 câu hỏi một đề thi
  - Khó đảm bảo phân bố Bloom đúng ma trận đặc tả
  - Câu hỏi dễ trùng lặp khi làm thủ công

- ✅ **Giải pháp đề xuất:**
  - Hệ thống DSS tự động sinh, kiểm định và ráp đề thi
  - Hỗ trợ giáo viên **ra quyết định**: chọn phương pháp phù hợp

**Hình minh hoạ:** Sơ đồ so sánh quy trình thủ công vs tự động (2 cột)

---

### Slide 3 — Tổng quan hệ thống DSS 3 tầng

**Tiêu đề:** Kiến trúc DSS 3 tầng

**Hình chính (chiếm 70% slide):**

```
┌─────────────────────────────────────────────────┐
│  INPUT: Ngữ cảnh lịch sử + Loại câu hỏi         │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│  TẦNG 1 — Sinh câu hỏi                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Rule-Based│  │ RAG+LLM  │  │  ViT5-FT │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│  TẦNG 2 — Kiểm định + DSS                       │
│  Verifier 8 tiêu chí → AHP/TOPSIS               │
│  → Khuyến nghị phương pháp                      │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│  TẦNG 3 — Ráp đề thi (ILP)                     │
│  Tối ưu có ràng buộc Bloom                      │
└─────────────────────────────────────────────────┘
```

---

### Slide 4 — Mục lục / Roadmap trình bày

**Tiêu đề:** Nội dung trình bày

| # | Phần | Nội dung |
|---|---|---|
| 1 | Dữ liệu | 2 nguồn, EDA |
| 2 | Phương pháp | 3 cách sinh MCQ |
| 3 | Verifier | 8 tiêu chí kiểm định |
| 4 | DSS | AHP/TOPSIS + ILP |
| 5 | Kết quả | So sánh định lượng |
| 6 | Demo | Dashboard live |

---

## PHẦN 2: DỮ LIỆU (5 slide)

---

### Slide 5 — Nguồn dữ liệu 1: UIT-ViQuAD 2.0

**Tiêu đề:** Nguồn 1 — UIT-ViQuAD 2.0

**Bố cục 2 cột:**

*Cột trái — Thông tin:*
- Dataset: `taidng/UIT-ViQuAD2.0` (HuggingFace)
- 39,569 câu hỏi, 184 bài Wikipedia
- Sau lọc lịch sử: **1,862 đoạn**
- **3,883 câu `is_impossible`** → nhãn vàng Evidence Match

*Cột phải — Ví dụ mẫu JSON:*
```json
{
  "context": "Chiến thắng Điện Biên Phủ...",
  "question": "Năm nào chiến thắng ĐBP?",
  "answer": {"text": "1954", "answer_start": 42},
  "is_impossible": false
}
```

**Điểm nhấn (màu nổi):** `answer_start` → Answer-aware QG miễn phí

---

### Slide 6 — Nguồn dữ liệu 2: VNHSGE-History

**Tiêu đề:** Nguồn 2 — VNHSGE môn Lịch sử

**Bố cục 2 cột:**

*Cột trái — Thông tin:*
- GitHub: `Xdao85/VNHSGE` (JSON format)
- **2,000 câu** đề thi THPT Quốc gia (2019–2023)
- Schema: `{ID, Question, Choice, Explanation}`
- **Không có cột Bloom sẵn** → gán bằng heuristic từ khoá

*Cột phải — Vai trò trong hệ thống:*
```
VNHSGE-History
      ↓
Gán nhãn Bloom (heuristic)
      ↓
Train TF-IDF + LogReg
      ↓
Bloom Classifier (acc=90%)
      ↓
Tích hợp vào Verifier
```

---

### Slide 7 — EDA: Corpus lịch sử

**Tiêu đề:** Phân tích dữ liệu — Corpus ViQuAD lịch sử

**3 biểu đồ nhỏ (lấy từ `eda_corpus_overview.png`):**

1. **Bar chart:** Split distribution (train=1,491 / dev=179 / test=192)
2. **Pie chart:** VN (28.8%) vs Thế giới (71.2%)
3. **Histogram:** n_tokens (median=171, range=100–774)

**Box thống kê quan trọng:**
```
1,862 đoạn | 13,997 câu hỏi
3,883 unanswerable (27.7%) → nhãn vàng
year_density median = 0.65 năm/100 token
```

---

### Slide 8 — EDA: Knowledge Base + VNHSGE Bloom

**Tiêu đề:** Phân tích dữ liệu — KB thực thể & Bloom VNHSGE

**Bố cục 2 cột:**

*Cột trái — NER Precision (bar chart):*
```
YEAR  ████████████  100% ✅
LOC   ████████████   97% ✅
PER   ███████████    90% ✅
ORG   ██████████     80% ✅
```
*underthesea cũ: LOC ~41% ❌ → ELECTRA: 97% ✅*

*Cột phải — Phân bố Bloom VNHSGE:*
```
Nhận biết    ███████████  67.5%
Thông hiểu   ████         26.2%
Vận dụng     █             5.8%
V.dụng cao   ░             0.5%
```
*Mất cân bằng → `class_weight='balanced'`*

---

### Slide 9 — Pipeline tiền xử lý

**Tiêu đề:** Pipeline tiền xử lý dữ liệu

**Sơ đồ dọc (flow chart):**

```
ViQuAD 2.0 (HuggingFace)
       ↓ survey_viquad.py
titles_inventory.csv (184 bài)
       ↓ [Rà tay 15 phút] is_history / is_vietnam
       ↓ build_corpus.py + split_balanced.py
corpus.parquet (1,862)          qa_pairs.parquet (13,997)
       ↓ ner_kb.py (ELECTRA backend)
       ↓ clean_entities.py (7 luật lọc)
entities.parquet (25,325)       distractor_pool.pkl

VNHSGE-History (GitHub)
       ↓ train_bloom_classifier.py
bloom_classifier.pkl (acc=90%)
```

**Điểm nhấn:** Mũi tên đỏ chỉ vào "Greedy load-balancing → chống rò rỉ"

---

## PHẦN 3: 3 PHƯƠNG PHÁP SINH MCQ (6 slide)

---

### Slide 10 — Tổng quan 3 phương pháp

**Tiêu đề:** 3 Phương pháp Sinh MCQ

**Bảng so sánh đặc điểm (trước khi chạy):**

| | Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|---|
| **Cơ chế** | Template + KB | Retrieval + LLM | Seq2Seq |
| **Online** | Không | Cần API | Không |
| **Chi phí** | $0 | ~$0.0002/câu | $0 |
| **Tốc độ** | ~8ms | ~5s | ~200ms |
| **Non-factoid** | ❌ | ✅ | ✅ |

**Schema đầu ra chung (MCQItem)** — 4 bất biến Pydantic validator

---

### Slide 11 — Phương pháp 1: Rule-Based

**Tiêu đề:** Rule-Based — Template + Knowledge Base

**Sơ đồ pipeline:**
```
context + wh_type
    ↓
NER entities (YEAR/PER/LOC)
    ↓
Template chọn theo wh_type
    ↓ "... diễn ra vào năm nào?"
Distractor pool (cùng type, cùng bucket thế kỷ)
    ↓ fallback nới bucket nếu thiếu
finalize_options() → MCQItem
```

**Ví dụ thật (box màu):**
```
Q: Chiến thắng Điện Biên Phủ diễn ra vào năm nào?
A. 1945    B. 1954 ←    C. 1975    D. 1986
Latency: 8ms | Cost: $0
```

**Điểm yếu đã biết:** Duplicate Check = 0.080 (template cứng → câu trùng nhau)

---

### Slide 12 — Phương pháp 2: RAG+LLM

**Tiêu đề:** RAG+LLM — Hybrid Retrieval + GPT-4o-mini

**Sơ đồ pipeline:**
```
topic + wh_type
    ↓
BM25 (exact match)  +  FAISS/vi-bi-encoder (semantic)
              ↓ RRF (Reciprocal Rank Fusion)
         Top-3 context
              ↓
    GPT-4o-mini few-shot prompt
    (ép trả evidence_sentence nguyên văn)
              ↓
    parse + validate → MCQItem
```

**Ví dụ thật (box màu):**
```
Q: Nguyên nhân nào dẫn đến thất bại của Pháp tại ĐBP?
A. Thiếu vũ khí hiện đại
B. Bị cô lập hoàn toàn ←
C. Thiếu quân số
D. Thời tiết bất lợi
Cost: $0.00019 | Latency: 5,114ms
```

**Điểm mạnh:** Sinh được Nguyên nhân/Ý nghĩa (non-factoid)

---

### Slide 13 — Phương pháp 3: ViT5 Fine-tuning

**Tiêu đề:** ViT5 Fine-tuning — Seq2Seq Answer-Aware QG

**Bố cục 2 cột:**

*Cột trái — 2 Stage train:*
```
Stage A (QG):
Input:  <type> thoi_gian </type>
        <ans> 1954 </ans>
        <ctx> Chiến thắng ĐBP... </ctx>
Output: Chiến thắng ĐBP diễn ra năm nào?

Stage B (DG):
Input:  <q> Chiến thắng... năm nào? </q>
        <ans> 1954 </ans>
        <ctx> ... </ctx>
Output: 1945 | 1975 | 1986
```

*Cột phải — Thông số train:*
```
Model:  VietAI/vit5-base (250M)
GPU:    RTX 3050 4GB VRAM
Precision: fp16
Batch:  4 × accum 8 = eff.32
Epochs: 5 + early stopping
Resume: checkpoint sau mỗi epoch

Stage A dev loss: ~1.78
Stage B dev loss: ~1.99
```

**Hạn chế phát hiện:** DG không học format `|` → thêm fallback entity từ context

---

### Slide 14 — So sánh output 3 phương pháp (demo)

**Tiêu đề:** Cùng 1 ngữ cảnh — 3 phương pháp sinh ra gì?

**Ngữ cảnh:** *"Chiến thắng Điện Biên Phủ diễn ra ngày 7 tháng 5 năm 1954..."*  
**Yêu cầu:** wh_type = thoi_gian, bloom = nhan_biet

**3 cột song song:**

| Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|
| *Q: Chiến thắng ĐBP diễn ra vào năm nào?* | *Q: Chiến thắng ĐBP kết thúc kháng chiến chống Pháp vào năm nào?* | *Q: Chiến dịch ĐBP diễn ra vào năm nào?* |
| A.1945 B.1954✓ C.1975 D.1986 | A.1945 B.1975 C.1954✓ D.1965 | A.1954✓ B.Điện Biên C.1945 D.Võ Nguyên Giáp |
| 8ms, $0 | 5,114ms, $0.0002 | ~200ms, $0 |

**Nhận xét:** ViT5 distractor kém (lẫn tên người, địa danh vào phương án năm)

---

### Slide 15 — Schema MCQItem chung

**Tiêu đề:** Schema chung — MCQItem (Pydantic v2)

**Sơ đồ fields + 4 bất biến (box màu):**

```python
MCQItem:
  item_id      : "itm_7f3a9c21"
  generator    : {method: "rule_based|rag_llm|vit5_ft"}
  source       : {context_id, context, is_vietnam}
  request      : {bloom_requested, wh_type_requested}
  question     : "Chiến thắng ĐBP diễn ra năm nào?"
  options      : [A.1945, B.1954✓, C.1975, D.1986]
  answer_key   : "B"
  answer_text  : "1954"
  evidence     : {sentence, found_in_context}
  verification : null  ← Verifier sẽ điền

4 BẤT BIẾN (validator ép):
① Đúng 4 phương án, nhãn A/B/C/D, không trùng
② Đúng 1 đáp án đúng (is_correct=True)
③ answer_key khớp phương án is_correct
④ evidence.sentence ⊆ context (substring)
```

---

## PHẦN 4: VERIFIER (4 slide)

---

### Slide 16 — Tổng quan Verifier

**Tiêu đề:** Verifier — 8 Tiêu chí Kiểm định

**Bảng 8 tiêu chí (màu theo trọng số AHP):**

| # | Tiêu chí | Phương pháp | Trọng số |
|---|---|---|---|
| 1 | **Evidence Match** | NLI (GPT-4o-mini) | **17.8%** |
| 2 | **Single Correct** | NLI 4 phương án | **13.2%** |
| 3 | **Historical Correctness** | KB lookup | **12.6%** |
| 4 | **Distractor Type** | NER + cosine | 10.3% |
| 5 | Question Clarity | Heuristic | 8.6% |
| 6 | Bloom Fidelity | VNHSGE clf / GPT | 7.4% |
| 7 | Duplicate Check | Jaccard trigram | 4.5% |
| 8 | Answer Position | Phân bố A/B/C/D | 4.5% |

**Công thức:**
```
verifier_score = Σ weight_i × score_i ∈ [0,1]
accepted ≥ 0.75  |  needs_review [0.55, 0.75)  |  rejected < 0.55
```

---

### Slide 17 — Bloom Fidelity: VNHSGE vs GPT

**Tiêu đề:** Bloom Fidelity — Tích hợp VNHSGE vào Verifier

**Sơ đồ luồng:**
```
VNHSGE-History (2,000 câu)
      ↓ Gán nhãn Bloom (từ khoá)
      ↓ Train TF-IDF + LogReg
bloom_classifier.pkl
      ↓
Verifier: check_bloom_fidelity()
      ├─ bloom_clf available? → VNHSGE (offline, $0)
      └─ fallback             → GPT-4o-mini (online)
```

**Bảng so sánh (90 câu mẫu):**

| | VNHSGE Classifier | GPT Zero-shot |
|---|---|---|
| Accuracy vs requested | 64% | **76%** |
| Agreement nhau | 76% | — |
| Chi phí | **$0** | ~$0.0009/90 câu |

**Ghi chú:** CLF đánh giá nội dung thực tế (thong_hieu cho câu "Nguyên nhân"),
GPT trả về bloom_requested (nhan_biet) → GPT accuracy cao hơn theo metric,
nhưng CLF có thể đúng hơn về sư phạm.

---

### Slide 18 — Nhãn vàng Evidence Match

**Tiêu đề:** Nhãn vàng — Tận dụng is_impossible của ViQuAD

**Sơ đồ:**
```
ViQuAD 2.0
  ├─ answerable (10,114 câu)  → sinh câu hỏi bình thường
  └─ is_impossible (3,883 câu) → nhãn vàng Evidence Match
                                  (câu không có bằng chứng trong context
                                   → phải bị Verifier đánh trượt)
```

**Ý nghĩa:** Không cần gán tay 3,883 nhãn — ViQuAD đã làm sẵn.

**Kết quả thực tế:**
- Rule-Based Evidence Match = **0.356** (thấp — template không kiểm tra context)
- RAG+LLM Evidence Match = **0.951** (cao — prompt ép trả evidence nguyên văn)

---

### Slide 19 — Kết quả Verifier

**Tiêu đề:** Kết quả Verifier — 3 Phương pháp

**Radar chart (`eda_verifier_comparison.png`) chiếm 60% slide**

**Bảng số liệu bên cạnh:**

| Tiêu chí | Rule | RAG | ViT5 |
|---|---|---|---|
| VQR | 27.9% | **62.2%** | 45.1% |
| Evidence Match | 0.356 | **0.951** | 0.799 |
| Single Correct | 0.278 | **0.594** | 0.374 |
| Distractor Type | **0.920** | 0.740 | 0.740 |
| Historical | **0.977** | 0.838 | 0.708 |
| Avg score | 0.614 | **0.803** | 0.717 |

---

## PHẦN 5: DSS — AHP/TOPSIS + ILP (5 slide)

---

### Slide 20 — AHP: Xây dựng trọng số

**Tiêu đề:** AHP — Xác định Trọng số Tiêu chí

**Bố cục 2 cột:**

*Cột trái — Trọng số (bar chart ngang):*
```
VQR              ████████████  25.7%
Evidence Match   ████████      17.8%
Single Correct   ██████        13.2%
Historical       ██████        12.6%
Distractor Type  █████         10.3%
Question Clarity ████           8.6%
Bloom Fidelity   ███            7.4%
Duplicate Check  ██             4.5%
```

*Cột phải — Thông số AHP:*
```
Ma trận so sánh cặp: 8×8
Thang Saaty: 1–9

λ_max   = 8.2270
CI      = 0.0324
RI(n=8) = 1.41
CR = CI/RI = 0.0230

CR = 0.023 < 0.1 ✅
→ Ma trận NHẤT QUÁN
```

---

### Slide 21 — TOPSIS: Xếp hạng phương án

**Tiêu đề:** TOPSIS — Xếp hạng 3 Phương pháp

**Sơ đồ các bước TOPSIS (flow nhỏ ở góc):**
```
Ma trận quyết định (3×8)
→ Chuẩn hoá vector
→ Ma trận có trọng số (AHP)
→ PIS+ và NIS-
→ d_PIS, d_NIS (khoảng cách Euclidean)
→ score = d_NIS / (d_PIS + d_NIS)
```

**Bảng kết quả (font lớn, màu nổi):**

| Hạng | Phương pháp | Score | d_PIS | d_NIS |
|---|---|---|---|---|
| 🥇 **1** | **RAG+LLM** | **0.8894** | 0.0185 | 0.1490 |
| 🥈 **2** | **ViT5-FT** | **0.5417** | 0.0748 | 0.0884 |
| 🥉 **3** | **Rule-Based** | **0.1584** | 0.1492 | 0.0281 |

**Điểm mạnh tương đối:**
- Rule-Based: Distractor Type, Historical, Bloom Fidelity
- RAG+LLM: VQR, Evidence Match, Single Correct, Question Clarity
- ViT5: Duplicate Check

---

### Slide 22 — Phân tích độ nhạy

**Tiêu đề:** Phân tích Độ nhạy — Thứ hạng có ổn định không?

**Biểu đồ đường (sensitivity.csv):**
- Trục X: Trọng số VQR (1% → 50%)
- Trục Y: TOPSIS score
- 3 đường: RAG+LLM (xanh), ViT5 (cam), Rule-based (xanh dương)
- RAG+LLM luôn trên cùng, không có điểm giao

**Box kết luận (màu xanh lá):**
```
✅ THỨ HẠNG ỔN ĐỊNH TUYỆT ĐỐI
Không có điểm giao khi VQR thay đổi 1%→50%
RAG+LLM > ViT5 > Rule-based
trong mọi kịch bản trọng số
```

**Ý nghĩa DSS:** Kết luận bền vững — không phụ thuộc vào quan điểm trọng số
của người dùng cụ thể.

---

### Slide 23 — ILP Ráp đề thi

**Tiêu đề:** ILP — Ráp đề thi tối ưu có ràng buộc

**Bài toán tối ưu:**
```
max  Σ verifier_score(i) · x(i)

Ràng buộc:
  Σ x(i) = K                      (đúng K câu)
  phân bố Bloom ≈ đặc tả ±1       (Bloom constraint)
  mỗi title ≥ 1 câu               (phủ đủ chủ đề)
  mỗi context_id ≤ 2 câu          (không lặp ngữ cảnh)
  x(i) ∈ {0, 1}                   (biến nhị phân)

Solver: PuLP/CBC (timeLimit=30s)
```

**Bảng so sánh 3 chiến lược:**

| Chiến lược | Tổng score | Tuân thủ Bloom | Thời gian |
|---|---|---|---|
| **ILP (tối ưu)** | **Cao nhất** | ✅ ±1 câu | ~30s |
| Greedy | ~95% ILP | ⚠️ xấp xỉ | <1ms |
| Random | Thấp nhất | ❌ ngẫu nhiên | <1ms |

---

### Slide 24 — Kết luận DSS tổng hợp

**Tiêu đề:** Kết luận DSS — Chiến lược nào cho tình huống nào?

**Ma trận quyết định (bảng lớn, màu sắc):**

| Tình huống | Khuyến nghị | Lý do |
|---|---|---|
| Câu Thời gian/Nhân vật/Địa điểm, offline, $0 | **Rule-Based** | Historical=0.977, $0, 8ms |
| Câu Nguyên nhân/Ý nghĩa, ưu tiên chất lượng | **RAG+LLM** | VQR=62%, Evidence=0.951 |
| Số lượng lớn, offline, chấp nhận distractor TB | **ViT5-QG + RAG-DG** | Hybrid tối ưu |
| Tổng thể, không ràng buộc | **RAG+LLM** | TOPSIS score=0.889 |

**Box kết luận cuối (font lớn, màu đỏ nổi):**
> *"Không có phương pháp nào thắng trên mọi tiêu chí.*  
> *Hybrid routing theo wh_type là chiến lược tối ưu."*

---

## PHẦN 6: KẾT QUẢ TỔNG HỢP (3 slide)

---

### Slide 25 — Bảng so sánh toàn diện

**Tiêu đề:** Bảng So sánh Toàn diện (số liệu thật)

**Bảng lớn chiếm toàn bộ slide:**

| | Rule-Based | RAG+LLM | ViT5-FT |
|---|---|---|---|
| **VQR** | 27.9% | **62.2%** | 45.1% |
| Evidence Match | 0.356 | **0.951** | 0.799 |
| Single Correct | 0.278 | **0.594** | 0.374 |
| Distractor Type | **0.920** | 0.740 | 0.740 |
| Historical | **0.977** | 0.838 | 0.708 |
| Bloom Fidelity | **0.926** | 0.862 | 0.751 |
| Duplicate Check | 0.080 | 0.650 | **0.733** |
| **Avg score** | 0.614 | **0.803** | 0.717 |
| **TOPSIS** | 0.1584 | **0.8894** | 0.5417 |
| Latency | **8ms** | 5,114ms | ~200ms |
| Chi phí/câu | **$0** | $0.00019 | **$0** |

---

### Slide 26 — Kỳ vọng vs Thực tế

**Tiêu đề:** Phân tích Kỳ vọng vs Thực tế

**3 block màu (mỗi phương pháp 1 block):**

🔧 **Rule-Based**
- Kỳ vọng: VQR~60%  →  Thực tế: **27.9%**
- Lý do: Template cứng → Duplicate=0.080
- Bài học: Cần đa dạng hoá template động

🤖 **RAG+LLM**
- Kỳ vọng: VQR~80%  →  Thực tế: **62.2%**
- Evidence=0.951 vượt kỳ vọng ✅
- Bài học: Prompt ép evidence nguyên văn rất hiệu quả

🧠 **ViT5-FT**
- Kỳ vọng: VQR~70%  →  Thực tế: **45.1%**
- Single Correct=0.374 thấp do distractor kém
- Bài học: Cần dữ liệu DG chất lượng cao (do người viết)

---

### Slide 27 — Đóng góp kỹ thuật

**Tiêu đề:** Đóng góp Kỹ thuật Nổi bật

**5 điểm (icon + mô tả ngắn):**

🔬 **NER LOC: 41% → 97%**
Phát hiện underthesea không đủ tốt → chuyển ELECTRA + 7 luật lọc

🛡️ **Nhãn vàng Evidence Match**
Tận dụng 3,883 câu `is_impossible` sẵn có → không cần gán tay

📚 **VNHSGE tích hợp thực sự**
Train Bloom classifier (acc=90%) từ 2,000 câu VNHSGE-History

⚡ **Checkpoint Resume**
Train ViT5 bị ngắt giữa chừng → resume tự động từ epoch cuối

🎯 **Hybrid Routing DSS**
Kết luận định lượng: không có phương pháp nào thắng toàn diện

---

## PHẦN 7: DEMO + KẾT LUẬN (4 slide)

---

### Slide 28 — Demo Dashboard (slide placeholder)

**Tiêu đề:** Demo — Dashboard Streamlit

**Nội dung slide:**

```
[SCREENSHOT Dashboard — Trang Demo Sinh Câu Hỏi]

Tính năng demo trực tiếp:
  1. Nhập ngữ cảnh lịch sử bất kỳ
  2. Chọn Wh-type + Bloom level
  3. Bấm "Sinh câu hỏi"
  4. Xem 3 phương pháp sinh song song
  5. So sánh chất lượng trực quan
```

*→ Chuyển sang demo live trên máy*

---

### Slide 29 — Demo AHP/TOPSIS realtime (slide placeholder)

**Tiêu đề:** Demo — AHP/TOPSIS Realtime

**Nội dung slide:**

```
[SCREENSHOT Dashboard — Trang AHP/TOPSIS]

Demo: Kéo slider trọng số
  - Tăng trọng số "Chi phí" → Rule-Based lên hạng
  - Tăng trọng số "VQR" → RAG+LLM càng vượt trội
  - Radar chart cập nhật realtime

→ Minh hoạ tại sao cần DSS thay vì
  quyết định bằng cảm tính
```

---

### Slide 30 — Kết luận

**Tiêu đề:** Kết luận

**3 cột:**

*Đã đạt được:*
- ✅ DSS 3 tầng hoàn chỉnh
- ✅ 3 phương pháp sinh câu hỏi
- ✅ Verifier 8 tiêu chí với nhãn vàng
- ✅ AHP/TOPSIS (CR=0.023)
- ✅ ILP ráp đề thi
- ✅ Dashboard Streamlit
- ✅ 2 nguồn dữ liệu tích hợp thực sự

*Hạn chế:*
- ⚠️ ViT5 DG distractor kém
- ⚠️ van_dung_cao F1=0%
- ⚠️ Chưa có human eval

*Hướng phát triển:*
- 🔮 Hybrid routing tự động
- 🔮 Augmentation dữ liệu DG
- 🔮 Mở rộng sang môn khác
- 🔮 Human eval với Cohen's kappa

---

### Slide 31 — Trang kết / Hỏi đáp

```
[LOGO TRƯỜNG]

CẢM ƠN QUÝ THẦY CÔ VÀ CÁC BẠN
ĐÃ LẮNG NGHE

━━━━━━━━━━━━━━━━━━━━━━━━━━

HỎI VÀ ĐÁP

━━━━━━━━━━━━━━━━━━━━━━━━━━

Source code: github.com/[repo]
Báo cáo chi tiết: BAO_CAO.md
```

---

## GHI CHÚ CHO NGƯỜI TRÌNH BÀY

### Phân bổ thời gian (tổng 20 phút)

| Phần | Slide | Thời gian |
|---|---|---|
| Mở đầu | 1–4 | 2 phút |
| Dữ liệu | 5–9 | 3 phút |
| 3 Phương pháp | 10–15 | 4 phút |
| Verifier | 16–19 | 3 phút |
| DSS (AHP/TOPSIS/ILP) | 20–24 | 3 phút |
| Kết quả tổng hợp | 25–27 | 2 phút |
| Demo + Kết luận | 28–31 | 3 phút |

### Slide quan trọng nhất (dành nhiều thời gian)

1. **Slide 3** — Kiến trúc 3 tầng DSS: hội đồng cần hiểu ngay đây là DSS, không phải NLP đơn thuần
2. **Slide 21** — TOPSIS kết quả: số liệu chính thức, giải thích rõ d_PIS và d_NIS
3. **Slide 22** — Độ nhạy: đây là điểm phân biệt DSS thật sự vs chỉ so sánh kỹ thuật
4. **Slide 28–29** — Demo live: ấn tượng nhất với hội đồng

### Câu hỏi hội đồng có thể hỏi

| Câu hỏi | Trả lời gợi ý |
|---|---|
| "Tại sao CR < 0.1 thì mới hợp lệ?" | CR là tỉ số nhất quán, < 0.1 nghĩa là ma trận so sánh cặp không mâu thuẫn quá mức (Saaty 1980) |
| "Tại sao không dùng FAHP hay BWM?" | AHP đủ cho 3 phương án + 8 tiêu chí, ít tham số hơn |
| "Evidence Match 0.951 của RAG có chắc không?" | Dùng NLI (GPT-4o-mini), có nhãn vàng `is_impossible` làm ground truth |
| "ViT5 DG kém — có cải thiện được không?" | Có: (1) dữ liệu DG do người viết, (2) ViT5-QG + RAG-DG hybrid |
| "VNHSGE classifier F1=0% ở van_dung_cao?" | Chỉ có 11 mẫu train — mất cân bằng nghiêm trọng, cần thêm dữ liệu |

---

*Đề cương này tương ứng với file BAO_CAO.md — mọi số liệu đều từ kết quả chạy thật.*
