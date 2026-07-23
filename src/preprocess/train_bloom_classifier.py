"""
Bloom Classifier từ VNHSGE — Nguồn dữ liệu thứ 2.

Bước 1: Tải toàn bộ file JSON lịch sử từ GitHub VNHSGE
Bước 2: Parse câu hỏi + gán nhãn Bloom bằng heuristic từ khoá
Bước 3: Train TF-IDF + Logistic Regression classifier (offline, ~5 phút)
Bước 4: Lưu model để Verifier dùng (thay GPT zero-shot cho Bloom Fidelity)

Cài:
    pip install scikit-learn joblib

Chạy:
    python src/preprocess/train_bloom_classifier.py --outdir models/bloom
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# ─────────────────────────── Danh sách file VNHSGE-History ──────────────────

BASE_URL = ("https://raw.githubusercontent.com/Xdao85/VNHSGE/main/"
            "Dataset/VNHSGE-V/JSON%20format/")

HISTORY_FILES = {
    "eval": [
        "eval/History/MET_His_IE_2019.json",
        "eval/History/MET_His_IE_2020.json",
        "eval/History/MET_His_IE_2021.json",
        "eval/History/MET_His_IE_2022.json",
        "eval/History/MET_His_IE_2023.json",
    ],
    "test": [
        "test/History/His_31.json", "test/History/His_32.json",
        "test/History/His_33.json", "test/History/His_34.json",
        "test/History/His_41.json",
    ],
    "train": [f"train/History/His_{i}.json" for i in range(1, 41)
              if i not in (31, 32, 33, 34)] + [
        "train/History/MET_His_OE_2022_301.json",
        "train/History/MET_His_OE_2022_302.json",
        "train/History/MET_His_OE_2022_303.json",
        "train/History/MET_His_OE_2022_304.json",
    ],
}

# ─────────────────────────── Tải dữ liệu ────────────────────────────────────

def download_vnhsge(cache_dir: str = "data/vnhsge") -> list[dict]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    all_items = []
    total_files = sum(len(v) for v in HISTORY_FILES.values())
    done = 0

    for split, files in HISTORY_FILES.items():
        for rel_path in files:
            done += 1
            url       = BASE_URL + rel_path.replace(" ", "%20")
            filename  = Path(rel_path).name
            local     = Path(cache_dir) / filename

            if not local.exists():
                try:
                    urllib.request.urlretrieve(url, str(local))
                except Exception as e:
                    print(f"  [{done}/{total_files}] SKIP {filename}: {e}")
                    continue

            try:
                data = json.load(open(local, encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        item["_split"] = split
                    all_items.extend(data)
                    print(f"  [{done}/{total_files}] {filename}: {len(data)} câu")
                else:
                    print(f"  [{done}/{total_files}] {filename}: format lạ, bỏ qua")
            except Exception as e:
                print(f"  [{done}/{total_files}] {filename}: lỗi đọc {e}")

    print(f"\nTổng: {len(all_items)} câu lịch sử VNHSGE")
    return all_items


# ─────────────────────────── Gán nhãn Bloom (heuristic) ─────────────────────
#
# VNHSGE không có cột Bloom trong JSON. Ta gán bằng từ khoá trong câu hỏi.
# Dựa trên đặc điểm đề thi THPT: phần lớn Nhận biết/Thông hiểu (mức 1-2),
# ít Vận dụng (mức 3-4).
#
# Tham chiếu: paper VNHSGE mô tả 4 mức: knowledge, comprehension,
# application, high application.

BLOOM_KEYWORDS = {
    "nhan_biet": [
        "là gì", "là ai", "gọi là", "có tên là", "được gọi", "thuộc",
        "năm nào", "khi nào", "ở đâu", "ai đã", "ai là", "sự kiện nào",
        "thành lập", "ra đời", "diễn ra", "xảy ra", "được ký", "được thành lập",
        "quốc gia nào", "tổ chức nào", "nước nào", "thời gian nào",
        "đứng thứ", "bao nhiêu", "giai đoạn nào", "thế kỷ nào",
    ],
    "thong_hieu": [
        "vì sao", "tại sao", "do đâu", "nguyên nhân", "mục đích",
        "ý nghĩa", "tác động", "hệ quả", "dẫn đến", "ảnh hưởng",
        "giải thích", "chứng tỏ", "thể hiện", "phản ánh", "biểu hiện",
        "nội dung chính", "điểm khác biệt", "điểm giống nhau",
        "vai trò", "đặc điểm", "tính chất",
    ],
    "van_dung": [
        "nhận xét", "đánh giá", "so sánh", "liên hệ", "rút ra",
        "bài học", "kinh nghiệm", "phân tích", "chứng minh",
        "giả sử", "nếu như", "trong bối cảnh", "hiện nay",
        "vận dụng", "áp dụng", "thực tiễn",
    ],
    "van_dung_cao": [
        "bình luận", "phê phán", "đề xuất", "giải pháp", "kiến nghị",
        "sáng tạo", "thiết kế", "xây dựng phương án",
        "tổng hợp", "hệ thống hoá",
    ],
}


def assign_bloom(question_text: str) -> str:
    """Gán nhãn Bloom dựa trên từ khoá trong câu hỏi.
    Ưu tiên: van_dung_cao > van_dung > thong_hieu > nhan_biet.
    """
    q = question_text.lower()
    for level in ["van_dung_cao", "van_dung", "thong_hieu"]:
        if any(kw in q for kw in BLOOM_KEYWORDS[level]):
            return level
    return "nhan_biet"   # mặc định cho câu factoid đơn giản


# ─────────────────────────── Parse câu hỏi ───────────────────────────────────

def extract_stem(question_text: str) -> str:
    """Tách phần thân câu hỏi (trước các lựa chọn A/B/C/D).
    VNHSGE có format: 'Câu N: <stem>\nA. ...\nB. ...'
    """
    # bỏ "Câu N:" ở đầu
    q = re.sub(r"^Câu\s+\d+[:.]\s*", "", question_text.strip())
    # cắt trước lựa chọn đầu tiên (A. hoặc A:)
    q = re.split(r"\n[A-D][.:\s]", q)[0]
    return q.strip()


def build_dataset(items: list[dict]) -> list[dict]:
    """Chuyển raw JSON sang [{text, bloom, split}]."""
    rows = []
    for item in items:
        raw_q  = str(item.get("Question", ""))
        stem   = extract_stem(raw_q)
        bloom  = assign_bloom(stem)
        rows.append({
            "id":    item.get("ID", ""),
            "text":  stem,
            "bloom": bloom,
            "split": item.get("_split", "train"),
        })
    return rows


# ─────────────────────────── Train TF-IDF + Logistic Regression ──────────────

def train_classifier(rows: list[dict], outdir: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    import joblib

    train = [r for r in rows if r["split"] == "train"]
    eval_ = [r for r in rows if r["split"] == "eval"]

    X_train = [r["text"] for r in train]
    y_train = [r["bloom"] for r in train]
    X_eval  = [r["text"] for r in eval_]
    y_eval  = [r["bloom"] for r in eval_]

    print(f"\nTrain: {len(train)} | Eval: {len(eval_)}")
    from collections import Counter
    print("Phân bố Bloom (train):", dict(Counter(y_train)))
    print("Phân bố Bloom (eval):", dict(Counter(y_eval)))

    # Pipeline TF-IDF + Logistic Regression
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",    # char n-gram — tốt cho tiếng Việt
            ngram_range=(2, 4),
            max_features=20000,
            sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",   # bù lệch phân bố Bloom
            random_state=42,
        )),
    ])

    clf.fit(X_train, y_train)

    # Đánh giá
    if X_eval:
        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        print(f"\nEval accuracy: {acc:.3f}")
        print(classification_report(y_eval, y_pred,
              target_names=["nhan_biet","thong_hieu","van_dung","van_dung_cao"],
              zero_division=0))
    else:
        acc = 0.0

    # Lưu model
    Path(outdir).mkdir(parents=True, exist_ok=True)
    model_path = Path(outdir) / "bloom_classifier.pkl"
    joblib.dump(clf, str(model_path))
    print(f"\nModel saved -> {model_path}")

    # Lưu metadata
    meta = {
        "accuracy": round(acc, 4),
        "n_train": len(train),
        "n_eval":  len(eval_),
        "labels":  ["nhan_biet","thong_hieu","van_dung","van_dung_cao"],
        "feature": "TF-IDF char 2-4gram",
        "model":   "LogisticRegression C=1.0 balanced",
        "source":  "VNHSGE-History",
    }
    import json
    json.dump(meta, open(Path(outdir)/"bloom_meta.json","w"), ensure_ascii=False, indent=2)
    return meta


# ─────────────────────────── Predict function (dùng trong Verifier) ──────────

def load_bloom_classifier(model_dir: str = "models/bloom"):
    """Load classifier đã train. Dùng trong Verifier thay GPT zero-shot."""
    import joblib
    path = Path(model_dir) / "bloom_classifier.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Chưa train: {path}. Chạy train_bloom_classifier.py trước.")
    return joblib.load(str(path))


def predict_bloom(clf, question: str) -> str:
    """Dự đoán mức Bloom cho một câu hỏi."""
    stem = extract_stem(question)
    return clf.predict([stem])[0]


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/vnhsge")
    ap.add_argument("--outdir",    default="models/bloom")
    ap.add_argument("--skip_download", action="store_true",
                    help="Bỏ qua tải về nếu đã có trong cache_dir")
    args = ap.parse_args()

    # Bước 1: Tải dữ liệu
    if not args.skip_download:
        print("Đang tải VNHSGE-History từ GitHub...")
        items = download_vnhsge(args.cache_dir)
    else:
        import json
        items = []
        for f in Path(args.cache_dir).glob("*.json"):
            try:
                data = json.load(open(f, encoding="utf-8"))
                if isinstance(data, list):
                    items.extend(data)
            except Exception:
                pass
        print(f"Loaded {len(items)} câu từ cache")

    if not items:
        print("Không có dữ liệu. Kiểm tra kết nối mạng.")
        return

    # Bước 2: Gán nhãn Bloom + parse
    print("\nGán nhãn Bloom bằng heuristic từ khoá...")
    rows = build_dataset(items)

    # Xem phân bố nhãn
    from collections import Counter
    bloom_dist = Counter(r["bloom"] for r in rows)
    print("Phân bố Bloom toàn bộ:")
    for k, v in sorted(bloom_dist.items()):
        print(f"  {k:15s}: {v:4d} ({v/len(rows)*100:.1f}%)")

    # In 5 mẫu đại diện mỗi mức
    print("\nVí dụ mẫu mỗi mức Bloom:")
    shown = set()
    for r in rows:
        if r["bloom"] not in shown:
            print(f"\n  [{r['bloom']}]")
            print(f"  {r['text'][:100]}...")
            shown.add(r["bloom"])
        if len(shown) == 4:
            break

    # Bước 3: Train classifier
    print("\nTrain TF-IDF + Logistic Regression...")
    meta = train_classifier(rows, args.outdir)

    print("\n=== HOÀN TẤT ===")
    print(f"  Nguồn dữ liệu: VNHSGE-History (2019–2023)")
    print(f"  Số câu        : {len(rows)}")
    print(f"  Eval accuracy : {meta['accuracy']:.3f}")
    print(f"  Model         : {args.outdir}/bloom_classifier.pkl")
    print()
    print("Để dùng trong Verifier:")
    print("  from train_bloom_classifier import load_bloom_classifier, predict_bloom")
    print("  clf = load_bloom_classifier('models/bloom')")
    print("  level = predict_bloom(clf, 'Chiến thắng Điện Biên Phủ diễn ra năm nào?')")


if __name__ == "__main__":
    main()
