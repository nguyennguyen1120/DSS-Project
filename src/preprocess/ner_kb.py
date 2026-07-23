"""
Bước 2 — Trích thực thể (NER) + xây Knowledge Base + Distractor pool.

Nền cho:
  - Rule-based: chọn đáp án theo wh_type, sinh distractor cùng loại/cùng miền.
  - Verifier:   Historical Correctness (đối chiếu KB), Distractor Type Match.

Kiến trúc (bù điểm yếu của underthesea):
  - REGEX mạnh cho YEAR/DATE/CENTURY  (NER tiếng Việt bắt thời gian rất kém).
  - underthesea.ner cho PER/LOC/ORG   (chạy theo TỪNG CÂU, gộp span B-/I-).
  - Hậu xử lý: map offset ký tự, bỏ chức danh, khử alias cơ bản.

Đầu vào : data/processed/corpus.parquet  (từ build_corpus.py)
Đầu ra  :
  - entities.parquet       — mọi thực thể + type + offset + chunk_id + is_vietnam
  - distractor_pool.pkl    — {(type, bucket) -> set(surface)}  cho rule-based
  - ner_quality_sample.csv — 100 thực thể ngẫu nhiên để CHẤM TAY precision

Chạy:
    pip install underthesea pandas pyarrow
    python src/preprocess/ner_kb.py --corpus data/processed/corpus.parquet \
        --outdir data/processed
"""

from __future__ import annotations

import argparse
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

# ----------------------------- Regex thời gian ------------------------------
# Đặt trước NER và độc lập với nó. Ưu tiên mẫu cụ thể trước mẫu chung.

DATE_FULL = re.compile(r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{3,4})\b")
DATE_NUM  = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{3,4})\b")
YEAR_NAM  = re.compile(r"\bnăm\s+(\d{1,4})\b")
YEAR_BARE = re.compile(r"\b(9[0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b")
CENTURY   = re.compile(r"\bthế\s+k[iỉyỷ]\s+([IVXLC]+|\d{1,2})\b", re.IGNORECASE)
# "trước Công nguyên" / "TCN" -> đánh dấu năm âm
BC_MARK   = re.compile(r"\btr(?:ước)?\.?\s*C(?:ông)?\.?\s*[Nn](?:guyên)?\b|\bTCN\b")

# chức danh cần tách khỏi tên người
TITLES = ["Chủ tịch", "Tổng thống", "Đại tướng", "Tướng", "Vua", "Hoàng đế",
          "Thủ tướng", "Tổng bí thư", "Giáo hoàng", "Nữ hoàng", "Quốc vương",
          "Hoàng hậu", "Thái tử", "Công tước", "Bá tước", "Thủ lĩnh", "Ông", "Bà"]
TITLE_RE = re.compile(r"^(?:" + "|".join(TITLES) + r")\s+")

# danh sách đen: bị NER nhận nhầm là PER nhưng không phải
PER_BLACKLIST = {"Người", "Bác", "Đảng", "Nhà nước", "Chính phủ", "Ông", "Bà"}


# ----------------------------- Tiện ích -------------------------------------

def normalize_vi(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def year_bucket(y: int) -> str:
    """Nhóm năm thành 'giai đoạn' để lấy distractor cùng thời. Đa miền nên
    dùng thế kỷ làm bucket, không dùng bảng giai đoạn VN cứng."""
    if y < 0:
        return "BC"
    return f"{(y // 100) + 1}C"     # 1954 -> 20C, 938 -> 10C


# ----------------------------- Trích thời gian (regex) ----------------------

def extract_temporal(text: str) -> list[dict]:
    ents, spans = [], []

    def add(m, typ, norm):
        s, e = m.span()
        # tránh chồng lấn với match đã có
        if any(not (e <= a or s >= b) for a, b in spans):
            return
        spans.append((s, e))
        ents.append({"surface": text[s:e], "type": typ,
                     "normalized": norm, "char_start": s, "char_end": e})

    for m in DATE_FULL.finditer(text):
        d, mo, y = m.groups()
        add(m, "DATE", f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
    for m in DATE_NUM.finditer(text):
        d, mo, y = m.groups()
        add(m, "DATE", f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
    for m in YEAR_NAM.finditer(text):
        y = int(m.group(1))
        # kiểm tra hậu tố TCN ngay sau
        tail = text[m.end():m.end() + 20]
        yy = -y if BC_MARK.search(tail) else y
        add(m, "YEAR", str(yy))
    for m in CENTURY.finditer(text):
        add(m, "CENTURY", m.group(1).upper())
    # YEAR_BARE cuối cùng (dễ trùng nhất) — chỉ thêm nếu chưa bị match
    for m in YEAR_BARE.finditer(text):
        y = int(m.group(1))
        tail = text[m.end():m.end() + 20]
        yy = -y if BC_MARK.search(tail) else y
        add(m, "YEAR", str(yy))

    return ents


# ----------------------------- Trích PER/LOC/ORG (underthesea) --------------

def _merge_iob(tagged: list[tuple]) -> list[tuple[str, str]]:
    """Gộp chuỗi B-/I- thành (surface, TYPE). tagged: [(word,pos,chunk,ner)]."""
    out, cur_words, cur_type = [], [], None
    for tok in tagged:
        word, ner = tok[0], tok[-1]
        if ner.startswith("B-"):
            if cur_type:
                out.append((" ".join(cur_words), cur_type))
            cur_words, cur_type = [word], ner[2:]
        elif ner.startswith("I-") and cur_type == ner[2:]:
            cur_words.append(word)
        else:
            if cur_type:
                out.append((" ".join(cur_words), cur_type))
            cur_words, cur_type = [], None
    if cur_type:
        out.append((" ".join(cur_words), cur_type))
    return out


# ----------------------------- Backend ELECTRA-NER (khuyến nghị) ------------

# VLSP2018 nhãn có thể là PER/PERSON, LOC/LOCATION, ORG/ORGANIZATION -> chuẩn hoá
_TYPE_MAP = {
    "PER": "PER", "PERSON": "PER", "B-PER": "PER", "I-PER": "PER",
    "LOC": "LOC", "LOCATION": "LOC", "B-LOC": "LOC", "I-LOC": "LOC",
    "ORG": "ORG", "ORGANIZATION": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
}


def make_electra_ner():
    """Tạo pipeline NER từ NlpHUST/ner-vietnamese-electra-base.
    aggregation_strategy='simple' -> tự gộp span, trả start/end offset."""
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)
    name = "NlpHUST/ner-vietnamese-electra-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForTokenClassification.from_pretrained(name)
    return pipeline("ner", model=mdl, tokenizer=tok,
                    aggregation_strategy="simple")


def extract_named_electra(text: str, ner_pipe) -> list[dict]:
    """PER/LOC/ORG bằng ELECTRA pipeline. Đã có sẵn offset -> khỏi tự dò."""
    ents = []
    try:
        results = ner_pipe(text)
    except Exception:
        return ents
    for r in results:
        raw_type = str(r.get("entity_group") or r.get("entity") or "")
        typ = _TYPE_MAP.get(raw_type.upper())
        if typ is None:
            continue
        surface = str(r.get("word", "")).strip()
        cs = r.get("start")
        ce = r.get("end")
        # offset từ pipeline map về text gốc; nếu thiếu thì dò chuỗi
        if cs is None or ce is None:
            idx = text.find(surface)
            cs, ce = (idx, idx + len(surface)) if idx >= 0 else (None, None)
        if not surface:
            continue
        ents.append({"surface": surface, "type": typ,
                     "normalized": surface.lower().replace(" ", "_"),
                     "char_start": cs, "char_end": ce})
    return ents


# ----------------------------- Trích PER/LOC/ORG (underthesea) --------------


def extract_named(text: str, ner_fn, sent_fn) -> list[dict]:
    """PER/LOC/ORG bằng underthesea, map offset về vị trí trong `text`."""
    ents = []
    cursor = 0
    for sent in sent_fn(text):
        # tìm vị trí câu trong text để cộng offset
        idx = text.find(sent, cursor)
        base = idx if idx >= 0 else cursor
        cursor = base + len(sent)
        try:
            tagged = ner_fn(sent)
        except Exception:
            continue
        for surface, typ in _merge_iob(tagged):
            if typ not in ("PER", "LOC", "ORG"):
                continue
            surface = surface.strip()
            # bỏ chức danh khỏi tên người
            if typ == "PER":
                surface = TITLE_RE.sub("", surface).strip()
            if not surface or surface in PER_BLACKLIST:
                continue
            # tìm offset trong câu -> cộng base
            rel = sent.find(surface)
            cs = base + rel if rel >= 0 else base
            ents.append({"surface": surface, "type": typ,
                         "normalized": surface.lower().replace(" ", "_"),
                         "char_start": cs, "char_end": cs + len(surface)})
    return ents


# ----------------------------- Pipeline chính -------------------------------

def process_corpus(df, named_fn) -> tuple[list[dict], dict]:
    """named_fn(text) -> list thực thể PER/LOC/ORG (đã có offset)."""
    from clean_entities import clean_entities

    all_ents = []
    for _, row in df.iterrows():
        text = normalize_vi(str(row["context"]))
        cid = row["context_id"]
        is_vn = bool(row.get("is_vietnam", False))

        raw = extract_temporal(text) + named_fn(text)
        ents = clean_entities(raw)          # <-- LỚP LỌC LUẬT

        for e in ents:
            # bucket: theo năm nếu là YEAR/DATE; theo miền nếu là PER/LOC/ORG
            if e["type"] in ("YEAR", "DATE"):
                try:
                    y = int(e["normalized"][:5].split("-")[0])
                    bucket = year_bucket(y)
                except ValueError:
                    bucket = "unk"
            else:
                bucket = "VN" if is_vn else "TG"
            e.update({"context_id": cid, "is_vietnam": is_vn, "bucket": bucket})
            all_ents.append(e)

    # distractor pool: {(type, bucket) -> set(surface)}
    pool: dict[tuple, set] = defaultdict(set)
    for e in all_ents:
        pool[(e["type"], e["bucket"])].add(e["surface"])

    return all_ents, {k: v for k, v in pool.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.parquet")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--backend", choices=["electra", "underthesea"],
                    default="electra", help="electra = NlpHUST (khuyến nghị)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.corpus)
    if args.limit:
        df = df.head(args.limit)

    # chọn backend NER cho PER/LOC/ORG
    if args.backend == "electra":
        pipe = make_electra_ner()
        named_fn = lambda text: extract_named_electra(text, pipe)
    else:
        from underthesea import ner as ner_fn
        from underthesea import sent_tokenize as sent_fn
        named_fn = lambda text: extract_named(text, ner_fn, sent_fn)

    print(f"Đang NER {len(df)} đoạn (backend={args.backend})... "
          "(có thể mất vài phút)")
    ents, pool = process_corpus(df, named_fn)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    edf = pd.DataFrame(ents)
    edf.to_parquet(outdir / "entities.parquet", index=False)
    with open(outdir / "distractor_pool.pkl", "wb") as f:
        pickle.dump(pool, f)

    # mẫu chấm tay precision (100 thực thể ngẫu nhiên)
    sample = edf.sample(n=min(100, len(edf)), random_state=42)[
        ["surface", "type", "normalized", "bucket", "context_id"]]
    sample = sample.assign(correct="")   # cột điền tay: 1=đúng, 0=sai
    sample.to_csv(outdir / "ner_quality_sample.csv", index=False,
                  encoding="utf-8-sig")

    # báo cáo
    print("\n=== THỐNG KÊ NER ===")
    print(f"  Tổng thực thể      : {len(edf):,}")
    print("  Theo type:")
    for typ, n in edf.type.value_counts().items():
        print(f"    {typ:8s}: {n:,}")
    print(f"  Distractor pool    : {len(pool)} nhóm (type, bucket)")
    print("\n  Top nhóm pool (đủ lớn để sinh distractor):")
    big = sorted(pool.items(), key=lambda kv: -len(kv[1]))[:10]
    for (typ, bucket), s in big:
        print(f"    ({typ:7s}, {bucket:4s}): {len(s):3d} thực thể")
    print(f"\nĐã lưu: entities.parquet ({len(edf)}), distractor_pool.pkl, "
          f"ner_quality_sample.csv")
    print("\n[VIỆC TIẾP] Mở ner_quality_sample.csv, chấm cột `correct` (1/0) "
          "trên 100 dòng -> báo cáo precision. Ngưỡng: YEAR>=0.95, PER/LOC>=0.85.")


if __name__ == "__main__":
    main()
