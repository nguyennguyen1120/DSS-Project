"""
Bước RAG-0: Build FAISS index + BM25 index từ corpus.parquet.
Chạy MỘT LẦN trước khi dùng rag_llm.py.

CHỈ index đoạn train+dev — TUYỆT ĐỐI không index test (chống rò rỉ).

Đầu ra:
  data/index/faiss.index     — FAISS flat L2
  data/index/bm25.pkl        — BM25Okapi
  data/index/index_meta.parquet — context_id, title, context, split, is_vietnam

Cài:
    pip install faiss-cpu sentence-transformers rank-bm25
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "preprocess"))

EMBED_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
BATCH_SIZE  = 64


def build_bm25(texts: list[str]):
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


def build_faiss(texts: list[str], model):
    import numpy as np
    import faiss

    print(f"  Encoding {len(texts)} đoạn bằng {EMBED_MODEL}...")
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        normalize_embeddings=True,          # cosine sim = dot product sau normalize
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner Product = cosine khi đã normalize
    index.add(embeddings)
    return index, embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  default="data/processed/corpus.parquet")
    ap.add_argument("--outdir",  default="data/index")
    ap.add_argument("--splits",  default="train,dev",
                    help="Chỉ index các split này (KHÔNG index test)")
    args = ap.parse_args()

    import pandas as pd
    import faiss
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(args.corpus)
    allowed = {s.strip() for s in args.splits.split(",")}
    df = df[df["split"].isin(allowed)].reset_index(drop=True)
    print(f"Index {len(df)} đoạn (split={allowed}) — test KHÔNG được index")

    texts = df["context"].tolist()

    # --- BM25 ---
    print("Building BM25...")
    bm25 = build_bm25(texts)

    # --- FAISS ---
    print(f"Loading embedding model {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    faiss_index, _ = build_faiss(texts, model)

    # --- Lưu ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(outdir / "faiss.index"))
    with open(outdir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    df[["context_id","title","context","split","is_vietnam"]].to_parquet(
        outdir / "index_meta.parquet", index=False)

    print(f"\n✅ Đã lưu index ({len(df)} đoạn) -> {outdir}/")
    print(f"   faiss.index, bm25.pkl, index_meta.parquet")
    print(f"\n[QUAN TRỌNG] Test split KHÔNG nằm trong index.")
    print(f"   Kiểm tra: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
