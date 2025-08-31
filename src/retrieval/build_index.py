import argparse, json, os, pickle
from typing import List, Dict
import faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ujson

def iter_corpus(paths: List[str], field: str) -> List[Dict]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = ujson.loads(line)
                yield obj

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())

def build_dense(texts: List[str], model_name: str, hnsw_m: int, hnsw_efC: int):
    st = SentenceTransformer(model_name)
    X = st.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, hnsw_m)
    index.hnsw.efConstruction = hnsw_efC
    index.add(X)
    return index, X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="finfact jsonl files")
    ap.add_argument("--extra_corpus", nargs="*", default=[], help="finguard jsonl files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--hnsw_efC", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # collect texts (FinFact evidence + optional long articles chunked externally)
    texts = []
    meta = []
    for obj in iter_corpus(args.corpus + args.extra_corpus, "text"):
        if "evidence" in obj:
            for j, ev in enumerate(obj["evidence"]):
                evn = normalize_text(ev)
                if evn:
                    meta.append({"id": obj.get("id"), "kind": "evidence", "i": j})
                    texts.append(evn)
        elif "text" in obj:
            # treat long article as one unit (optional: pre-chunk with scripts/chunk_corpus.py)
            t = normalize_text(obj["text"])
            if t:
                meta.append({"id": obj.get("id"), "kind": "article"})
                texts.append(t)

    # BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)  # per paper :contentReference[oaicite:10]{index=10}

    # Dense + FAISS HNSW
    index, X = build_dense(texts, args.encoder, args.hnsw_m, args.hnsw_efC)

    # save
    faiss.write_index(index, os.path.join(args.out_dir, "dense.faiss"))
    with open(os.path.join(args.out_dir, "dense_meta.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "meta": meta}, f)
    with open(os.path.join(args.out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts}, f)

if __name__ == "__main__":
    main()
