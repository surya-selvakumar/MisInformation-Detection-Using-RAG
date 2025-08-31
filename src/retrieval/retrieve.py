import pickle, os, faiss, numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

def rrf_fuse(sparse_ranks, dense_ranks, k=60):
    # reciprocal rank fusion: sum 1/(k + rank)
    fused = {}
    for i, r in enumerate(sparse_ranks):
        fused[r] = fused.get(r, 0.0) + 1.0/(k + i + 1)
    for i, r in enumerate(dense_ranks):
        fused[r] = fused.get(r, 0.0) + 1.0/(k + i + 1)
    return [rid for rid, _ in sorted(fused.items(), key=lambda x: -x[1])]

class HybridRetriever:
    def __init__(self, indices_dir: str, encoder="sentence-transformers/all-mpnet-base-v2", rrf_k=60):
        self.rrf_k = rrf_k
        self.dense_idx = faiss.read_index(os.path.join(indices_dir, "dense.faiss"))
        with open(os.path.join(indices_dir, "dense_meta.pkl"), "rb") as f:
            dm = pickle.load(f)
        self.texts = dm["texts"]; self.meta = dm["meta"]
        with open(os.path.join(indices_dir, "bm25.pkl"), "rb") as f:
            bm = pickle.load(f)
        self.bm25 = bm["bm25"]
        self.st = SentenceTransformer(encoder)

    def retrieve(self, query: str, topk=5) -> List[Tuple[str, Dict]]:
        # Sparse ranks
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_order = list(np.argsort(-sparse_scores))
        # Dense ranks
        qv = self.st.encode([query], normalize_embeddings=True)
        D,I = self.dense_idx.search(qv, k=min(100, len(self.texts)))
        dense_order = list(I[0])
        # Fuse
        fused_ids = rrf_fuse(sparse_order[:100], dense_order[:100], k=self.rrf_k)
        out = []
        for idx in fused_ids[:topk]:
            out.append((self.texts[idx], self.meta[idx] if idx < len(self.meta) else {}))
        return out
