import json, os, ujson
from typing import Dict, List
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

@dataclass
class DataConfig:
    task: str
    labels: List[str]
    map: Dict[str,str] = None  # for FinGuard Real/Fake -> True/False

def load_jsonl(path): 
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(ujson.loads(line))
    return rows

def load_split(data_dir: str, split: str):
    return load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))

def to_hf_dataset(rows: List[dict]) -> Dataset:
    return Dataset.from_list(rows)

def compute_class_weights(labels: List[str], y: List[str]):
    from collections import Counter
    import numpy as np
    c = Counter(y)
    total = sum(c.values())
    freqs = {k: v/total for k,v in c.items()}
    w = {k: (freqs[k] ** -0.5) for k in freqs}  # inverse sqrt freq :contentReference[oaicite:11]{index=11}
    z = sum(w.values())
    for k in w: w[k] = w[k]/z
    return w

def chunk_text(text: str, tokenizer: AutoTokenizer, chunk_size=320, overlap=64):
    # sentence aware optional; here use token overlaps per paper (320/64) :contentReference[oaicite:12]{index=12}
    toks = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks=[]
    i=0
    while i < len(toks):
        seg = toks[i:i+chunk_size]
        if not seg: break
        chunks.append(tokenizer.decode(seg))
        i += (chunk_size - overlap)
    return chunks
