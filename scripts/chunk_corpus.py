import argparse, ujson, os
from transformers import AutoTokenizer
from src.datasets import chunk_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    with open(args.in_file, "r", encoding="utf-8") as f, open(args.out_file, "w", encoding="utf-8") as g:
        for line in f:
            ex = ujson.loads(line)
            chunks = chunk_text(ex["text"], tok, args.chunk_size, args.overlap)
            for i, ch in enumerate(chunks):
                g.write(ujson.dumps({"id": f"{ex['id']}::chunk{i}", "text": ch, "label": ex["label"]})+"\n")

if __name__ == "__main__":
    main()
