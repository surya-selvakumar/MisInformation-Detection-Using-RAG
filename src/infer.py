import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.retrieval.retrieve import HybridRetriever
from src.utils.prompts import build_prompt
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--indices_dir", required=True)
    ap.add_argument("--claim", required=True)
    args = ap.parse_args()
    retriever = HybridRetriever(indices_dir=args.indices_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map="auto")

    evs = [t for t,_ in retriever.retrieve(args.claim, topk=5)]
    prompt = build_prompt(args.claim, evs)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=160, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
