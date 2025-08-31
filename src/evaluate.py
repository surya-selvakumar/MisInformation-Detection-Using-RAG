import argparse, os, ujson
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.retrieval.retrieve import HybridRetriever
from src.utils.prompts import build_prompt
from src.utils.metrics import macro_scores, explanation_scores

def parse_prediction(text: str):
    # Expect "Prediction: X. Explanation: ..."
    import re
    m = re.search(r"Prediction:\s*([A-Za-z\s]+)\.", text)
    label = m.group(1).strip() if m else "Not Enough Information"
    m2 = re.search(r"Explanation:\s*(.*)", text, re.S)
    expl = m2.group(1).strip() if m2 else ""
    return label, expl

def generate(model, tok, prompt, max_new_tokens=128):
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["finfact","finguard"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--indices_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args()

    retriever = HybridRetriever(indices_dir=args.indices_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map="auto")

    # load test set
    rows=[]
    with open(os.path.join(args.data_dir,"test.jsonl"),"r",encoding="utf-8") as f:
        for line in f:
            rows.append(ujson.loads(line))

    y_true, y_pred, refs, hyps = [], [], [], []
    for ex in tqdm(rows):
        claim = ex.get("claim", ex.get("text",""))
        evs = ex.get("evidence", None)
        if not evs:
            evs = [t for t,_ in retriever.retrieve(claim, topk=5)]
        prompt = build_prompt(claim, evs, restrict_binary=(args.dataset=="finguard"))
        out = generate(model, tok, prompt)
        label, expl = parse_prediction(out)
        gold = ex["label"]
        if args.dataset=="finguard":
            gold = {"Real":"True","Fake":"False"}[gold]
        y_true.append(gold); y_pred.append(label)
        if args.dataset=="finfact":
            refs.append(ex.get("rationale",""))  # optional reference rationale
            hyps.append(expl)

    # metrics
    labels = ["True","False"] if args.dataset=="finguard" else ["True","False","Not Enough Information"]
    scores = macro_scores(y_true, y_pred, labels=labels)
    print("Macro metrics:", scores)
    if args.dataset=="finfact" and all(isinstance(r,str) for r in refs) and any(refs):
        xs = explanation_scores(refs, hyps)
        print("Explanation metrics:", xs)

if __name__ == "__main__":
    main()
