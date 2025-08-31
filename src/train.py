import os, argparse, json, numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from src.models.lora_model import load_lora_model
from src.models.hf_trainer import WeightedLabelTrainer
from src.retrieval.retrieve import HybridRetriever
from src.utils.prompts import build_prompt, tag_label
from src.datasets import load_split, to_hf_dataset
from tqdm import tqdm

def format_example(ex, retriever: HybridRetriever, restrict_binary: bool, topk: int):
    claim = ex["claim"] if "claim" in ex else ex["text"]
    evs = ex.get("evidence", None)
    if not evs:
        # retrieve from index if not provided
        evs = [t for t,_ in retriever.retrieve(claim, topk=topk)]
    prompt = build_prompt(claim, evs, restrict_binary=restrict_binary)
    # store gold text for SFT (label + explanation)
    gold_label = ex["label"]
    if "map" in ex and ex["map"]:  # for FinGuard
        gold_label = ex["map"].get(gold_label, gold_label)
    target = tag_label(gold_label, "Citing [1], [2].")  # minimal target; explanations can be teacher-forced if available
    return {"text": prompt + "\n" + target}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["finfact","finguard"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--indices_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    restrict_binary = (args.dataset == "finguard")

    # load model/tokenizer
    model, tok = load_lora_model(args.base_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # retriever
    retriever = HybridRetriever(indices_dir=args.indices_dir)

    # data
    train_rows = load_split(args.data_dir, "train")
    val_rows   = load_split(args.data_dir, "val")

    # map() to prompt+target text
    def _fmt(batch):
        out=[]
        for ex in batch:
            ex2 = dict(ex)
            d = format_example(ex2, retriever, restrict_binary, args.top_k)
            out.append(d)
        return {"text":[o["text"] for o in out]}

    ds_train = to_hf_dataset(train_rows).map(_fmt, batched=True, remove_columns=to_hf_dataset(train_rows).column_names)
    ds_val   = to_hf_dataset(val_rows).map(_fmt, batched=True, remove_columns=to_hf_dataset(val_rows).column_names)

    def tok_fn(examples):
        return tok(examples["text"], truncation=True, max_length=4096, padding=False)

    ds_train = ds_train.map(tok_fn, batched=True)
    ds_val   = ds_val.map(tok_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args_tr = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, fp16=args.fp16, logging_steps=50,
        evaluation_strategy="steps", eval_steps=500, save_steps=500, save_total_limit=2, lr_scheduler_type="cosine",
        report_to="none", load_best_model_at_end=True, metric_for_best_model="eval_loss"
    )
    trainer = WeightedLabelTrainer(
        model=model, args=args_tr, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=data_collator, tokenizer=tok
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
