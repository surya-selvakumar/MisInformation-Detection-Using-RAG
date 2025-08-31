# Misinformation Detection Using Affective Information and Retrieval-Augmented Generation

End-to-end, reproducible **Hugging Face** codebase for detecting (financial) misinformation with a **hybrid RAG** pipeline and concise, cited explanations. The system combines:

- **Hybrid retrieval**: BM25 (sparse) + Sentence-Transformer (dense) fused via **Reciprocal Rank Fusion (RRF)**.
- **Chunking**: 320-token windows with 64-token overlap for evidence passages.
- **Generation**: LoRA-tuned instruction LLM (e.g., Llama-3.1-8B-Instruct or Gemma-2-9B-it) producing  
  `Prediction: <True/False/Not Enough Information>. Explanation: ...` with citations `[1], [2]`.
- **Evaluation**: Macro Accuracy/Precision/Recall/F1; on FinFact, optional **ROUGE-L** and **BERTScore** for explanation quality.
- **Affective signals (optional)**: sentiment/emotion tags can be added to prompts as contextual features.

> Supports **FinFact** (3-class: True/False/NEI) and **FinGuard** (2-class: Real/Fake → mapped to True/False).

---

## Table of Contents

- [1. Repository Structure](#1-repository-structure)
- [2. Setup](#2-setup)
- [3. Data](#3-data)
- [4. Build Retrieval Indices](#4-build-retrieval-indices)
- [5. Train (LoRA)](#5-train-lora)
- [6. Evaluate](#6-evaluate)
- [7. Inference](#7-inference)
- [8. Configuration & Hyperparameters](#8-configuration--hyperparameters)
- [9. Reproducing the Research (Step-by-Step)](#9-reproducing-the-research-step-by-step)
- [10. Troubleshooting & FAQ](#10-troubleshooting--faq)
- [11. Results Artifacts](#11-results-artifacts)
- [12. License & Citation](#12-license--citation)

---

## 1. Repository Structure

```
.
├─ README.md
├─ requirements.txt
├─ config/
│  ├─ finfact.json            # task labels, top_k, RRF k, chunk sizes, BM25 params
│  └─ finguard.json
├─ data/                      # place your JSONL splits here (see §3)
│  ├─ finfact/{train,val,test}.jsonl
│  └─ finguard/{train,val,test}.jsonl
├─ indices/                   # created by §4 (bm25.pkl, dense_meta.pkl, dense.faiss)
├─ src/
│  ├─ datasets.py             # JSONL loading, token chunking helpers
│  ├─ train.py                # LoRA fine-tuning (HF Trainer)
│  ├─ evaluate.py             # metrics + (FinFact) explanation metrics
│  ├─ infer.py                # single-claim inference with retrieval
│  ├─ retrieval/
│  │  ├─ build_index.py       # builds BM25 + FAISS HNSW indices from corpus
│  │  └─ retrieve.py          # hybrid retriever with RRF fusion
│  ├─ models/
│  │  ├─ lora_model.py        # base model + LoRA (Q/V only by default)
│  │  └─ hf_trainer.py        # custom Trainer (label-token up-weighting)
│  └─ utils/
│     ├─ prompts.py           # system + prompt templates
│     └─ metrics.py           # macro metrics, ROUGE-L, BERTScore, bootstrap CIs
└─ scripts/
   └─ chunk_corpus.py         # optional: pre-chunk long articles (FinGuard)
```

---

## 2. Setup

### 2.1. Python environment

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate            # (Windows) .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> **Torch/CUDA**: If you require a specific CUDA build, install `torch` from pytorch.org **before** `requirements.txt`.

### 2.2. Hugging Face access

Some base models (e.g., Llama-3.x) are gated:

```bash
huggingface-cli login
```

If access is restricted, use `--base_model google/gemma-2-9b-it` in the commands below.

---

## 3. Data

**Google Drive (provided):**  
`https://drive.google.com/drive/folders/1qDH9z74Z3NpmyrJtmEF7ncEbnBqYt_wF?usp=sharing`

Place (or symlink) files to:

```
data/finfact/train.jsonl
data/finfact/val.jsonl
data/finfact/test.jsonl

data/finguard/train.jsonl
data/finguard/val.jsonl
data/finguard/test.jsonl
```

**Expected formats**

**FinFact (3-class):**
```json
{"id":"...", "claim":"...", "evidence":["paragraph 1","paragraph 2", "..."], "label":"True"}
```

**FinGuard (2-class):**
```json
{"id":"...", "text":"full article text...", "label":"Real"}  // "Real" or "Fake" → mapped to True/False
```

> If your field names differ, update `src/retrieval/build_index.py` (`iter_corpus`) and the example formatting in `src/train.py`.

---

## 4. Build Retrieval Indices

This **creates** the on-disk indices used by training/evaluation/inference:

- `indices/bm25.pkl` — BM25 index & texts  
- `indices/dense_meta.pkl` — texts + per-chunk metadata  
- `indices/dense.faiss` — FAISS HNSW dense index

```bash
python -m src.retrieval.build_index   --corpus data/finfact/train.jsonl data/finfact/val.jsonl data/finfact/test.jsonl   --extra_corpus data/finguard/train.jsonl data/finguard/val.jsonl data/finguard/test.jsonl   --out_dir indices   --encoder sentence-transformers/all-mpnet-base-v2   --hnsw_m 32 --hnsw_efC 200
```

> **Long articles?** You can pre-chunk FinGuard:
```bash
python -m scripts.chunk_corpus   --in_file data/finguard/train.jsonl   --out_file data/finguard/train.chunks.jsonl   --base_model meta-llama/Meta-Llama-3.1-8B-Instruct   --chunk_size 320 --overlap 64
# then pass *.chunks.jsonl via --extra_corpus when building indices
```

---

## 5. Train (LoRA)

LoRA on Q/V (`r=16`, `alpha=32`, `dropout=0.05`). Adjust base model to fit access/VRAM.

**FinFact (3-class):**
```bash
python -m src.train   --dataset finfact   --data_dir data/finfact   --indices_dir indices   --base_model meta-llama/Meta-Llama-3.1-8B-Instruct   --output_dir outputs/finfact_lora   --per_device_train_batch_size 1   --gradient_accumulation_steps 32   --num_train_epochs 3   --learning_rate 2e-5   --warmup_steps 500   --fp16   --gradient_checkpointing   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05   --top_k 5
```

**FinGuard (2-class):**
```bash
python -m src.train   --dataset finguard   --data_dir data/finguard   --indices_dir indices   --base_model google/gemma-2-9b-it   --output_dir outputs/finguard_lora   --per_device_train_batch_size 1   --gradient_accumulation_steps 32   --num_train_epochs 3   --learning_rate 2e-5   --warmup_steps 500   --fp16   --gradient_checkpointing   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05   --top_k 5
```

> **Hardware tips**: 1× 16–24 GB GPU is ideal. If OOM, lower context length, reduce batch/accumulation, or use a smaller model.

---

## 6. Evaluate

Computes macro metrics; on FinFact, also ROUGE-L & BERTScore for explanations.

```bash
# FinFact
python -m src.evaluate   --dataset finfact   --data_dir data/finfact   --indices_dir indices   --model_dir outputs/finfact_lora   --bootstrap 1000   # optional: CI via bootstrap
```

```bash
# FinGuard
python -m src.evaluate   --dataset finguard   --data_dir data/finguard   --indices_dir indices   --model_dir outputs/finguard_lora
```

The script prints `accuracy, macro_precision, macro_recall, macro_f1` and (FinFact) `rougeL, bertscore_f1` if reference rationales exist.

---

## 7. Inference

Judge a single claim with retrieved evidence:

```bash
python -m src.infer   --model_dir outputs/finfact_lora   --indices_dir indices   --claim "Company X increased total revenue by 10% YoY in Q4 2023."
```

Output includes **Prediction** and **Explanation** with citations like `[1], [2]`.

---

## 8. Configuration & Hyperparameters

See `config/finfact.json` and `config/finguard.json`:

```json
{
  "top_k": 5,
  "rrf_k": 60,
  "chunk_size": 320,
  "chunk_overlap": 64,
  "bm25": {"k1": 1.5, "b": 0.75}
}
```

**Defaults used**
- Retrieval: BM25 (k1=1.5, b=0.75) + `all-mpnet-base-v2` dense encoder → FAISS HNSW; **RRF k=60**; `top_k=5`.
- Chunking: 320 / overlap 64.
- LoRA: r=16, alpha=32, dropout=0.05; target modules: `q_proj`, `v_proj`.
- Trainer: AdamW 2e-5, warmup 500, cosine decay, fp16, grad checkpointing, batch size 1 + grad-accum 32.

---

## 9. Reproducing the Research (Step-by-Step)

1. **Create env** and `pip install -r requirements.txt`.
2. **Login to HF** (`huggingface-cli login`) or choose an open base model (`google/gemma-2-9b-it`).
3. **Download data** from the Drive link and place under `data/...` as shown in §3.
4. **Build indices** (Section 4) → creates `indices/{bm25.pkl,dense_meta.pkl,dense.faiss}`.
5. **Train** for FinFact and/or FinGuard (Section 5).
6. **Evaluate** (Section 6) to obtain macro metrics and, if available, explanation metrics.
7. **Inference** (Section 7) for qualitative examples.

> **Determinism**: Add to `src/train.py`:
> ```python
> from transformers.trainer_utils import set_seed
> set_seed(42)
> ```
> and set `PYTHONHASHSEED=0`. Some variation is still expected.

---

## 10. Troubleshooting & FAQ

- **Where do the `.pkl` files come from?**  
  They are **generated** by `src/retrieval/build_index.py` (see §4).
- **Data “file not found”**  
  Ensure exact paths: `data/{finfact|finguard}/{train|val|test}.jsonl`.
- **Gated model access**  
  `huggingface-cli login`, or switch to `--base_model google/gemma-2-9b-it`.
- **CUDA OOM**  
  Reduce context length, lower batch/accumulation, or use a smaller model.
- **FAISS/BLAS issues**  
  Use `faiss-cpu` (already in `requirements.txt`). Apple Silicon: CPU build works.
- **Different schemas**  
  Adapt `iter_corpus` in `src/retrieval/build_index.py` and example formatting in `src/train.py`.

---

## 11. Results Artifacts

During training/evaluation you will get:
- `outputs/<run_name>/` — final LoRA-adapted weights + tokenizer
- `indices/` — BM25 + FAISS indices
- Console logs with metrics (pipe to files if desired)

Export results to a file:
```bash
python -m src.evaluate ... > results_finfact.json
```

---

## 12. License & Citation

**License**: MIT (update if your institution requires otherwise).

If you use this repository, please cite:
```bibtex
@software{misinfo_affective_rag_2025,
  title        = {Misinformation Detection Using Affective Information and Retrieval-Augmented Generation},
  author       = {<Your Name>},
  year         = {2025},
  url          = {https://github.com/<your-user-or-org>/<your-repo>}
}
```

**Maintainer:** <Your Name> · your.email@domain
