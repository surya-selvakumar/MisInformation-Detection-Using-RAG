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

