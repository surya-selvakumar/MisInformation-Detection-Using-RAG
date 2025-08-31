from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from rouge_score import rouge_scorer
import bert_score

def macro_scores(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    return {"accuracy": float(acc), "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}

def explanation_scores(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = np.mean([rs.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps)])
    P, R, F1 = bert_score.score(hyps, refs, lang="en", verbose=False)
    return {"rougeL": float(rougeL), "bertscore_f1": float(F1.mean())}

def bootstrap_ci(values: List[float], n=1000, alpha=0.05, rng=None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(123)
    vals = np.array(values)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), len(vals))
        boots.append(vals[idx].mean())
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)
