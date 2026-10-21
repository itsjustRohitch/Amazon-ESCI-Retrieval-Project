"""
src/evaluation.py
-----------------
Metrics and evaluation loops for the ESCI benchmark.

Relevance
    Graded gains for nDCG: E = 3, S = 2, C = 1, I = 0.
    Binary metrics (P@k, Recall@k, AP, PR curve) count only Exact matches as
    relevant. In this sample 79% of judged candidates are E or S, so treating
    E+S as relevant makes even a random ordering score P@10 near 0.8 and hides
    the differences between models.

Protocols
    rerank     rank only the judged candidates of each query (ESCI Task 1)
    retrieval  rank the whole catalog; unjudged products count as irrelevant
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ESCI_GAINS = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}
BINARY_RELEVANCE_THRESHOLD = 3.0
PR_RECALL_LEVELS = np.linspace(0.0, 1.0, 11)


def _n_relevant(ground_truth: Dict[str, float], threshold: float) -> int:
    return sum(1 for g in ground_truth.values() if g >= threshold)


def compute_precision_at_k(ranked_pids, ground_truth, k=10, relevance_threshold=BINARY_RELEVANCE_THRESHOLD):
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if ground_truth.get(pid, 0.0) >= relevance_threshold)
    # divide by k (not by the list length) so short candidate lists are not rewarded
    return hits / k


def compute_recall_at_k(ranked_pids, ground_truth, k=10, relevance_threshold=BINARY_RELEVANCE_THRESHOLD):
    total = _n_relevant(ground_truth, relevance_threshold)
    if total == 0:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if ground_truth.get(pid, 0.0) >= relevance_threshold)
    return hits / total


def compute_average_precision(ranked_pids, ground_truth, relevance_threshold=BINARY_RELEVANCE_THRESHOLD):
    total = _n_relevant(ground_truth, relevance_threshold)
    if total == 0:
        return 0.0
    hits, acc = 0, 0.0
    for i, pid in enumerate(ranked_pids, 1):
        if ground_truth.get(pid, 0.0) >= relevance_threshold:
            hits += 1
            acc += hits / i
    return acc / total


def compute_ndcg_at_k(ranked_pids, ground_truth, k=10):
    """nDCG@k with exponential gain (2^g - 1) / log2(i + 1)."""
    if k <= 0:
        return 0.0
    dcg = sum((2.0 ** ground_truth.get(pid, 0.0) - 1.0) / np.log2(i + 1)
              for i, pid in enumerate(ranked_pids[:k], 1))
    ideal = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / np.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_interpolated_pr_curve(ranked_pids, ground_truth,
                                  relevance_threshold=BINARY_RELEVANCE_THRESHOLD, num_points=11):
    """11-point interpolated precision: p_interp(r) = max_{r' >= r} p(r')."""
    levels = np.linspace(0.0, 1.0, num_points)
    total = _n_relevant(ground_truth, relevance_threshold)
    if total == 0 or not ranked_pids:
        return np.zeros(num_points)

    rel = np.array([ground_truth.get(pid, 0.0) >= relevance_threshold for pid in ranked_pids])
    hits = np.cumsum(rel)
    precision = hits / np.arange(1, len(rel) + 1)
    recall = hits / total
    return np.array([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in levels])


def build_qrels(benchmark_df: pd.DataFrame) -> Dict[Any, Dict[str, float]]:
    qrels: Dict[Any, Dict[str, float]] = {}
    for qid, pid, lab in benchmark_df[["query_id", "product_id", "esci_label"]].itertuples(index=False):
        qrels.setdefault(qid, {})[pid] = DEFAULT_ESCI_GAINS.get(lab, 0.0)
    return qrels


def evaluate_run(
    run: Dict[Any, List[str]],
    qrels: Dict[Any, Dict[str, float]],
    k: int = 10,
    deep_k: int = 100,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Score one system. `run` maps query_id -> ranked product ids.
    Returns a per-query metric table and the mean 11-point PR curve.
    Queries with no Exact product are skipped (AP and recall are undefined).
    """
    rows, curves = [], []
    for qid, ranked in run.items():
        gt = qrels[qid]
        if _n_relevant(gt, BINARY_RELEVANCE_THRESHOLD) == 0:
            continue
        rows.append({
            "query_id": qid,
            f"P@{k}": compute_precision_at_k(ranked, gt, k),
            f"R@{k}": compute_recall_at_k(ranked, gt, k),
            f"R@{deep_k}": compute_recall_at_k(ranked, gt, deep_k),
            "AP": compute_average_precision(ranked, gt),
            f"nDCG@{k}": compute_ndcg_at_k(ranked, gt, k),
        })
        curves.append(compute_interpolated_pr_curve(ranked, gt))
    return pd.DataFrame(rows).set_index("query_id"), np.mean(curves, axis=0)


def summarise(per_query: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    table = pd.DataFrame({name: df.mean() for name, df in per_query.items()}).T
    table = table.rename(columns={"AP": "MAP"})
    table.index.name = "Model"
    return table


def paired_tests(per_query: Dict[str, pd.DataFrame], metric: str = "nDCG@10") -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank test for every pair of systems."""
    from scipy.stats import wilcoxon

    names = list(per_query)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            x, y = per_query[a][metric].align(per_query[b][metric], join="inner")
            diff = x - y
            out.loc[a, b] = 1.0 if (diff == 0).all() else wilcoxon(x, y, zero_method="zsplit").pvalue
    return out


def run_rerank(ranker, queries: pd.DataFrame, candidates: Dict[Any, List[str]]) -> Dict[Any, List[str]]:
    return {qid: [pid for pid, _ in ranker.score_candidates(q, candidates[qid])]
            for qid, q in queries[["query_id", "query"]].itertuples(index=False)}


def run_retrieval(ranker, queries: pd.DataFrame, depth: int = 100) -> Dict[Any, List[str]]:
    return {qid: [pid for pid, _ in ranker.rank(q, top_k=depth)]
            for qid, q in queries[["query_id", "query"]].itertuples(index=False)}


def evaluate_models_on_benchmark(
    models: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    k: int = 10,
    output_dir: Optional[str] = "evaluation_results",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Re-ranking evaluation of several models. Kept for scripts that only need the table."""
    qrels = build_qrels(benchmark_df)
    queries = benchmark_df[["query_id", "query"]].drop_duplicates("query_id")
    candidates = shuffled_candidates(benchmark_df, seed)
    per_query, curves = {}, {}
    for name, ranker in models.items():
        per_query[name], curves[name] = evaluate_run(run_rerank(ranker, queries, candidates), qrels, k)
    table = summarise(per_query)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        table.round(4).to_csv(os.path.join(output_dir, "model_comparison_table.csv"))
    return table, curves


def shuffled_candidates(benchmark_df: pd.DataFrame, seed: int = 42) -> Dict[Any, List[str]]:
    """
    Candidate lists per query in a fixed random order. Sparse models give many
    candidates the same score (often zero) and a stable sort keeps their input
    order, so the order must not carry any information about the labels.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for qid, grp in benchmark_df.groupby("query_id", sort=False):
        pids = grp["product_id"].tolist()
        rng.shuffle(pids)
        out[qid] = pids
    return out


if __name__ == "__main__":
    ranked = ["P1", "P2", "P3", "P4", "P5"]
    gt = {"P1": 3.0, "P2": 2.0, "P3": 0.0, "P4": 3.0, "P5": 0.0}
    print(f"P@5    {compute_precision_at_k(ranked, gt, 5):.4f}")
    print(f"R@5    {compute_recall_at_k(ranked, gt, 5):.4f}")
    print(f"AP     {compute_average_precision(ranked, gt):.4f}")
    print(f"nDCG@5 {compute_ndcg_at_k(ranked, gt, 5):.4f}")
