"""
src/evaluation.py
-----------------
Rigorous Evaluation Benchmark Engine for Project 14 (Product Search using Amazon ESCI).

Implements the 4 core Information Retrieval metrics:
1. Precision@10: Ratio of relevant items in top-10.
2. Recall@K: Proportion of total relevant items retrieved.
3. MAP (Mean Average Precision): Ranking quality across the entire result list.
4. nDCG (Normalized Discounted Cumulative Gain): Handles graded multi-level ESCI relevance (E > S > C > I).

Also generates:
- Combined Precision-Recall (P-R) curves for all 4 models on a single figure.
- Comprehensive Performance Comparison Table (Markdown & CSV).
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Standard Graded Relevance Mapping for Amazon ESCI
# E (Exact) = 3, S (Substitute) = 2, C (Complement) = 1, I (Irrelevant) = 0
DEFAULT_ESCI_GAINS = {
    "E": 3.0,
    "S": 2.0,
    "C": 1.0,
    "I": 0.0
}

# Binary relevance cutoff: Items with gain >= 2.0 (Exact and Substitute) are relevant
BINARY_RELEVANCE_THRESHOLD = 2.0


def compute_precision_at_k(
    ranked_pids: List[str],
    ground_truth: Dict[str, float],
    k: int = 10,
    relevance_threshold: float = BINARY_RELEVANCE_THRESHOLD
) -> float:
    """
    Calculate Precision@K:
    Number of relevant items in top-K divided by K.
    """
    if k <= 0:
        return 0.0

    top_k = ranked_pids[:k]
    if not top_k:
        return 0.0

    rel_count = sum(1 for pid in top_k if ground_truth.get(pid, 0.0) >= relevance_threshold)
    return rel_count / k


def compute_recall_at_k(
    ranked_pids: List[str],
    ground_truth: Dict[str, float],
    k: int = 10,
    relevance_threshold: float = BINARY_RELEVANCE_THRESHOLD
) -> float:
    """
    Calculate Recall@K:
    Number of relevant items in top-K divided by total relevant items in candidate set.
    """
    total_relevant = sum(1 for gain in ground_truth.values() if gain >= relevance_threshold)
    if total_relevant == 0:
        return 0.0

    top_k = ranked_pids[:k]
    rel_retrieved = sum(1 for pid in top_k if ground_truth.get(pid, 0.0) >= relevance_threshold)
    return rel_retrieved / total_relevant


def compute_average_precision(
    ranked_pids: List[str],
    ground_truth: Dict[str, float],
    relevance_threshold: float = BINARY_RELEVANCE_THRESHOLD
) -> float:
    """
    Calculate Average Precision (AP) for a single query:
    AP = sum_{k=1}^N [ P@k * rel(k) ] / total_relevant
    """
    total_relevant = sum(1 for gain in ground_truth.values() if gain >= relevance_threshold)
    if total_relevant == 0:
        return 0.0

    running_rel = 0
    precision_sum = 0.0

    for i, pid in enumerate(ranked_pids, 1):
        if ground_truth.get(pid, 0.0) >= relevance_threshold:
            running_rel += 1
            precision_sum += (running_rel / i)

    return precision_sum / total_relevant


def compute_ndcg_at_k(
    ranked_pids: List[str],
    ground_truth: Dict[str, float],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K) with graded relevance:
    DCG@K = sum_{i=1}^K [ (2^gain - 1) / log2(i + 1) ]
    nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0:
        return 0.0

    top_k = ranked_pids[:k]
    if not top_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, pid in enumerate(top_k, 1):
        gain = ground_truth.get(pid, 0.0)
        dcg += (2.0 ** gain - 1.0) / np.log2(i + 1)

    # Ideal DCG (sorted true gains descending)
    ideal_gains = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = 0.0
    for i, gain in enumerate(ideal_gains, 1):
        idcg += (2.0 ** gain - 1.0) / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_interpolated_pr_curve(
    ranked_pids: List[str],
    ground_truth: Dict[str, float],
    relevance_threshold: float = BINARY_RELEVANCE_THRESHOLD,
    num_points: int = 11
) -> np.ndarray:
    """
    Compute 11-point interpolated precision values at standard recall levels:
    [0.0, 0.1, 0.2, ..., 1.0].
    """
    recall_levels = np.linspace(0.0, 1.0, num_points)
    total_relevant = sum(1 for gain in ground_truth.values() if gain >= relevance_threshold)
    if total_relevant == 0:
        return np.zeros(num_points)

    precisions = []
    recalls = []
    running_rel = 0

    for i, pid in enumerate(ranked_pids, 1):
        if ground_truth.get(pid, 0.0) >= relevance_threshold:
            running_rel += 1
        precisions.append(running_rel / i)
        recalls.append(running_rel / total_relevant)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # 11-point interpolation: p_interp(r) = max_{r' >= r} p(r')
    interpolated = []
    for r in recall_levels:
        eligible_p = precisions[recalls >= r]
        if len(eligible_p) > 0:
            interpolated.append(np.max(eligible_p))
        else:
            interpolated.append(0.0)

    return np.array(interpolated)


def evaluate_models_on_benchmark(
    models: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    k: int = 10,
    output_dir: str = "evaluation_results"
) -> Tuple[pd.DataFrame, str]:
    """
    Run full standardized evaluation benchmark across all models:
    - Precision@10
    - Recall@10
    - MAP
    - nDCG@10
    - Precision-Recall curves

    Args:
        models: Dictionary of {model_name: model_instance}.
        benchmark_df: DataFrame containing ['query_id', 'query', 'product_id', 'esci_label'].
        k: Ranking cutoff (default: 10).
        output_dir: Directory to save plots and comparison tables.

    Returns:
        Tuple[pd.DataFrame, str]: (Metrics DataFrame, Path to saved PR-curve plot).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group benchmark by query
    grouped = benchmark_df.groupby("query_id")
    print(f"Evaluating {len(models)} models across {len(grouped):,} test queries...")

    # Data structures to accumulate metrics per model
    model_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"P@10": [], "Recall": [], "MAP": [], "nDCG": []}
        for name in models
    }
    model_pr_curves: Dict[str, List[np.ndarray]] = {name: [] for name in models}

    for qid, group in grouped:
        query_str = group["query"].iloc[0]
        candidate_pids = list(group["product_id"].values)

        # Ground truth mapping: pid -> gain
        ground_truth = {
            row["product_id"]: DEFAULT_ESCI_GAINS.get(row["esci_label"], 0.0)
            for _, row in group.iterrows()
        }

        # Skip queries with zero relevant items
        if sum(1 for g in ground_truth.values() if g >= BINARY_RELEVANCE_THRESHOLD) == 0:
            continue

        for model_name, ranker in models.items():
            scored_candidates = ranker.score_candidates(query_str, candidate_pids)
            ranked_pids = [pid for pid, _ in scored_candidates]

            p_10 = compute_precision_at_k(ranked_pids, ground_truth, k=k)
            rec = compute_recall_at_k(ranked_pids, ground_truth, k=k)
            ap = compute_average_precision(ranked_pids, ground_truth)
            ndcg = compute_ndcg_at_k(ranked_pids, ground_truth, k=k)
            pr_11 = compute_interpolated_pr_curve(ranked_pids, ground_truth)

            model_metrics[model_name]["P@10"].append(p_10)
            model_metrics[model_name]["Recall"].append(rec)
            model_metrics[model_name]["MAP"].append(ap)
            model_metrics[model_name]["nDCG"].append(ndcg)
            model_pr_curves[model_name].append(pr_11)

    # Compute mean summary metrics
    summary_rows = []
    for name in models:
        row = {
            "Model": name,
            "Precision@10": np.mean(model_metrics[name]["P@10"]),
            "Recall": np.mean(model_metrics[name]["Recall"]),
            "MAP": np.mean(model_metrics[name]["MAP"]),
            "nDCG": np.mean(model_metrics[name]["nDCG"])
        }
        summary_rows.append(row)

    metrics_df = pd.DataFrame(summary_rows).set_index("Model")
    csv_path = os.path.join(output_dir, "model_comparison_table.csv")
    metrics_df.round(4).to_csv(csv_path)

    # Plot Precision-Recall curves on a single graph
    plt.figure(figsize=(9, 6))
    recall_axis = np.linspace(0.0, 1.0, 11)
    palette = {
        "TF-IDF": "#e67e22",
        "BM25": "#2980b9",
        "Sentence Transformer": "#8e44ad",
        "Hybrid (RRF)": "#27ae60"
    }

    for name in models:
        mean_curve = np.mean(model_pr_curves[name], axis=0)
        color = palette.get(name, None)
        plt.plot(recall_axis, mean_curve, marker="o", linewidth=2.2, label=name, color=color)

    plt.title("Precision-Recall Curves Across 4 Ranking Paradigms (Amazon ESCI)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall Level", fontsize=11)
    plt.ylabel("Interpolated Precision", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.0)
    plt.legend(title="Ranking Model", fontsize=10, loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    pr_plot_path = os.path.join(output_dir, "precision_recall_curves.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()

    print(f"\n[DONE] Master Comparison Table saved to: {csv_path}")
    print(f"[DONE] Precision-Recall Curve Plot saved to: {pr_plot_path}")

    return metrics_df, pr_plot_path


if __name__ == "__main__":
    # Self-test evaluation formulas on mock data
    mock_ranked = ["P1", "P2", "P3", "P4", "P5"]
    mock_gt = {"P1": 3.0, "P2": 2.0, "P3": 0.0, "P4": 1.0, "P5": 0.0}

    p10 = compute_precision_at_k(mock_ranked, mock_gt, k=5)
    rec = compute_recall_at_k(mock_ranked, mock_gt, k=5)
    ap = compute_average_precision(mock_ranked, mock_gt)
    ndcg = compute_ndcg_at_k(mock_ranked, mock_gt, k=5)

    print("=" * 60)
    print("DEMO: IR Metrics Verification")
    print(f"Precision@5: {p10:.4f}")
    print(f"Recall@5:    {rec:.4f}")
    print(f"MAP (AP):    {ap:.4f}")
    print(f"nDCG@5:      {ndcg:.4f}")
    print("=" * 60)
