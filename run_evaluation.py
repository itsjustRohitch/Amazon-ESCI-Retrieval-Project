"""
run_evaluation.py
-----------------
Master Evaluation Runner for Project 14: Product Search using Amazon ESCI.

1. Loads the benchmark dataset (real Amazon queries, candidate products, and ESCI labels).
2. Fits all 4 Core Ranking Paradigms:
   - Model 1: TF-IDF (Vector Space Model)
   - Model 2: BM25 (Probabilistic Okapi BM25)
   - Model 3: Sentence Transformer (Dense Semantic Vectors via all-MiniLM-L6-v2)
   - Model 4: Hybrid RRF (Reciprocal Rank Fusion blending BM25 + Dense)
3. Evaluates Precision@10, Recall, MAP, and nDCG across test queries.
4. Generates:
   - Precision-Recall curves plotted on a single graph ('evaluation_results/precision_recall_curves.png')
   - Performance comparison table ('evaluation_results/model_comparison_table.csv')
"""

import os
import time
import pandas as pd
import numpy as np
from tabulate import tabulate

from src.models.tfidf_ranker import TFIDFRanker
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion
from src.evaluation import evaluate_models_on_benchmark


def main():
    print("=" * 80)
    print("PROJECT 14: STANDARDIZED INFORMATION RETRIEVAL BENCHMARK EVALUATION")
    print("Dataset: Amazon ESCI (Shopping Queries Dataset - Task 1 English)")
    print("=" * 80)

    benchmark_path = os.path.join("data", "benchmark_sample.parquet")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")

    print(f"\n[Step 1/4] Loading benchmark data from {benchmark_path}...")
    df_benchmark = pd.read_parquet(benchmark_path)
    total_rows = len(df_benchmark)
    n_queries = df_benchmark["query_id"].nunique()
    n_products = df_benchmark["product_id"].nunique()
    print(f"Loaded {total_rows:,} candidate judgments across {n_queries:,} test queries and {n_products:,} unique products.")

    # Evaluate on a representative sample of 250 queries for swift execution
    eval_queries = df_benchmark["query_id"].unique()
    np.random.seed(42)
    sample_qids = np.random.choice(eval_queries, size=min(250, len(eval_queries)), replace=False)
    df_eval = df_benchmark[df_benchmark["query_id"].isin(sample_qids)].copy()
    print(f"Selected {len(sample_qids)} evaluation queries ({len(df_eval):,} candidate judgments).")

    # Build unique catalog for model indexing
    unique_catalog = df_eval[["product_id", "product_title"]].drop_duplicates(subset=["product_id"]).copy()
    corpus = unique_catalog["product_title"].fillna("").tolist()
    product_ids = unique_catalog["product_id"].tolist()
    print(f"Extracted unique candidate product catalog: {len(corpus):,} items.")

    print("\n[Step 2/4] Initializing and fitting the 4 Core Ranking Models...")
    t0 = time.time()

    # 1. TF-IDF
    print("  -> Fitting Model 1: TF-IDF (Vector Space Model)...")
    tfidf = TFIDFRanker().fit(corpus, product_ids)

    # 2. BM25
    print("  -> Fitting Model 2: BM25 (Okapi BM25 Probabilistic)...")
    bm25 = BM25Ranker(k1=1.5, b=0.75).fit(corpus, product_ids)

    # 3. Dense Transformer
    print("  -> Fitting Model 3: Sentence Transformer (all-MiniLM-L6-v2 dense embeddings)...")
    dense = DenseRanker().fit(corpus, product_ids, batch_size=128, show_progress_bar=False)

    # 4. Hybrid RRF
    print("  -> Fitting Model 4: Hybrid RRF (BM25 + Dense Ranker fusion)...")
    hybrid = ReciprocalRankFusion(rankers=[bm25, dense], weights=[1.0, 1.0], k=60)

    print(f"All 4 models successfully initialized and fitted in {time.time() - t0:.2f}s.")

    models = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "Sentence Transformer": dense,
        "Hybrid (RRF)": hybrid
    }

    print("\n[Step 3/4] Running quantitative benchmark across Precision@10, Recall, MAP, and nDCG...")
    t_eval = time.time()
    metrics_df, pr_plot_path = evaluate_models_on_benchmark(
        models=models,
        benchmark_df=df_eval,
        k=10,
        output_dir="evaluation_results"
    )
    print(f"Benchmark completed in {time.time() - t_eval:.2f}s.")

    print("\n[Step 4/4] Final Deliverables Generated:")
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(tabulate(metrics_df.round(4), headers="keys", tablefmt="github"))
    print(f"\nSaved CSV Table:     file:///{os.path.abspath('evaluation_results/model_comparison_table.csv').replace(chr(92), '/')}")
    print(f"Saved P-R Curves:    file:///{os.path.abspath(pr_plot_path).replace(chr(92), '/')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
