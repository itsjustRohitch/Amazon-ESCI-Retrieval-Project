"""
run_evaluation.py
-----------------
Full benchmark for Project 14 (Amazon ESCI product search).

    1. Build the catalog (every product judged in the sample) and split the
       1,000 queries into 300 dev and 700 test queries.
    2. On dev only (settings picked by mean nDCG@10 over both protocols):
       BM25 k1/b grid, TF-IDF weighting variants, preprocessing
       ablations and the RRF constant / combination sweep.
    3. On test: the four models with the settings chosen on dev, under both
       protocols (re-ranking judged candidates, full-catalog retrieval),
       plus a random baseline, significance tests and latency.

Everything is written to evaluation_results/. Figures are drawn separately by
report/make_figures.py from these files.
"""

import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.evaluation import (build_qrels, evaluate_run, paired_tests, run_rerank,
                            run_retrieval, shuffled_candidates, summarise)
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion
from src.models.tfidf_ranker import TFIDFRanker
from src.preprocessing import preprocess_text

OUT = "evaluation_results"
DATA = "data"
SEED = 42
N_DEV = 300
DEPTH = 100


def product_text(title, brand):
    title = title or ""
    if isinstance(brand, str) and brand and brand.lower() not in title.lower():
        return f"{title} {brand}"
    return title


def build_catalog(df):
    cat = df[["product_id", "product_title", "product_brand"]].drop_duplicates("product_id").reset_index(drop=True)
    cat["product_title"] = cat["product_title"].fillna("")
    cat["text"] = [product_text(t, b) for t, b in zip(cat.product_title, cat.product_brand)]
    return cat


def load_dense(catalog):
    """Encode the catalog once and cache it; later runs and the web app reuse the cache."""
    cat_path = os.path.join(DATA, "catalog_cache.parquet")
    emb_path = os.path.join(DATA, "embeddings_cache.npy")
    dense = DenseRanker()
    pids = catalog.product_id.tolist()
    if os.path.exists(cat_path) and os.path.exists(emb_path):
        cached = pd.read_parquet(cat_path)
        if cached.product_id.tolist() == pids and "text" in cached:
            return dense.load_cached_embeddings(np.load(emb_path), pids)
    print("  encoding catalog with the sentence transformer (one time)...")
    dense.fit(catalog.text.tolist(), pids, batch_size=128, show_progress_bar=True)
    catalog[["product_id", "product_title", "product_brand", "text"]].to_parquet(cat_path, index=False)
    np.save(emb_path, dense.doc_embeddings.astype(np.float32))
    return dense


class RandomRanker:
    """Lower bound for the re-ranking protocol."""

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def score_candidates(self, query, candidate_ids):
        order = list(candidate_ids)
        self.rng.shuffle(order)
        return [(pid, 0.0) for pid in order]


def analyzer(stem=True, stop=True):
    return lambda t: preprocess_text(t, apply_stemming=stem, apply_stopwords=stop)


def ndcg(run, qrels):
    return evaluate_run(run, qrels)[0]["nDCG@10"].mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    df = pd.read_parquet(os.path.join(DATA, "benchmark_sample.parquet"))
    catalog = build_catalog(df)
    corpus, pids = catalog.text.tolist(), catalog.product_id.tolist()
    qrels = build_qrels(df)
    candidates = shuffled_candidates(df, SEED)
    queries = df[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    dev_ids = set(rng.choice(queries.query_id, size=N_DEV, replace=False).tolist())
    dev = queries[queries.query_id.isin(dev_ids)]
    test = queries[~queries.query_id.isin(dev_ids)]
    print(f"{len(df):,} judgments | {len(queries):,} queries ({len(dev)} dev / {len(test)} test) | {len(catalog):,} products")

    # ------------------------------------------------------------------ fit
    t0 = time.time(); bm25 = BM25Ranker().fit(corpus, pids); t_bm25 = time.time() - t0
    t0 = time.time(); tfidf = TFIDFRanker().fit(corpus, pids); t_tfidf = time.time() - t0
    t0 = time.time(); dense = load_dense(catalog); t_dense = time.time() - t0
    index_stats = bm25.index.stats()
    print("index:", index_stats)

    # ------------------------------------------------------------------ dev: BM25 grid
    print("\n[dev] BM25 k1 x b grid")
    grid = []
    for k1 in [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]:
        for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bm25.set_params(k1, b)
            grid.append({"k1": k1, "b": b,
                         "rerank": ndcg(run_rerank(bm25, dev, candidates), qrels),
                         "retrieval": ndcg(run_retrieval(bm25, dev, DEPTH), qrels)})
    grid = pd.DataFrame(grid)
    grid["both"] = (grid.rerank + grid.retrieval) / 2
    grid.to_csv(os.path.join(OUT, "dev_bm25_grid.csv"), index=False)
    best = grid.sort_values("both", ascending=False).iloc[0]
    bm25.set_params(float(best.k1), float(best.b))
    print(f"  best k1={best.k1} b={best.b}  mean nDCG={best.both:.4f}")

    # ------------------------------------------------------------------ dev: TF-IDF weighting
    print("[dev] TF-IDF weighting variants")
    variants = {"raw tf, idf": dict(sublinear_tf=False, use_idf=True),
                "log tf, idf": dict(sublinear_tf=True, use_idf=True),
                "raw tf, no idf": dict(sublinear_tf=False, use_idf=False),
                "log tf, no idf": dict(sublinear_tf=True, use_idf=False)}
    tf_rows, fitted = [], {}
    for name, kw in variants.items():
        r = TFIDFRanker(**kw).fit(corpus, pids)
        fitted[name] = r
        tf_rows.append({"variant": name,
                        "rerank": ndcg(run_rerank(r, dev, candidates), qrels),
                        "retrieval": ndcg(run_retrieval(r, dev, DEPTH), qrels)})
    tf_rows = pd.DataFrame(tf_rows)
    tf_rows["both"] = (tf_rows.rerank + tf_rows.retrieval) / 2
    tf_rows.to_csv(os.path.join(OUT, "dev_tfidf_variants.csv"), index=False)
    best_tf = tf_rows.sort_values("both", ascending=False).iloc[0].variant
    tfidf = fitted[best_tf]
    print(tf_rows.round(4).to_string(index=False))

    # ------------------------------------------------------------------ dev: preprocessing ablation
    print("[dev] preprocessing ablation")
    abl = []
    for label, stem, stop in [("stem + stopwords", True, True), ("no stemming", False, True),
                              ("keep stopwords", True, False), ("neither", False, False)]:
        a = analyzer(stem, stop)
        b_r = BM25Ranker(k1=bm25.k1, b=bm25.b, analyzer=a).fit(corpus, pids)
        t_r = TFIDFRanker(**variants[best_tf], analyzer=a).fit(corpus, pids)
        for model, r in [("BM25", b_r), ("TF-IDF", t_r)]:
            abl.append({"setting": label, "model": model, "vocabulary": r.index.vocab_size,
                        "rerank": ndcg(run_rerank(r, dev, candidates), qrels),
                        "retrieval": ndcg(run_retrieval(r, dev, DEPTH), qrels)})
    pd.DataFrame(abl).to_csv(os.path.join(OUT, "dev_preprocessing_ablation.csv"), index=False)

    # ------------------------------------------------------------------ dev: RRF
    print("[dev] RRF constant and combinations")
    rrf_rows = []
    for k in [1, 5, 10, 20, 40, 60, 100, 200]:
        h = ReciprocalRankFusion([bm25, dense], k=k)
        rrf_rows.append({"k": k, "combo": "BM25 + Dense",
                         "rerank": ndcg(run_rerank(h, dev, candidates), qrels),
                         "retrieval": ndcg(run_retrieval(h, dev, DEPTH), qrels)})
    rrf_k = pd.DataFrame(rrf_rows)
    rrf_k["both"] = (rrf_k.rerank + rrf_k.retrieval) / 2
    best_k = int(rrf_k.sort_values("both", ascending=False).iloc[0].k)
    combos = {"BM25 + Dense": [bm25, dense], "TF-IDF + Dense": [tfidf, dense],
              "BM25 + TF-IDF": [bm25, tfidf], "All three": [bm25, tfidf, dense]}
    combo_rows = []
    for name, rs in combos.items():
        h = ReciprocalRankFusion(rs, k=best_k)
        combo_rows.append({"combo": name,
                           "rerank": ndcg(run_rerank(h, dev, candidates), qrels),
                           "retrieval": ndcg(run_retrieval(h, dev, DEPTH), qrels)})
    rrf_k.to_csv(os.path.join(OUT, "dev_rrf_k.csv"), index=False)
    pd.DataFrame(combo_rows).to_csv(os.path.join(OUT, "dev_rrf_combos.csv"), index=False)
    print(f"  best k={best_k}")

    # ------------------------------------------------------------------ test
    hybrid = ReciprocalRankFusion([bm25, dense], k=best_k)
    models = {"TF-IDF": tfidf, "BM25": bm25, "Sentence Transformer": dense, "Hybrid (RRF)": hybrid}

    print("\n[test] re-ranking protocol")
    rr_q, rr_c, rr_runs = {}, {}, {}
    for name, m in {**models, "Random": RandomRanker()}.items():
        rr_runs[name] = run_rerank(m, test, candidates)
        rr_q[name], rr_c[name] = evaluate_run(rr_runs[name], qrels)

    print("[test] full-catalog retrieval protocol")
    rt_q, rt_c, latency = {}, {}, {}
    for m in models.values():  # warm-up, so the first timed model does not pay for loading
        run_retrieval(m, test.head(20), DEPTH)
    for name, m in models.items():
        t0 = time.perf_counter()
        run = run_retrieval(m, test, DEPTH)
        latency[name] = (time.perf_counter() - t0) * 1000 / len(test)
        rt_q[name], rt_c[name] = evaluate_run(run, qrels)

    # every judged candidate is inside the first 100 when re-ranking, so R@100 is always 1 there
    rerank_table, retrieval_table = summarise(rr_q).drop(columns="R@100"), summarise(rt_q)
    rerank_table.to_csv(os.path.join(OUT, "metrics_rerank.csv"))
    retrieval_table.to_csv(os.path.join(OUT, "metrics_retrieval.csv"))
    rerank_table.drop(index="Random").round(4).to_csv(os.path.join(OUT, "model_comparison_table.csv"))
    for tag, per_q in [("rerank", rr_q), ("retrieval", rt_q)]:
        pd.concat(per_q, names=["model"]).to_csv(os.path.join(OUT, f"per_query_{tag}.csv"))
        pd.DataFrame({k: v for k, v in (rr_c if tag == "rerank" else rt_c).items()}).to_csv(
            os.path.join(OUT, f"pr_curve_{tag}.csv"), index_label="recall_level_idx")
    paired_tests({k: v for k, v in rr_q.items() if k != "Random"}).to_csv(os.path.join(OUT, "wilcoxon_rerank.csv"))
    paired_tests(rt_q).to_csv(os.path.join(OUT, "wilcoxon_retrieval.csv"))

    # ------------------------------------------------------------------ analysis tables for the report
    # label mix at each rank position (re-ranking)
    pos_rows = []
    for name, run in rr_runs.items():
        for qid, ranked in run.items():
            for pos, pid in enumerate(ranked[:10], 1):
                g = qrels[qid][pid]
                pos_rows.append({"model": name, "pos": pos, "label": "ESCI"[[3, 2, 1, 0].index(int(g))]})
    pd.DataFrame(pos_rows).groupby(["model", "pos", "label"]).size().unstack(fill_value=0).to_csv(
        os.path.join(OUT, "label_by_position.csv"))

    # per-query features
    qfeat = test.set_index("query_id").copy()
    qfeat["n_words"] = qfeat["query"].str.split().str.len()
    qfeat["oov_share"] = [np.mean([t not in bm25.idf for t in preprocess_text(q)] or [1.0]) for q in qfeat["query"]]
    qfeat["has_digit"] = qfeat["query"].str.contains(r"\d")
    qfeat.to_csv(os.path.join(OUT, "test_query_features.csv"))

    # where dense and BM25 disagree most (retrieval nDCG)
    diff = (rt_q["Sentence Transformer"]["nDCG@10"] - rt_q["BM25"]["nDCG@10"]).rename("dense_minus_bm25")
    ex = qfeat.join(diff).join(rt_q["Hybrid (RRF)"]["nDCG@10"].rename("hybrid"))
    ex = ex.join(rt_q["BM25"]["nDCG@10"].rename("bm25")).join(rt_q["Sentence Transformer"]["nDCG@10"].rename("dense"))
    ex.sort_values("dense_minus_bm25").to_csv(os.path.join(OUT, "bm25_vs_dense_examples.csv"))

    # term statistics for the index section
    df_counts = pd.Series(bm25.index.df).sort_values(ascending=False)
    df_counts.to_csv(os.path.join(OUT, "term_document_frequency.csv"), header=["df"])
    tok_lens = pd.Series([len(preprocess_text(t)) for t in corpus])

    meta = {
        "judgments": len(df), "queries": len(queries), "dev_queries": len(dev), "test_queries": len(test),
        "products": len(catalog), "label_counts": df.esci_label.value_counts().to_dict(),
        "index": index_stats, "title_tokens_mean": float(tok_lens.mean()),
        "fit_seconds": {"BM25": t_bm25, "TF-IDF": t_tfidf, "Sentence Transformer (cached)": t_dense},
        "latency_ms_per_query": latency,
        "chosen": {"bm25_k1": bm25.k1, "bm25_b": bm25.b, "tfidf_variant": best_tf,
                   "tfidf_sublinear": tfidf.sublinear_tf, "tfidf_idf": tfidf.use_idf, "rrf_k": best_k},
        "runtime_s": time.time() - t_start,
    }
    with open(os.path.join(OUT, "run_summary.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)

    print("\nRe-ranking (test, %d queries)" % len(test))
    print(tabulate(rerank_table.round(4), headers="keys", tablefmt="github"))
    print("\nFull-catalog retrieval (test)")
    print(tabulate(retrieval_table.round(4), headers="keys", tablefmt="github"))
    print("\nms / query:", {k: round(v, 2) for k, v in latency.items()})
    print(f"done in {meta['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
