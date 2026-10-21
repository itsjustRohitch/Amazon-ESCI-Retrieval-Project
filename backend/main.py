"""
backend/main.py
---------------
FastAPI server for the search interface.

Models are built once at startup with the settings chosen on the dev queries
(read from evaluation_results/run_summary.json if it exists). If a query typed
in the UI is one of the benchmark queries, the response also carries the ESCI
judgments for it, so the interface can mark each result E / S / C / I and show
nDCG@10 for the list.

Run:  python -m uvicorn backend.main:app --port 8000
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation import DEFAULT_ESCI_GAINS, compute_ndcg_at_k
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion
from src.models.tfidf_ranker import TFIDFRanker
from src.preprocessing import NLTK_STOPWORDS, clean_text, stem_tokens, tokenize

state: Dict[str, Any] = {}
MODEL_KEYS = ["tfidf", "bm25", "dense", "hybrid"]
MODEL_NAMES = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Sentence Transformer", "hybrid": "Hybrid (RRF)"}
EVAL_DIR = os.path.join(ROOT, "evaluation_results")


def _read_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    catalog = pd.read_parquet(os.path.join(ROOT, "data", "catalog_cache.parquet"))
    emb = np.load(os.path.join(ROOT, "data", "embeddings_cache.npy"))
    if "text" not in catalog or len(catalog) != len(emb):
        raise RuntimeError("catalog cache is out of date, run `python run_evaluation.py` once first")

    summary = _read_json(os.path.join(EVAL_DIR, "run_summary.json"), {})
    chosen = summary.get("chosen", {})
    corpus, pids = catalog.text.tolist(), catalog.product_id.tolist()

    tfidf = TFIDFRanker(sublinear_tf=chosen.get("tfidf_sublinear", True),
                        use_idf=chosen.get("tfidf_idf", True)).fit(corpus, pids)
    bm25 = BM25Ranker(k1=chosen.get("bm25_k1", 1.2), b=chosen.get("bm25_b", 0.75)).fit(corpus, pids)
    dense = DenseRanker().load_cached_embeddings(emb, pids)
    rrf_k = int(chosen.get("rrf_k", 60))

    bench = pd.read_parquet(os.path.join(ROOT, "data", "benchmark_sample.parquet"))
    judged: Dict[str, Dict[str, str]] = {}
    for q, pid, lab in bench[["query", "product_id", "esci_label"]].itertuples(index=False):
        judged.setdefault(q.strip().lower(), {})[pid] = lab

    state.update(
        tfidf=tfidf, bm25=bm25, dense=dense,
        hybrid=ReciprocalRankFusion([bm25, dense], k=rrf_k), rrf_k=rrf_k,
        info={p: (t, b if isinstance(b, str) else "") for p, t, b in
              catalog[["product_id", "product_title", "product_brand"]].itertuples(index=False)},
        judged=judged, queries=sorted(judged), summary=summary,
    )
    print(f"ready: {len(pids):,} products, {len(judged):,} judged queries, {time.time() - t0:.1f}s")
    yield
    state.clear()


app = FastAPI(title="ESCI product search", version="2.0", lifespan=lifespan)


def preprocessing_trace(q: str) -> Dict[str, Any]:
    cleaned = clean_text(q)
    tokens = tokenize(cleaned)
    kept = [t for t in tokens if t not in NLTK_STOPWORDS and len(t) >= 2]
    stems = stem_tokens(kept)
    bm25: BM25Ranker = state["bm25"]
    terms = []
    for s in dict.fromkeys(stems):
        df = bm25.index.df.get(s, 0)
        terms.append({"term": s, "df": df, "idf": round(bm25.idf.get(s, 0.0), 3)})
    return {"raw": q, "cleaned": cleaned,
            "tokens": [{"text": t, "removed": t not in kept} for t in tokens],
            "stems": stems, "terms": terms}


def _rank(model: str, q: str, k: int):
    """Returns ranked (pid, score) pairs and, per pid, the explanation shown in the UI."""
    if model in ("tfidf", "bm25"):
        r = state[model]
        pairs = r.rank(q, top_k=k)
        return pairs, {pid: {"terms": {t: round(v, 4) for t, v in r.explain(q, pid).items()}}
                       for pid, s in pairs if s > 0}
    if model == "dense":
        return state["dense"].rank(q, top_k=k), {}
    # same candidate depth as ReciprocalRankFusion.rank, so scores match the evaluated model
    fetch = max(k * 3, 50)
    lists = [state["bm25"].rank(q, top_k=fetch), state["dense"].rank(q, top_k=fetch)]
    pos = [{pid: i + 1 for i, (pid, _) in enumerate(lst)} for lst in lists]
    fused = ReciprocalRankFusion.fuse_rank_lists(lists, k=state["rrf_k"])[:k]
    return fused, {pid: {"bm25_rank": pos[0].get(pid), "dense_rank": pos[1].get(pid)} for pid, _ in fused}


def _results(model: str, q: str, k: int) -> Dict[str, Any]:
    labels = state["judged"].get(q.strip().lower())
    t0 = time.perf_counter()
    pairs, why = _rank(model, q, max(k, 10))
    ms = (time.perf_counter() - t0) * 1000
    ndcg = None
    if labels:
        gains = {p: DEFAULT_ESCI_GAINS[l] for p, l in labels.items()}
        ndcg = round(compute_ndcg_at_k([p for p, _ in pairs], gains, 10), 4)
    items = []
    for i, (pid, score) in enumerate(pairs[:k], 1):
        title, brand = state["info"].get(pid, ("", ""))
        items.append({"rank": i, "product_id": pid, "title": title, "brand": brand,
                      "score": round(float(score), 5),
                      "label": labels.get(pid) if labels else None, "why": why.get(pid)})
    return {"model": model, "name": MODEL_NAMES[model], "latency_ms": round(ms, 2),
            "ndcg10": ndcg, "judged": len(labels) if labels else 0, "results": items}


def _check(q: str, model: Optional[str] = None):
    if not q.strip():
        raise HTTPException(400, "empty query")
    if model is not None and model not in MODEL_KEYS:
        raise HTTPException(400, f"model must be one of {MODEL_KEYS}")


@app.get("/api/search")
def search(q: str = Query(...), model: str = "bm25", k: int = Query(10, ge=1, le=50)):
    _check(q, model)
    out = _results(model, q, k)
    out.update(query=q, preprocessing=preprocessing_trace(q), params=_params())
    return out


@app.get("/api/compare")
def compare(q: str = Query(...), k: int = Query(10, ge=1, le=20)):
    _check(q)
    return {"query": q, "preprocessing": preprocessing_trace(q),
            "runs": [_results(m, q, k) for m in MODEL_KEYS]}


@app.get("/api/queries")
def sample_queries(n: int = Query(8, ge=1, le=50), seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    qs = state["queries"]
    return {"queries": [qs[i] for i in rng.choice(len(qs), size=min(n, len(qs)), replace=False)]}


def _params():
    b = state["bm25"]
    t = state["tfidf"]
    return {"bm25": {"k1": b.k1, "b": b.b},
            "tfidf": {"sublinear_tf": t.sublinear_tf, "idf": t.use_idf},
            "rrf": {"k": state["rrf_k"], "fused": ["BM25", "Sentence Transformer"]},
            "dense": {"model": state["dense"].model_name}}


def _csv(name, **kw):
    path = os.path.join(EVAL_DIR, name)
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


@app.get("/api/evaluation")
def evaluation():
    out: Dict[str, Any] = {"summary": state["summary"], "params": _params()}
    for key, name in [("rerank", "metrics_rerank.csv"), ("retrieval", "metrics_retrieval.csv")]:
        df = _csv(name)
        out[key] = df.round(4).to_dict(orient="records") if df is not None else []
    for key in ["rerank", "retrieval"]:
        df = _csv(f"pr_curve_{key}.csv", index_col=0)
        out[f"pr_{key}"] = {c: df[c].round(4).tolist() for c in df.columns} if df is not None else {}
    grid = _csv("dev_bm25_grid.csv")
    out["bm25_grid"] = grid.round(4).to_dict(orient="records") if grid is not None else []
    return out


@app.get("/api/index")
def index_stats():
    b: BM25Ranker = state["bm25"]
    top = sorted(b.index.df.items(), key=lambda x: -x[1])[:15]
    return {"stats": b.index.stats(),
            "top_terms": [{"term": t, "df": d, "idf": round(b.idf[t], 3)} for t, d in top],
            "dataset": {k: state["summary"].get(k) for k in
                        ["judgments", "queries", "dev_queries", "test_queries", "products", "label_counts"]}}


@app.get("/health")
def health():
    return {"status": "ok", "products": len(state.get("info", {}))}


FRONTEND = os.path.join(ROOT, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(FRONTEND, "index.html"))
