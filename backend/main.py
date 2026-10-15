"""
backend/main.py
---------------
High-Performance FastAPI REST Server for Project 14: Product Search using Amazon ESCI.

Features:
- Models loaded ONCE into memory at startup (Zero per-request reload overhead).
- Complete decoupling of backend and frontend via JSON REST API contracts.
- Fully documented OpenAPI/Swagger endpoints at `/docs`.
- CORS-enabled for arbitrary frontend development (React, Next.js, Vue, or Vanilla JS).
- Serves the integrated lightweight frontend at `/`.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, tokenize, remove_stopwords, stem_tokens
from src.models.tfidf_ranker import TFIDFRanker
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion


# Global state store for models & catalog
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load data and initialize all 4 ranking models once on startup.
    """
    print("[FastAPI Lifespan] Loading Amazon ESCI product catalog and models...")
    t0 = time.time()

    catalog_cache_path = os.path.join(PROJECT_ROOT, "data", "catalog_cache.parquet")
    emb_cache_path = os.path.join(PROJECT_ROOT, "data", "embeddings_cache.npy")

    if os.path.exists(catalog_cache_path) and os.path.exists(emb_cache_path):
        catalog_df = pd.read_parquet(catalog_cache_path)
        cached_embeddings = np.load(emb_cache_path)
    else:
        benchmark_path = os.path.join(PROJECT_ROOT, "data", "benchmark_sample.parquet")
        df_data = pd.read_parquet(benchmark_path)
        catalog_df = df_data[["product_id", "product_title", "product_brand"]].drop_duplicates(subset=["product_id"]).head(5000).copy()
        catalog_df["product_title"] = catalog_df["product_title"].fillna("Untitled Product")
        catalog_df["product_brand"] = catalog_df["product_brand"].fillna("Generic")
        cached_embeddings = None

    corpus = catalog_df["product_title"].tolist()
    product_ids = catalog_df["product_id"].tolist()

    id_to_info = {
        row["product_id"]: {
            "title": row["product_title"],
            "brand": row["product_brand"]
        }
        for _, row in catalog_df.iterrows()
    }

    # Initialize all 4 models
    tfidf = TFIDFRanker().fit(corpus, product_ids)
    bm25 = BM25Ranker(k1=1.5, b=0.75).fit(corpus, product_ids)
    dense = DenseRanker()
    if cached_embeddings is not None:
        dense.load_cached_embeddings(cached_embeddings, product_ids)
    else:
        dense.fit(corpus, product_ids, batch_size=128, show_progress_bar=False)

    hybrid = ReciprocalRankFusion(rankers=[bm25, dense], weights=[1.0, 1.0], k=60)

    state["tfidf"] = tfidf
    state["bm25"] = bm25
    state["dense"] = dense
    state["hybrid"] = hybrid
    state["id_to_info"] = id_to_info
    state["catalog_size"] = len(corpus)

    print(f"[FastAPI Lifespan] Search engine ready with {len(corpus):,} products in {time.time() - t0:.2f}s!")
    yield
    print("[FastAPI Lifespan] Shutting down search engine...")
    state.clear()


app = FastAPI(
    title="Amazon ESCI Product Search API",
    description="REST API for multi-paradigm e-commerce Information Retrieval (TF-IDF, BM25, Sentence Transformers, and Hybrid RRF).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for arbitrary client frameworks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Pydantic Models
class PreprocessingDetails(BaseModel):
    cleaned: str
    tokenized: List[str]
    stopwords_removed: List[str]
    stemmed: List[str]


class SearchResultItem(BaseModel):
    rank: int
    product_id: str
    title: str
    brand: str
    score: float


class SearchResponse(BaseModel):
    query: str
    model: str
    latency_ms: float
    total_results: int
    preprocessing: PreprocessingDetails
    results: List[SearchResultItem]


class CompareResponse(BaseModel):
    query: str
    preprocessing: PreprocessingDetails
    latency_ms: float
    rankings: Dict[str, List[SearchResultItem]]


@app.get("/health", tags=["Health"])
def health_check():
    """Service health and catalog status."""
    return {
        "status": "healthy",
        "catalog_size": state.get("catalog_size", 0),
        "available_models": ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
    }


@app.get("/api/search", response_model=SearchResponse, tags=["Search"])
def search_products(
    q: str = Query(..., description="User search query string"),
    model: str = Query("hybrid", description="Ranking model: 'hybrid', 'bm25', 'dense', 'tfidf'"),
    k: int = Query(10, ge=1, le=50, description="Top K results cutoff")
):
    """
    Search product catalog using the selected ranking paradigm.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Scratch Preprocessing details
    clean_q = clean_text(q)
    toks = tokenize(clean_q)
    no_stop = remove_stopwords(toks)
    stemmed = stem_tokens(no_stop)

    prep = PreprocessingDetails(
        cleaned=clean_q,
        tokenized=toks,
        stopwords_removed=no_stop,
        stemmed=stemmed
    )

    model_key = model.lower().strip()
    if model_key in ["hybrid", "rrf"]:
        ranker = state["hybrid"]
        model_name = "Hybrid RRF (BM25 + Sentence Transformer)"
    elif model_key in ["bm25", "okapi"]:
        ranker = state["bm25"]
        model_name = "BM25 Probabilistic Retrieval"
    elif model_key in ["dense", "transformer", "minilm"]:
        ranker = state["dense"]
        model_name = "Sentence Transformer (all-MiniLM-L6-v2)"
    elif model_key in ["tfidf", "tf-idf"]:
        ranker = state["tfidf"]
        model_name = "TF-IDF Vector Space Model"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: '{model}'. Choose from: hybrid, bm25, dense, tfidf")

    t0 = time.time()
    ranked_pairs = ranker.rank(q, top_k=k)
    elapsed_ms = (time.time() - t0) * 1000

    id_to_info = state["id_to_info"]
    results = []
    for rank, (pid, score) in enumerate(ranked_pairs, 1):
        info = id_to_info.get(pid, {"title": "Unknown Title", "brand": "Generic"})
        results.append(SearchResultItem(
            rank=rank,
            product_id=pid,
            title=info["title"],
            brand=info["brand"],
            score=round(score, 5)
        ))

    return SearchResponse(
        query=q,
        model=model_name,
        latency_ms=round(elapsed_ms, 2),
        total_results=len(results),
        preprocessing=prep,
        results=results
    )


@app.get("/api/compare", response_model=CompareResponse, tags=["Search"])
def compare_models(
    q: str = Query(..., description="User search query string"),
    k: int = Query(5, ge=1, le=20, description="Top K results cutoff per model")
):
    """
    Run the same query concurrently across all 4 models and return side-by-side rankings.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    clean_q = clean_text(q)
    toks = tokenize(clean_q)
    no_stop = remove_stopwords(toks)
    stemmed = stem_tokens(no_stop)

    prep = PreprocessingDetails(
        cleaned=clean_q,
        tokenized=toks,
        stopwords_removed=no_stop,
        stemmed=stemmed
    )

    id_to_info = state["id_to_info"]
    models_dict = {
        "TF-IDF": state["tfidf"],
        "BM25": state["bm25"],
        "Sentence Transformer": state["dense"],
        "Hybrid (RRF)": state["hybrid"]
    }

    t0 = time.time()
    rankings_output = {}

    for name, ranker in models_dict.items():
        pairs = ranker.rank(q, top_k=k)
        item_list = []
        for rank, (pid, score) in enumerate(pairs, 1):
            info = id_to_info.get(pid, {"title": "Unknown Title", "brand": "Generic"})
            item_list.append(SearchResultItem(
                rank=rank,
                product_id=pid,
                title=info["title"],
                brand=info["brand"],
                score=round(score, 5)
            ))
        rankings_output[name] = item_list

    elapsed_ms = (time.time() - t0) * 1000

    return CompareResponse(
        query=q,
        preprocessing=prep,
        latency_ms=round(elapsed_ms, 2),
        rankings=rankings_output
    )


@app.get("/api/metrics", tags=["Evaluation"])
def get_evaluation_metrics():
    """
    Return the pre-computed standardized evaluation comparison table.
    """
    csv_path = os.path.join(PROJECT_ROOT, "evaluation_results", "model_comparison_table.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Evaluation metrics table not found. Run run_evaluation.py first.")

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return {
        "status": "success",
        "benchmark": "Amazon ESCI Task 1 English",
        "metrics_table": records
    }


@app.get("/api/eda", tags=["Evaluation"])
def get_eda_summary():
    """
    Return high-level summary statistics from the raw Amazon ESCI dataset.
    """
    return {
        "raw_total_judgments": 2621288,
        "english_task1_judgments": 601354,
        "unique_queries": 29844,
        "unique_products": 482105,
        "avg_depth_per_query": 20.15,
        "class_distribution": {
            "Exact (E)": {"count": 261527, "percentage": 43.49},
            "Substitute (S)": {"count": 211191, "percentage": 35.12},
            "Irrelevant (I)": {"count": 101447, "percentage": 16.87},
            "Complement (C)": {"count": 27189, "percentage": 4.52}
        },
        "query_word_counts": {
            "median": 3.0,
            "mean": 3.22,
            "pct_90": 5.0
        }
    }


# Mount evaluation results static files (for curves image)
eval_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
if os.path.exists(eval_dir):
    app.mount("/evaluation_results", StaticFiles(directory=eval_dir), name="evaluation_results")

# Mount frontend static files
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend not found, access API at /docs"}
