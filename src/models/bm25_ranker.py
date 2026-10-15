"""
src/models/bm25_ranker.py
-------------------------
Model 2: BM25 (Best Matching 25) Probabilistic Ranking Framework.

Implements Okapi BM25 with term frequency saturation (k1) and document length normalization (b).
Integrates seamlessly with our scratch-built text preprocessing pipeline.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text


class BM25Ranker:
    """
    Probabilistic Okapi BM25 retrieval model for e-commerce search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, apply_stemming: bool = True):
        self.k1 = k1
        self.b = b
        self.apply_stemming = apply_stemming
        self.bm25: Optional[BM25Okapi] = None
        self.product_ids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, corpus: List[str], product_ids: List[str]) -> "BM25Ranker":
        """
        Tokenize the product catalog using our preprocessing pipeline and build the BM25 index.

        Args:
            corpus (List[str]): List of product titles or catalog descriptions.
            product_ids (List[str]): Matching product IDs.
        """
        assert len(corpus) == len(product_ids), "Corpus and product_ids must have identical length."
        self.product_ids = list(product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # Preprocess each catalog item into token lists
        tokenized_corpus = [
            preprocess_text(text, apply_stemming=self.apply_stemming, return_tokens=True)
            for text in corpus
        ]

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.is_fitted = True
        return self

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-K products ranked by BM25 probabilistic score.

        Args:
            query (str): Raw user search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[str, float]]: List of (product_id, bm25_score) sorted descending.
        """
        if not self.is_fitted or self.bm25 is None:
            raise RuntimeError("BM25Ranker must be fitted before ranking.")

        q_tokens = preprocess_text(query, apply_stemming=self.apply_stemming, return_tokens=True)
        if not q_tokens:
            return [(pid, 0.0) for pid in self.product_ids[:top_k]]

        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.product_ids[idx], float(scores[idx])) for idx in top_indices]
        return results

    def score_candidates(self, query: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Score a specific candidate subset of products for query-level benchmark evaluation.

        Args:
            query (str): User query.
            candidate_ids (List[str]): List of candidate product IDs to rank.

        Returns:
            List[Tuple[str, float]]: Ranked candidate pairs (product_id, score).
        """
        if not self.is_fitted or self.bm25 is None:
            raise RuntimeError("BM25Ranker must be fitted before scoring.")

        q_tokens = preprocess_text(query, apply_stemming=self.apply_stemming, return_tokens=True)
        if not q_tokens:
            return [(pid, 0.0) for pid in candidate_ids]

        all_scores = self.bm25.get_scores(q_tokens)
        results = []
        for pid in candidate_ids:
            if pid in self.pid_to_idx:
                idx = self.pid_to_idx[pid]
                results.append((pid, float(all_scores[idx])))
            else:
                results.append((pid, 0.0))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


if __name__ == "__main__":
    catalog = [
        "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
        "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
        "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
        "Anker USB-C to Lightning Fast Charging Cable for iPhone",
        "Sony WH-1000XM4 Wireless Noise-Canceling Overhead Headphones",
        "Men's Breathable Mesh Running Shoes Non-Slip Athletic Sneakers"
    ]
    pids = ["B001", "B002", "B003", "B004", "B005", "B006"]

    ranker = BM25Ranker()
    ranker.fit(catalog, pids)

    test_q = "iphone 13 pro"
    top_results = ranker.rank(test_q, top_k=3)
    print(f"BM25 Query: '{test_q}'")
    for rank, (pid, score) in enumerate(top_results, 1):
        print(f"  Rank {rank}: Product {pid} | BM25 Score: {score:.4f} | Title: {catalog[pids.index(pid)]}")
