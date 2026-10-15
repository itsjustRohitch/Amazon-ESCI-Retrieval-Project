"""
src/models/tfidf_ranker.py
--------------------------
Model 1: TF-IDF Classical Vector Space Model for E-Commerce Product Retrieval.

Weights terms using Term Frequency - Inverse Document Frequency (TF-IDF)
and computes ranking scores using Cosine Similarity against the query vector.
Integrates directly with our scratch-built text preprocessing pipeline.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text


class TFIDFRanker:
    """
    Classical TF-IDF Vector Space Model retriever.
    """

    def __init__(self, apply_stemming: bool = True):
        self.apply_stemming = apply_stemming
        # Use our scratch preprocessing as the tokenizer / preprocessor
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: preprocess_text(text, apply_stemming=self.apply_stemming, return_tokens=True),
            preprocessor=None,
            token_pattern=None,
            norm="l2",
            sublinear_tf=True
        )
        self.doc_vectors = None
        self.product_ids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, corpus: List[str], product_ids: List[str]) -> "TFIDFRanker":
        """
        Fit TF-IDF vocabulary on the product catalog and build document index matrix.

        Args:
            corpus (List[str]): List of raw product titles or catalog descriptions.
            product_ids (List[str]): Corresponding product IDs.
        """
        assert len(corpus) == len(product_ids), "Corpus and product_ids must have identical length."
        self.product_ids = list(product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # Fit and transform corpus into sparse TF-IDF matrix
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True
        return self

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-K products ranked by TF-IDF cosine similarity.

        Args:
            query (str): Raw user search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[str, float]]: List of (product_id, similarity_score) sorted descending.
        """
        if not self.is_fitted:
            raise RuntimeError("TFIDFRanker must be fitted before ranking.")

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.product_ids[idx], float(scores[idx])) for idx in top_indices]
        return results

    def score_candidates(self, query: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Score and rank a specific subset of candidate products for a given query
        (used for Amazon ESCI benchmark query-level candidate evaluations).

        Args:
            query (str): User query.
            candidate_ids (List[str]): List of candidate product IDs to evaluate.

        Returns:
            List[Tuple[str, float]]: Candidates ranked by score descending.
        """
        if not self.is_fitted:
            raise RuntimeError("TFIDFRanker must be fitted before scoring.")

        q_vec = self.vectorizer.transform([query])
        valid_indices = [self.pid_to_idx[pid] for pid in candidate_ids if pid in self.pid_to_idx]
        valid_pids = [pid for pid in candidate_ids if pid in self.pid_to_idx]

        if not valid_indices:
            return [(pid, 0.0) for pid in candidate_ids]

        sub_matrix = self.doc_vectors[valid_indices]
        scores = cosine_similarity(q_vec, sub_matrix).flatten()

        scored = list(zip(valid_pids, scores.tolist()))
        # Include missing candidate IDs with score 0
        missing = [(pid, 0.0) for pid in candidate_ids if pid not in self.pid_to_idx]
        all_scored = scored + missing
        all_scored.sort(key=lambda x: x[1], reverse=True)
        return all_scored


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

    ranker = TFIDFRanker()
    ranker.fit(catalog, pids)

    test_q = "iphone 13 pro"
    top_results = ranker.rank(test_q, top_k=3)
    print(f"Query: '{test_q}'")
    for rank, (pid, score) in enumerate(top_results, 1):
        print(f"  Rank {rank}: Product {pid} | Cosine Sim: {score:.4f} | Title: {catalog[pids.index(pid)]}")
