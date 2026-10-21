"""
src/models/dense_ranker.py
--------------------------
Model 3: Dense Semantic Vector Retriever using Sentence Transformers.

Encodes product titles and user queries into dense 384-dimensional semantic embedding vectors
using 'sentence-transformers/all-MiniLM-L6-v2'. Computes semantic relevance using Cosine Similarity.
Bridges the vocabulary mismatch problem where queries and products use different words for the same concept.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.preprocessing import clean_text


class DenseRanker:
    """
    Dense Semantic Vector Retriever powered by Sentence Transformers.

    Uses all-MiniLM-L6-v2 embeddings (384 dimensions).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.doc_embeddings: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.product_ids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}

    def fit(
        self,
        corpus: List[str],
        product_ids: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = True
    ) -> "DenseRanker":
        """
        Encode the product catalog into dense semantic embeddings.

        Args:
            corpus (List[str]): List of product titles or catalog descriptions.
            product_ids (List[str]): Corresponding product IDs.
            batch_size (int): Encoding batch size.
            show_progress_bar (bool): Whether to display progress during embedding.
        """
        assert len(corpus) == len(product_ids), "Corpus and product_ids must have the same length."
        self.product_ids = list(product_ids)
        self.pid_to_idx = {
            pid: i for i, 
            pid in enumerate(self.product_ids)
        }

        # Clean text for transformer encoding (natural word order, no stemming)
        cleaned_corpus: List[str] = [clean_text(doc) for doc in corpus]

        # Compute normalized embeddings so dot-product equals cosine similarity
        self.doc_embeddings = self.model.encode(
            cleaned_corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        self.is_fitted = True
        return self

    def load_cached_embeddings(self, embeddings: np.ndarray, product_ids: List[str]) -> "DenseRanker":
        """
        Instantly load precomputed embedding matrix from disk cache.
        """
        assert len(embeddings) == len(product_ids), "Embeddings and product_ids length mismatch."
        self.product_ids = list(product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.doc_embeddings = embeddings
        self.is_fitted = True
        return self

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-K products ranked by semantic cosine similarity.

        Args:
            query (str): Raw user search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[str, float]]: List of (product_id, cosine_similarity) sorted descending.
        """
        if self.doc_embeddings is None or not self.is_fitted:
            raise RuntimeError("DenseRanker must be fitted before ranking.")

        cleaned_q = clean_text(query)
        q_emb = self.model.encode(
            [cleaned_q],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # Since embeddings are L2-normalized, dot product equals cosine similarity
        scores = np.dot(self.doc_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.product_ids[idx], float(scores[idx])) for idx in top_indices]
        return results

    def score_candidates(self, query: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Score and rank a candidate subset of products for query-level evaluation.

        Args:
            query (str): User query.
            candidate_ids (List[str]): Candidate product IDs.

        Returns:
            List[Tuple[str, float]]: Ranked candidate pairs (product_id, score).
        """
        if not self.is_fitted or self.doc_embeddings is None:
            raise RuntimeError("DenseRanker must be fitted before scoring.")

        cleaned_q = clean_text(query)
        q_emb = self.model.encode(
            [cleaned_q],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        results = []
        for pid in candidate_ids:
            if pid in self.pid_to_idx:
                idx = self.pid_to_idx[pid]
                sim = float(np.dot(self.doc_embeddings[idx], q_emb))
                results.append((pid, sim))
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

    ranker = DenseRanker()
    ranker.fit(catalog, pids, show_progress_bar=False)

    test_q = "smart cell phones"
    top_results = ranker.rank(test_q, top_k=3)
    print(f"Dense Query: '{test_q}' (Testing semantic match without keyword overlap)")
    for rank, (pid, score) in enumerate(top_results, 1):
        print(f"  Rank {rank}: Product {pid} | Cosine Sim: {score:.4f} | Title: {catalog[pids.index(pid)]}")
