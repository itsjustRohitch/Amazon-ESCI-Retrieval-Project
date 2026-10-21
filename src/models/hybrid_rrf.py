"""
src/models/hybrid_rrf.py
------------------------
Model 4: Hybrid Reciprocal Rank Fusion (RRF) Ranking System.

Combines sparse retrieval (BM25 / TF-IDF) and dense semantic retrieval (Sentence Transformer)
using rank-based reciprocal aggregation:

    RRF_Score(d) = sum_{m in M} [ w_m / (k + rank_m(d)) ]

Where:
- rank_m(d) is the 1-indexed rank of document d in model m
- k is the rank discount constant (standard default: k = 60)
- w_m is an optional model importance weight (default: 1.0)

RRF solves the score-calibration problem by operating strictly on relative ranks rather than
heterogeneous raw score distributions.
"""

from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) hybrid combiner.
    """

    def __init__(
        self,
        rankers: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        k: int = 60
    ):
        """
        Initialize RRF hybrid combiner.

        Args:
            rankers (List[Any], optional): List of ranker instances (e.g. [bm25_ranker, dense_ranker]).
            weights (List[float], optional): Weight for each ranker (defaults to 1.0 each).
            k (int): RRF constant (default: 60).
        """
        self.rankers = rankers or []
        self.k = k
        self.weights = weights or ([1.0] * len(self.rankers))
        if len(self.weights) != len(self.rankers):
            raise ValueError("Length of weights must match length of rankers.")

    @staticmethod
    def fuse_rank_lists(
        ranked_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Pure RRF fusion function on pre-computed ranked lists.

        Args:
            ranked_lists: List of rankings, each ranking is List[(product_id, original_score)].
            weights: List of float weights for each ranking list.
            k: RRF smoothing constant.

        Returns:
            List[Tuple[str, float]]: Fused rankings (product_id, rrf_score) sorted descending.
        """
        if not ranked_lists:
            return []

        w = weights if weights is not None else [1.0] * len(ranked_lists)
        rrf_scores: Dict[str, float] = defaultdict(float)

        for model_idx, r_list in enumerate(ranked_lists):
            model_weight = w[model_idx]
            for rank_0, (pid, _) in enumerate(r_list):
                rank = rank_0 + 1  # 1-indexed rank
                rrf_scores[pid] += model_weight / (k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform hybrid search across all registered rankers and fuse top results.

        Args:
            query (str): User query.
            top_k (int): Number of top fused results.

        Returns:
            List[Tuple[str, float]]: Top fused (product_id, rrf_score).
        """
        if not self.rankers:
            raise RuntimeError("No rankers provided to ReciprocalRankFusion.")

        # Gather rankings from each sub-ranker (fetch more candidates than needed before fusing)
        fetch_k = max(top_k * 3, 50)
        ranked_lists = [ranker.rank(query, top_k=fetch_k) for ranker in self.rankers]

        fused = self.fuse_rank_lists(ranked_lists, weights=self.weights, k=self.k)
        return fused[:top_k]

    def score_candidates(self, query: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Score and rank a candidate subset using RRF for query-level benchmark evaluation.

        Args:
            query (str): User query.
            candidate_ids (List[str]): List of candidate product IDs.

        Returns:
            List[Tuple[str, float]]: Fused ranked candidates.
        """
        if not self.rankers:
            raise RuntimeError("No rankers provided to ReciprocalRankFusion.")

        ranked_lists = [
            ranker.score_candidates(query, candidate_ids) for ranker in self.rankers
        ]

        fused = self.fuse_rank_lists(ranked_lists, weights=self.weights, k=self.k)

        # Ensure any missing candidate is appended with score 0
        fused_pids = {pid for pid, _ in fused}
        for pid in candidate_ids:
            if pid not in fused_pids:
                fused.append((pid, 0.0))

        return fused


if __name__ == "__main__":
    from src.models.bm25_ranker import BM25Ranker
    from src.models.dense_ranker import DenseRanker

    catalog = [
        "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
        "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
        "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
        "Anker USB-C to Lightning Fast Charging Cable for iPhone",
        "Sony WH-1000XM4 Wireless Noise-Canceling Overhead Headphones",
        "Men's Breathable Mesh Running Shoes Non-Slip Athletic Sneakers"
    ]
    pids = ["B001", "B002", "B003", "B004", "B005", "B006"]

    print("Initializing BM25 and Dense rankers...")
    bm25 = BM25Ranker().fit(catalog, pids)
    dense = DenseRanker().fit(catalog, pids, show_progress_bar=False)

    hybrid = ReciprocalRankFusion(rankers=[bm25, dense], weights=[1.0, 1.0], k=60)

    test_q = "apple cell phone 128gb"
    top_results = hybrid.rank(test_q, top_k=3)

    print(f"\nHybrid RRF Query: '{test_q}'")
    for rank, (pid, score) in enumerate(top_results, 1):
        print(f"  Rank {rank}: Product {pid} | RRF Score: {score:.6f} | Title: {catalog[pids.index(pid)]}")
