"""
tests/test_models.py
--------------------
Automated test suite for all 4 ranking models:
1. TF-IDF Ranker
2. BM25 Ranker
3. Dense Ranker (Sentence Transformers)
4. Hybrid RRF Ranker
"""

import unittest
from src.models.tfidf_ranker import TFIDFRanker
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion


class TestRankingModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalog = [
            "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
            "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
            "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
            "Anker USB-C to Lightning Fast Charging Cable for iPhone",
            "Sony WH-1000XM4 Wireless Noise-Canceling Overhead Headphones",
            "Men's Breathable Mesh Running Shoes Non-Slip Athletic Sneakers"
        ]
        cls.pids = ["B001", "B002", "B003", "B004", "B005", "B006"]

        cls.tfidf = TFIDFRanker().fit(cls.catalog, cls.pids)
        cls.bm25 = BM25Ranker().fit(cls.catalog, cls.pids)
        cls.dense = DenseRanker().fit(cls.catalog, cls.pids, show_progress_bar=False)
        cls.hybrid = ReciprocalRankFusion(rankers=[cls.bm25, cls.dense], k=60)

    def test_tfidf_exact_keyword(self):
        results = self.tfidf.rank("iphone 13 pro", top_k=2)
        top_pid, top_score = results[0]
        self.assertEqual(top_pid, "B001")
        self.assertGreater(top_score, 0.0)

    def test_bm25_exact_keyword(self):
        results = self.bm25.rank("wireless headphones", top_k=2)
        top_pid, top_score = results[0]
        self.assertEqual(top_pid, "B005")  # Sony WH-1000XM4 Wireless Headphones
        self.assertGreater(top_score, 0.0)

    def test_dense_semantic_matching(self):
        # Query uses synonyms: 'jogging sneakers' -> matches 'running shoes athletic sneakers'
        results = self.dense.rank("jogging sneakers", top_k=2)
        top_pid, top_score = results[0]
        self.assertEqual(top_pid, "B006")  # Running Shoes Athletic Sneakers
        self.assertGreater(top_score, 0.3)

    def test_hybrid_rrf_combines_strengths(self):
        results = self.hybrid.rank("apple mobile phone 128gb", top_k=3)
        top_pid, top_score = results[0]
        # Should rank iPhone 13 Pro (B001) as #1
        self.assertEqual(top_pid, "B001")
        self.assertGreater(top_score, 0.0)

    def test_candidate_scoring(self):
        candidates = ["B001", "B002", "B006"]
        scored = self.bm25.score_candidates("samsung galaxy smartphone", candidates)
        self.assertEqual(len(scored), len(candidates))
        # Top candidate must be B002 (Samsung Galaxy S22)
        self.assertEqual(scored[0][0], "B002")


if __name__ == "__main__":
    unittest.main()
