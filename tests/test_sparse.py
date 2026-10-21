"""
tests/test_sparse.py
--------------------
Checks for the inverted index, TF-IDF and BM25. These do not need the
sentence transformer, so they run in a couple of seconds.
"""

import math
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.indexing import InvertedIndex
from src.models.bm25_ranker import BM25Ranker
from src.models.tfidf_ranker import TFIDFRanker
from src.preprocessing import preprocess_text

CATALOG = [
    "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
    "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
    "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
    "Anker USB-C to Lightning Fast Charging Cable for iPhone",
    "Sony WH-1000XM4 Wireless Noise-Canceling Overhead Headphones",
    "Men's Breathable Mesh Running Shoes Non-Slip Athletic Sneakers",
    "iPhone iPhone iPhone charging dock",
]
PIDS = [f"B{i:03d}" for i in range(1, len(CATALOG) + 1)]


class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        self.idx = InvertedIndex().build(CATALOG)

    def test_document_frequency(self):
        # 'iphon' (stemmed) occurs in docs 0, 2, 3 and 6
        ids, tfs = self.idx.get("iphon")
        self.assertEqual(ids.tolist(), [0, 2, 3, 6])
        self.assertEqual(tfs.tolist(), [1, 1, 1, 3])
        self.assertEqual(self.idx.df["iphon"], 4)

    def test_lengths(self):
        self.assertEqual(self.idx.num_docs, len(CATALOG))
        self.assertEqual(self.idx.doc_len[6], len(preprocess_text(CATALOG[6])))

    def test_missing_term(self):
        ids, tfs = self.idx.get("zzzz")
        self.assertEqual(len(ids), 0)


class TestTFIDF(unittest.TestCase):

    def test_matches_sklearn(self):
        """Our scores must equal sklearn's TfidfVectorizer with the same settings."""
        ours = TFIDFRanker(sublinear_tf=True).fit(CATALOG, PIDS)
        vec = TfidfVectorizer(tokenizer=preprocess_text, preprocessor=None, token_pattern=None,
                              lowercase=False, sublinear_tf=True, norm="l2")
        docs = vec.fit_transform(CATALOG)
        for q in ["iphone 13 pro", "wireless headphones", "charging cable iphone iphone"]:
            ref = (vec.transform([q]) @ docs.T).toarray().ravel()
            np.testing.assert_allclose(ours.score_all(q), ref, atol=1e-9)

    def test_ranking(self):
        r = TFIDFRanker().fit(CATALOG, PIDS)
        self.assertEqual(r.rank("iphone 13 pro", top_k=1)[0][0], "B001")

    def test_explain_sums_to_score(self):
        r = TFIDFRanker().fit(CATALOG, PIDS)
        q = "apple iphone case"
        for pid, score in r.rank(q, top_k=3):
            self.assertAlmostEqual(sum(r.explain(q, pid).values()), score, places=9)

    def test_unknown_query(self):
        r = TFIDFRanker().fit(CATALOG, PIDS)
        self.assertTrue(np.all(r.score_all("qwertyuiop") == 0))


class TestBM25(unittest.TestCase):

    def setUp(self):
        self.r = BM25Ranker(k1=1.2, b=0.75).fit(CATALOG, PIDS)

    def test_hand_computed_score(self):
        idx = self.r.index
        n, avgdl = idx.num_docs, idx.avg_doc_len
        d = 6  # "iphone iphone iphone charging dock"
        expected = 0.0
        for term in ["iphon", "charg"]:
            df = idx.df[term]
            tf = float(idx.get(term)[1][list(idx.get(term)[0]).index(d)])
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            expected += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * idx.doc_len[d] / avgdl))
        self.assertAlmostEqual(self.r.score_all("iphone charging")[d], expected, places=6)

    def test_saturation(self):
        # with k1 = 0 term frequency is ignored, so a title that repeats 'iphone'
        # gets no more credit than one that says it once
        r0 = BM25Ranker(k1=0.0, b=0.0).fit(CATALOG, PIDS)
        s = r0.score_all("iphone")
        self.assertAlmostEqual(s[6], s[0])

    def test_ranking_and_candidates(self):
        self.assertEqual(self.r.rank("wireless headphones", top_k=1)[0][0], "B005")
        scored = self.r.score_candidates("samsung galaxy smartphone", ["B001", "B002", "B006"])
        self.assertEqual(scored[0][0], "B002")
        self.assertEqual(len(scored), 3)

    def test_explain_sums_to_score(self):
        q = "iphone 13 charging cable"
        for pid, score in self.r.rank(q, top_k=3):
            self.assertAlmostEqual(sum(self.r.explain(q, pid).values()), score, places=6)

    def test_set_params_reuses_index(self):
        idx = self.r.index
        self.r.set_params(2.0, 0.3)
        self.assertIs(self.r.index, idx)
        self.assertEqual((self.r.k1, self.r.b), (2.0, 0.3))


if __name__ == "__main__":
    unittest.main()
