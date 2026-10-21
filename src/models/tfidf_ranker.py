"""
src/models/tfidf_ranker.py
--------------------------
Model 1: TF-IDF vector space model, written on top of our own inverted index.

Weights (defaults match sklearn's TfidfVectorizer(sublinear_tf=True), which the
tests use as a reference):

    tf(t, d)  = 1 + ln(f_td)            if sublinear_tf, else f_td
    idf(t)    = ln((1 + N) / (1 + df_t)) + 1
    w(t, d)   = tf(t, d) * idf(t)
    score     = cos(q, d) = sum_t w(t,q) * w(t,d) / (|q| * |d|)

Scoring is term-at-a-time: for each query term we walk its postings list and
add the contribution into a score accumulator, so only documents sharing at
least one term with the query are ever touched.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.indexing import InvertedIndex, default_analyzer


class TFIDFRanker:

    def __init__(
        self,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        analyzer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.sublinear_tf = sublinear_tf
        self.use_idf = use_idf
        self.analyzer = analyzer or default_analyzer
        self.index: Optional[InvertedIndex] = None
        self.idf: Dict[str, float] = {}
        self.doc_norm: np.ndarray = np.zeros(0)
        self.product_ids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}
        self.is_fitted = False

    def _tf(self, f):
        return 1.0 + np.log(f) if self.sublinear_tf else f

    def fit(self, corpus: List[str], product_ids: List[str]) -> "TFIDFRanker":
        assert len(corpus) == len(product_ids), "corpus and product_ids differ in length"
        self.product_ids = list(product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.index = InvertedIndex(self.analyzer).build(corpus)

        n = self.index.num_docs
        sq_norm = np.zeros(n, dtype=np.float64)
        for term, (ids, tfs) in self.index.postings.items():
            idf = np.log((1 + n) / (1 + self.index.df[term])) + 1.0 if self.use_idf else 1.0
            self.idf[term] = idf
            sq_norm[ids] += (self._tf(tfs) * idf) ** 2

        self.doc_norm = np.sqrt(sq_norm)
        self.doc_norm[self.doc_norm == 0] = 1.0
        self.is_fitted = True
        return self

    def _query_weights(self, query: str) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for tok in self.analyzer(query):
            if tok in self.idf:  # terms unseen in the catalog cannot match anything
                counts[tok] = counts.get(tok, 0) + 1
        weights = {t: float(self._tf(c)) * self.idf[t] for t, c in counts.items()}
        norm = np.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {t: w / norm for t, w in weights.items()}

    def score_all(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TFIDFRanker must be fitted before ranking.")
        scores = np.zeros(self.index.num_docs, dtype=np.float64)
        for term, wq in self._query_weights(query).items():
            ids, tfs = self.index.get(term)
            scores[ids] += wq * self._tf(tfs) * self.idf[term]
        return scores / self.doc_norm

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self.score_all(query)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.product_ids[i], float(scores[i])) for i in top]

    def score_candidates(self, query: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        scores = self.score_all(query)
        out = [(pid, float(scores[self.pid_to_idx[pid]]) if pid in self.pid_to_idx else 0.0)
               for pid in candidate_ids]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def explain(self, query: str, product_id: str) -> Dict[str, float]:
        """Per-term share of the cosine score for one product (used by the UI)."""
        d = self.pid_to_idx[product_id]
        out = {}
        for term, wq in self._query_weights(query).items():
            ids, tfs = self.index.get(term)
            pos = np.searchsorted(ids, d)
            hit = pos < len(ids) and ids[pos] == d
            out[term] = float(wq * self._tf(tfs[pos]) * self.idf[term] / self.doc_norm[d]) if hit else 0.0
        return out


if __name__ == "__main__":
    catalog = [
        "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
        "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
        "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
        "Anker USB-C to Lightning Fast Charging Cable for iPhone",
    ]
    pids = ["B001", "B002", "B003", "B004"]
    ranker = TFIDFRanker().fit(catalog, pids)
    for pid, s in ranker.rank("iphone 13 pro", top_k=3):
        print(f"{pid}  {s:.4f}  {ranker.explain('iphone 13 pro', pid)}")
