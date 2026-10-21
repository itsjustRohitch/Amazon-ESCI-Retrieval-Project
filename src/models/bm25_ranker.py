"""
src/models/bm25_ranker.py
-------------------------
Model 2: Okapi BM25, scored directly over our inverted index.

    idf(t)      = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))
    score(q, d) = sum_{t in q} idf(t) * f_td * (k1 + 1)
                                     / (f_td + k1 * (1 - b + b * |d| / avgdl))

The "1 +" inside the log (Lucene's variant) keeps idf positive, so a term that
appears in more than half of the catalog still adds a small amount instead of
pushing the score down, which happens with the original Robertson idf.

k1 controls how quickly repeated terms saturate, b controls how strongly long
titles are penalised. Both are tuned on the dev queries in run_evaluation.py.
"""

from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.indexing import InvertedIndex, default_analyzer


class BM25Ranker:

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        analyzer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer or default_analyzer
        self.index: Optional[InvertedIndex] = None
        self.idf: Dict[str, float] = {}
        self.product_ids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}
        self.is_fitted = False

    def fit(self, corpus: List[str], product_ids: List[str]) -> "BM25Ranker":
        assert len(corpus) == len(product_ids), "corpus and product_ids differ in length"
        self.product_ids = list(product_ids)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.index = InvertedIndex(self.analyzer).build(corpus)
        n = self.index.num_docs
        self.idf = {t: float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
                    for t, df in self.index.df.items()}
        self.is_fitted = True
        return self

    def set_params(self, k1: float, b: float) -> "BM25Ranker":
        # k1 and b are only used at query time, so the index can be reused while tuning
        self.k1, self.b = k1, b
        return self

    def _term_scores(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        ids, tfs = self.index.get(term)
        norm = self.k1 * (1.0 - self.b + self.b * self.index.doc_len[ids] / self.index.avg_doc_len)
        return ids, self.idf[term] * tfs * (self.k1 + 1.0) / (tfs + norm)

    def score_all(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("BM25Ranker must be fitted before ranking.")
        scores = np.zeros(self.index.num_docs, dtype=np.float64)
        # a repeated query term counts once per occurrence, as in the original formula
        for term, qtf in Counter(self.analyzer(query)).items():
            if term in self.idf:
                ids, contrib = self._term_scores(term)
                scores[ids] += qtf * contrib
        return scores

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
        """BM25 contribution of each query term for one product (used by the UI)."""
        d = self.pid_to_idx[product_id]
        out = {}
        for term, qtf in Counter(self.analyzer(query)).items():
            if term not in self.idf:
                out[term] = 0.0
                continue
            ids, contrib = self._term_scores(term)
            pos = np.searchsorted(ids, d)
            out[term] = float(qtf * contrib[pos]) if pos < len(ids) and ids[pos] == d else 0.0
        return out


if __name__ == "__main__":
    catalog = [
        "Apple iPhone 13 Pro (128GB, Sierra Blue) - Unlocked",
        "Samsung Galaxy S22 5G Android Smartphone 128GB Phantom Black",
        "Apple iPhone 13 Protective Silicone Case with MagSafe - Chalk Pink",
        "Anker USB-C to Lightning Fast Charging Cable for iPhone",
    ]
    pids = ["B001", "B002", "B003", "B004"]
    ranker = BM25Ranker().fit(catalog, pids)
    for pid, s in ranker.rank("iphone 13 pro", top_k=3):
        print(f"{pid}  {s:.4f}  {ranker.explain('iphone 13 pro', pid)}")
