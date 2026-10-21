"""
src/indexing.py
---------------
Inverted index over the product catalog, shared by the TF-IDF and BM25 rankers.

For every term we keep a postings list of (doc_id, term_frequency) stored as two
numpy arrays, plus the document frequency. Per-document lengths are kept for
BM25 length normalisation.

    term  ->  doc_ids = [3, 17, 254, ...]
              tfs     = [1,  2,   1, ...]
"""

from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.preprocessing import preprocess_text


def default_analyzer(text: str) -> List[str]:
    return preprocess_text(text, apply_stemming=True, return_tokens=True)


class InvertedIndex:

    def __init__(self, analyzer: Optional[Callable[[str], List[str]]] = None):
        self.analyzer = analyzer or default_analyzer
        self.postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.df: Dict[str, int] = {}
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, corpus: Iterable[str]) -> "InvertedIndex":
        builder: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        lengths = []

        for doc_id, text in enumerate(corpus):
            tokens = self.analyzer(text or "")
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                builder[term].append((doc_id, tf))

        self.num_docs = len(lengths)
        self.doc_len = np.asarray(lengths, dtype=np.float32)
        self.avg_doc_len = float(self.doc_len.mean()) if self.num_docs else 0.0

        # doc ids are appended in increasing order, so each list is already sorted
        for term, plist in builder.items():
            ids = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (ids, tfs)
            self.df[term] = len(plist)
        return self

    def get(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
        return self.postings.get(term, empty)

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    def stats(self) -> Dict[str, float]:
        plens = np.array(list(self.df.values())) if self.df else np.zeros(1)
        return {
            "documents": self.num_docs,
            "vocabulary": self.vocab_size,
            "postings": int(plens.sum()),
            "avg_doc_len": round(self.avg_doc_len, 2),
            "singleton_terms": int((plens == 1).sum()),
            "longest_postings": int(plens.max()),
        }
