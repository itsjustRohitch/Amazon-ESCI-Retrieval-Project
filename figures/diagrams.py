"""
figures/diagrams.py
------------------
Architecture and worked-example diagrams for the report. The worked examples
(preprocessing, postings, RRF) are produced by running the real code, so the
numbers in them are what the system actually outputs.

    python figures/diagrams.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, MUTED, RULE, PAPER = "#1b1b19", "#77756e", "#cfccc1", "#ffffff"
SPARSE, DENSE, HYB, EVAL = "#e6f0fb", "#ece9fa", "#fbe9e8", "#eaf6f0"
plt.rcParams.update({"font.family": ["Calibri", "Segoe UI", "DejaVu Sans"], "font.size": 10, "savefig.dpi": 220})
MONO = "Consolas"


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body=None, fc=PAPER, ec=INK, size=10, mono_body=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=size, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=size - 1.5, color="#3d3c38",
                family=MONO if mono_body else None, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, fontweight="bold", color=INK)


def arrow(ax, p, q, color=INK, style="-|>", lw=1.1, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------- 1. architecture

def architecture():
    fig, ax = canvas(12, 7.2)
    ax.text(0.2, 6.95, "OFFLINE  (once, about 1 minute)", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(0.2, 3.55, "ONLINE  (per query, milliseconds)", fontsize=9, color=MUTED, fontweight="bold")
    ax.plot([0.2, 11.8], [3.8, 3.8], color=RULE, lw=1)

    box(ax, 0.2, 5.3, 2.1, 1.4, "ESCI sample", "20,156 judgments\n1,000 queries\n19,876 products")
    box(ax, 2.8, 5.3, 2.1, 1.4, "Product text", "title\n+ brand if missing\nfrom title")
    box(ax, 5.4, 5.85, 2.3, 0.95, "Preprocessing", "clean, tokenize,\nstop-words, Porter", size=9.5)
    box(ax, 5.4, 4.15, 2.3, 1.2, "MiniLM-L6-v2", "384-d, L2-normalised\nencoder", fc=DENSE, size=9.5)
    box(ax, 8.3, 5.85, 3.5, 0.95, "Inverted index", "term → [(doc, tf)], df, |d|, avgdl", fc=SPARSE, size=9.5)
    box(ax, 8.3, 4.15, 3.5, 1.2, "Embedding matrix", "19,876 × 384 float32\ncached in data/*.npy", fc=DENSE, size=9.5)
    arrow(ax, (2.3, 6.0), (2.8, 6.0))
    arrow(ax, (4.9, 6.3), (5.4, 6.3)); arrow(ax, (4.9, 5.6), (5.4, 4.8), rad=0.0)
    arrow(ax, (7.7, 6.3), (8.3, 6.3)); arrow(ax, (7.7, 4.75), (8.3, 4.75))

    box(ax, 0.2, 1.55, 1.7, 1.3, "Query", '"apple airpods\npro case"', size=10)
    box(ax, 2.4, 2.25, 2.0, 1.0, "Preprocess", "same pipeline", size=9.5)
    box(ax, 2.4, 0.55, 2.0, 1.0, "Encode", "MiniLM, 384-d", fc=DENSE, size=9.5)
    box(ax, 4.9, 2.75, 2.3, 0.75, "TF-IDF  cosine", fc=SPARSE, size=9.5)
    box(ax, 4.9, 1.85, 2.3, 0.75, "BM25  (k1, b)", fc=SPARSE, size=9.5)
    box(ax, 4.9, 0.55, 2.3, 1.0, "Dense  q · d", "dot product of\nunit vectors", fc=DENSE, size=9.5)
    box(ax, 7.75, 1.2, 1.9, 1.3, "RRF", "Σ 1 / (k + rank)", fc=HYB)
    box(ax, 10.1, 1.2, 1.7, 1.3, "Top-k", "FastAPI\n+ web UI", size=10)
    arrow(ax, (1.9, 2.4), (2.4, 2.75)); arrow(ax, (1.9, 2.0), (2.4, 1.05))
    arrow(ax, (4.4, 2.85), (4.9, 3.1)); arrow(ax, (4.4, 2.6), (4.9, 2.2))
    arrow(ax, (4.4, 1.05), (4.9, 1.05))
    arrow(ax, (7.2, 2.2), (7.75, 2.0)); arrow(ax, (7.2, 1.05), (7.75, 1.55))
    arrow(ax, (9.65, 1.85), (10.1, 1.85))
    ax.text(7.75, 0.75, "TF-IDF, BM25 and Dense are\nalso served on their own", fontsize=8.5, color=MUTED, va="top")
    ax.text(10.1, 3.35, "sparse scorers read the index,\ndense reads the matrix", fontsize=8.5, color=MUTED, va="top")
    save(fig, "diagram_architecture.png")


# ---------------------------------------------------------------------------- 2. preprocessing

def preprocessing_example():
    from src.preprocessing import clean_text, remove_stopwords, stem_tokens, tokenize
    raw = "Women's Running Shoes &amp; Cushioned Sneakers - Size 8.5, Non-Slip (Black/White)"
    cleaned = clean_text(raw)
    toks = tokenize(cleaned)
    kept = remove_stopwords(toks)
    stems = stem_tokens(kept)
    removed = [t for t in toks if t not in kept]

    rows = [("Raw title", raw), ("1  Clean", cleaned), ("2  Tokenize", "  ".join(f"[{t}]" for t in toks)),
            ("3  Stop-words", "  ".join(f"[{t}]" for t in kept) + f"     removed: {', '.join(removed) or 'none'}"),
            ("4  Porter stem", "  ".join(f"[{t}]" for t in stems))]
    fig, ax = canvas(12, 3.5)
    y = 3.1
    for i, (k, v) in enumerate(rows):
        ax.text(0.2, y, k, fontsize=10, fontweight="bold", va="center", color=INK if i else MUTED)
        ax.add_patch(FancyBboxPatch((2.0, y - 0.25), 9.8, 0.5, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#f6f5f1" if i == 0 else PAPER, ec=RULE, lw=0.9))
        ax.text(2.15, y, v, fontsize=9.2, family=MONO, va="center", color=INK)
        if i < len(rows) - 1:
            arrow(ax, (1.0, y - 0.2), (1.0, y - 0.48), color=MUTED)
        y -= 0.68
    save(fig, "diagram_preprocessing.png")
    return raw, stems


# ---------------------------------------------------------------------------- 3. inverted index

def index_example():
    from src.models.bm25_ranker import BM25Ranker
    docs = ["Apple AirPods Pro Case, Silicone Cover",
            "AirPods Pro (2nd Generation) Wireless Earbuds",
            "Silicone Case for iPhone 13 Pro"]
    r = BM25Ranker(k1=1.2, b=0.75).fit(docs, ["d1", "d2", "d3"])
    q = "airpods pro case"
    scores = r.score_all(q)
    terms = ["airpod", "pro", "case", "silicon", "appl", "earbud"]

    fig, ax = canvas(12, 4.1)
    ax.text(0.2, 3.85, "Catalog", fontsize=10, fontweight="bold")
    for i, d in enumerate(docs):
        ax.text(0.2, 3.4 - i * 0.45, f"d{i + 1}", fontsize=9.5, color=MUTED, family=MONO)
        ax.text(0.6, 3.4 - i * 0.45, d, fontsize=9.5)
    ax.text(4.9, 3.85, "Postings  (doc : tf)", fontsize=10, fontweight="bold")
    ax.text(8.1, 3.85, "df   idf", fontsize=10, fontweight="bold", family=MONO)
    for i, t in enumerate(terms):
        ids, tfs = r.index.get(t)
        y = 3.4 - i * 0.42
        ax.text(4.9, y, t, family=MONO, fontsize=9.5, fontweight="bold")
        x = 6.0
        for d, tf in zip(ids, tfs):
            ax.add_patch(FancyBboxPatch((x, y - 0.15), 0.62, 0.32, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=SPARSE, ec="#9db7d3", lw=0.8))
            ax.text(x + 0.31, y, f"d{d + 1}:{int(tf)}", ha="center", va="center", fontsize=8.5, family=MONO)
            x += 0.7
        ax.text(8.1, y, f"{r.index.df[t]:<4} {r.idf[t]:.3f}", family=MONO, fontsize=9.5, va="center")

    ax.text(9.8, 3.85, f'BM25 for "{q}"', fontsize=10, fontweight="bold")
    order = sorted(range(3), key=lambda i: -scores[i])
    for j, i in enumerate(order):
        ax.text(9.8, 3.4 - j * 0.45, f"{j + 1}.  d{i + 1}   {scores[i]:.3f}", family=MONO, fontsize=9.5)
    ax.text(9.8, 1.85, f"k1 = 1.2, b = 0.75\navgdl = {r.index.avg_doc_len:.2f} stems\nonly postings of query\nterms are read",
            fontsize=8.8, color=MUTED, va="top", linespacing=1.4)
    ax.plot([4.6, 4.6], [1.0, 4.0], color=RULE); ax.plot([9.55, 9.55], [1.0, 4.0], color=RULE)
    save(fig, "diagram_index.png")


# ---------------------------------------------------------------------------- 4. RRF worked example

def rrf_example(query="bluetooth speaker waterproof", k_show=5):
    """Uses the real catalog and models; needs the embedding cache from run_evaluation.py."""
    import json
    import numpy as np
    import pandas as pd
    from src.models.bm25_ranker import BM25Ranker
    from src.models.dense_ranker import DenseRanker
    from src.models.hybrid_rrf import ReciprocalRankFusion

    cat = pd.read_parquet(os.path.join(ROOT, "data", "catalog_cache.parquet"))
    with open(os.path.join(ROOT, "evaluation_results", "run_summary.json")) as f:
        ch = json.load(f)["chosen"]
    pids = cat.product_id.tolist()
    title = dict(zip(cat.product_id, cat.product_title))
    bm25 = BM25Ranker(k1=ch["bm25_k1"], b=ch["bm25_b"]).fit(cat.text.tolist(), pids)
    dense = DenseRanker().load_cached_embeddings(np.load(os.path.join(ROOT, "data", "embeddings_cache.npy")), pids)
    k = ch["rrf_k"]
    lb, ld = bm25.rank(query, 50), dense.rank(query, 50)
    fused = ReciprocalRankFusion.fuse_rank_lists([lb, ld], k=k)[:k_show]
    rb = {p: i + 1 for i, (p, _) in enumerate(lb)}
    rd = {p: i + 1 for i, (p, _) in enumerate(ld)}

    def short(p, n=40):
        t = title[p]
        return t if len(t) <= n else t[: n - 1] + "…"

    fig, ax = canvas(13, 3.9)
    cols = [(0.2, "BM25", lb[:k_show], SPARSE), (3.95, "Dense (MiniLM)", ld[:k_show], DENSE)]
    for x, name, lst, fc in cols:
        ax.text(x, 3.6, name, fontsize=10.5, fontweight="bold")
        for i, (p, s) in enumerate(lst):
            y = 3.1 - i * 0.5
            ax.add_patch(FancyBboxPatch((x, y - 0.19), 3.5, 0.38, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=RULE))
            ax.text(x + 0.08, y, f"{i + 1}", fontsize=9, family=MONO, va="center", color=MUTED)
            ax.text(x + 0.35, y, short(p, 36), fontsize=8.6, va="center")
    x = 7.7
    ax.text(x, 3.6, f"RRF, k = {k}", fontsize=10.5, fontweight="bold")
    for i, (p, s) in enumerate(fused):
        y = 3.1 - i * 0.5
        ax.add_patch(FancyBboxPatch((x, y - 0.19), 5.1, 0.38, boxstyle="round,pad=0,rounding_size=0.04", fc=HYB, ec=RULE))
        ax.text(x + 0.08, y, f"{i + 1}", fontsize=9, family=MONO, va="center", color=MUTED)
        ax.text(x + 0.35, y, short(p, 28), fontsize=8.6, va="center")
        parts = [f"1/({k}+{r})" for r in (rb.get(p), rd.get(p)) if r]
        ax.text(x + 5.0, y, " + ".join(parts) + f" = {s:.4f}", fontsize=7.8, family=MONO, color="#3d3c38",
                va="center", ha="right")
    ax.text(0.2, 0.25, f'Query: "{query}".  Each product gets 1/(k + rank) from every list it appears in; '
                      f'products near the top of both lists win.', fontsize=9, color=MUTED)
    save(fig, "diagram_rrf.png")


# ---------------------------------------------------------------------------- 5. evaluation protocol

def protocol():
    fig, ax = canvas(12, 3.6)
    box(ax, 0.2, 1.2, 2.2, 1.5, "1,000 queries", "fixed seed 42", size=10)
    box(ax, 3.0, 2.15, 2.2, 0.95, "300 dev", "tune k1, b, RRF k,\nablations", fc="#f6f5f1", size=9.5)
    box(ax, 3.0, 0.75, 2.2, 0.95, "700 test", "reported once", fc=EVAL, size=9.5)
    arrow(ax, (2.4, 2.2), (3.0, 2.6)); arrow(ax, (2.4, 1.7), (3.0, 1.25))
    box(ax, 6.0, 2.1, 2.7, 1.25, "Protocol A: re-rank", "order the ~20 judged\ncandidates of the query\n(ESCI Task 1)", fc=EVAL, size=9.5)
    box(ax, 6.0, 0.3, 2.7, 1.25, "Protocol B: retrieve", "rank all 19,876 products;\nunjudged = not relevant", fc=EVAL, size=9.5)
    arrow(ax, (5.2, 1.3), (6.0, 2.6)); arrow(ax, (5.2, 1.1), (6.0, 0.95))
    box(ax, 9.3, 1.0, 2.5, 1.85, "Metrics", "P@10, R@10, R@100\nMAP, nDCG@10\n11-pt PR curve\nWilcoxon test", size=9.5)
    arrow(ax, (8.7, 2.7), (9.3, 2.2)); arrow(ax, (8.7, 0.95), (9.3, 1.5))
    ax.text(9.3, 0.55, "gains: E 3, S 2, C 1, I 0\nbinary relevant: Exact", fontsize=8.5, color=MUTED, va="top")
    save(fig, "diagram_protocol.png")


# ---------------------------------------------------------------------------- 6. web application request flow

def web_flow():
    fig, ax = canvas(12, 4.6)
    lanes = [(0.9, "Browser\n(app.js)"), (3.9, "FastAPI\n(backend/main.py)"), (6.9, "Rankers\n(src/models)"),
             (9.9, "Index and caches\n(data/, in memory)")]
    for x, name in lanes:
        box(ax, x - 0.85, 3.75, 1.7, 0.7, name, size=9)
        ax.plot([x, x], [0.2, 3.75], color=RULE, lw=1.2, ls="--")
    steps = [
        (0.9, 3.9, 3.35, "GET /api/search?q=...&model=bm25"),
        (3.9, 3.9, 2.95, "clean, tokenize, stem the query"),
        (3.9, 6.9, 2.55, "rank(q, k)  or  explain(q, product)"),
        (6.9, 9.9, 2.15, "read postings of query stems / query vector · matrix"),
        (9.9, 6.9, 1.75, "scores for all 19,876 products"),
        (6.9, 3.9, 1.35, "top-k ids, per-term contributions"),
        (3.9, 3.9, 0.95, "attach titles, ESCI labels, nDCG@10"),
        (3.9, 0.9, 0.55, "JSON, rendered as the result list"),
    ]
    for x0, x1, y, label in steps:
        if x0 == x1:
            ax.add_patch(FancyBboxPatch((x0 + 0.08, y - 0.13), 2.6, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc="#f6f5f1", ec=RULE))
            ax.text(x0 + 0.18, y, label, fontsize=8.5, va="center")
        else:
            arrow(ax, (x0, y), (x1, y), lw=1.0)
            ax.text((x0 + x1) / 2, y + 0.07, label, fontsize=8.5, ha="center", va="bottom")
    save(fig, "diagram_web.png")


if __name__ == "__main__":
    architecture()
    preprocessing_example()
    index_example()
    protocol()
    web_flow()
    if "--no-rrf" not in sys.argv:
        rrf_example()
