"""
figures/diagrams.py
------------------
Architecture and worked-example diagrams for the report. The worked examples
(preprocessing, postings, RRF) are produced by running the real code, so the
numbers in them are what the system actually outputs.

Every diagram is drawn at its printed size: the canvas is 6.5 inches wide (the
text width of the report) and coordinates are in inches, so a 10 pt label
prints at 10 pt.

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

INK, BODY, MUTED, RULE, PAPER = "#1b1b19", "#33322e", "#6f6d66", "#c9c6bb", "#ffffff"
SPARSE, DENSE, HYB, EVAL, SOFT = "#e6f0fb", "#ece9fa", "#fbe9e8", "#eaf6f0", "#f6f5f1"
plt.rcParams.update({"font.family": ["Calibri", "Segoe UI", "DejaVu Sans"], "savefig.dpi": 300})
MONO = "Consolas"
TITLE, TEXT, NOTE, CODE = 10.5, 9.5, 9, 8.5   # point sizes as printed

W = 6.5


def canvas(h, w=W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body=None, fc=PAPER, ec=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.32, body, ha="center", va="top", fontsize=TEXT, color=BODY, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=TITLE, fontweight="bold",
                color=INK, linespacing=1.2)


def arrow(ax, p, q, color=INK, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------- 1. architecture

def architecture():
    fig, ax = canvas(5.35)
    bw, gap = 1.33, 0.36
    xs = [0.05 + i * (bw + gap) for i in range(4)]
    h = 0.82

    ax.text(0.05, 5.2, "OFFLINE, once (about a minute)", fontsize=NOTE, color=MUTED, fontweight="bold", va="center")
    ax.plot([0.05, W - 0.05], [5.05, 5.05], color=RULE, lw=0.8)
    box(ax, xs[0], 4.05, bw, h, "ESCI sample", "19,876 products\n1,000 queries")
    box(ax, xs[1], 4.05, bw, h, "Product text", "title, plus brand\nif missing")
    box(ax, xs[2], 4.05, bw, h, "Preprocess", "clean, tokenize,\nstop-words, stem")
    box(ax, xs[3], 4.05, bw, h, "Inverted index", "postings, df,\ntitle lengths", fc=SPARSE)
    box(ax, xs[2], 3.0, bw, h, "MiniLM encoder", "384-d vectors", fc=DENSE)
    box(ax, xs[3], 3.0, bw, h, "Embeddings", "19,876 × 384,\ncached on disk", fc=DENSE)
    for i in range(3):
        arrow(ax, (xs[i] + bw, 4.46), (xs[i + 1], 4.46))
    arrow(ax, (xs[1] + bw * 0.75, 4.05), (xs[2], 3.41))
    arrow(ax, (xs[2] + bw, 3.41), (xs[3], 3.41))

    ax.text(0.05, 2.7, "ONLINE, per query (milliseconds)", fontsize=NOTE, color=MUTED, fontweight="bold", va="center")
    ax.plot([0.05, W - 0.05], [2.55, 2.55], color=RULE, lw=0.8)
    cw, cg = 1.04, 0.29
    cx = [0.05 + i * (cw + cg) for i in range(5)]
    box(ax, cx[0], 0.85, cw, 0.82, "Query", "“apple airpods\npro case”")
    box(ax, cx[1], 1.5, cw, 0.62, "Preprocess", "same steps")
    box(ax, cx[1], 0.15, cw, 0.62, "Encode", "MiniLM", fc=DENSE)
    box(ax, cx[2], 1.85, cw, 0.42, "TF-IDF", fc=SPARSE)
    box(ax, cx[2], 1.2, cw, 0.42, "BM25", fc=SPARSE)
    box(ax, cx[2], 0.15, cw, 0.62, "Dense", "cosine", fc=DENSE)
    box(ax, cx[3], 0.62, cw, 0.62, "RRF", "Σ 1/(k + rank)", fc=HYB)
    box(ax, cx[4], 0.55, cw, 0.76, "Top-k", "API and\nweb page")
    arrow(ax, (cx[0] + cw, 1.45), (cx[1], 1.81))
    arrow(ax, (cx[0] + cw, 1.05), (cx[1], 0.46))
    arrow(ax, (cx[1] + cw, 1.95), (cx[2], 2.06))
    arrow(ax, (cx[1] + cw, 1.7), (cx[2], 1.41))
    arrow(ax, (cx[1] + cw, 0.46), (cx[2], 0.46))
    arrow(ax, (cx[2] + cw, 1.41), (cx[3], 1.1))
    arrow(ax, (cx[2] + cw, 0.46), (cx[3], 0.78))
    arrow(ax, (cx[3] + cw, 0.93), (cx[4], 0.93))
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

    rows = [("Raw title", raw),
            ("Step 1. Clean: decode entities, drop punctuation, lower-case", cleaned),
            ("Step 2. Tokenize: keep sizes and hyphenated words whole", " ".join(f"[{t}]" for t in toks)),
            (f"Step 3. Remove stop-words and one-letter tokens (drops “{', '.join(removed)}”)",
             " ".join(f"[{t}]" for t in kept)),
            ("Step 4. Porter stemmer", " ".join(f"[{t}]" for t in stems))]
    row_h = 0.92
    fig, ax = canvas(row_h * len(rows))
    y = row_h * len(rows)
    for i, (label, text) in enumerate(rows):
        ax.text(0.05, y - 0.13, label, fontsize=TEXT, fontweight="bold", va="center", color=INK if i else MUTED)
        ax.add_patch(FancyBboxPatch((0.05, y - 0.62), W - 0.1, 0.34, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=SOFT if i == 0 else PAPER, ec=RULE, lw=0.8))
        ax.text(0.14, y - 0.45, text, fontsize=CODE, family=MONO, va="center", color=INK)
        if i < len(rows) - 1:
            arrow(ax, (W / 2, y - 0.63), (W / 2, y - row_h + 0.01), color=MUTED, lw=0.9)
        y -= row_h
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

    fig, ax = canvas(4.25)
    ax.text(0.05, 4.1, "Catalog", fontsize=TITLE, fontweight="bold", va="center")
    for i, d in enumerate(docs):
        ax.text(0.05, 3.8 - i * 0.24, f"d{i + 1}", fontsize=TEXT, color=MUTED, family=MONO, va="center")
        ax.text(0.4, 3.8 - i * 0.24, d, fontsize=TEXT, va="center")
    ax.plot([0.05, W - 0.05], [3.0, 3.0], color=RULE, lw=0.8)

    top = 2.75
    ax.text(0.05, top, "Stem", fontsize=TITLE, fontweight="bold", va="center")
    ax.text(0.9, top, "Postings (doc : tf)", fontsize=TITLE, fontweight="bold", va="center")
    ax.text(2.95, top, "df", fontsize=TITLE, fontweight="bold", va="center")
    ax.text(3.35, top, "idf", fontsize=TITLE, fontweight="bold", va="center")
    for i, t in enumerate(terms):
        ids, tfs = r.index.get(t)
        y = top - 0.4 - i * 0.37
        ax.text(0.05, y, t, family=MONO, fontsize=TEXT, fontweight="bold", va="center")
        x = 0.9
        for dd, tf in zip(ids, tfs):
            ax.add_patch(FancyBboxPatch((x, y - 0.13), 0.58, 0.26, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=SPARSE, ec="#9db7d3", lw=0.7))
            ax.text(x + 0.29, y, f"d{dd + 1}:{int(tf)}", ha="center", va="center", fontsize=CODE, family=MONO)
            x += 0.65
        ax.text(2.95, y, f"{r.index.df[t]}", family=MONO, fontsize=TEXT, va="center")
        ax.text(3.35, y, f"{r.idf[t]:.3f}", family=MONO, fontsize=TEXT, va="center")

    ax.plot([4.15, 4.15], [0.25, 2.9], color=RULE, lw=0.8)
    ax.text(4.35, top, f"BM25, “{q}”", fontsize=TITLE, fontweight="bold", va="center")
    order = sorted(range(3), key=lambda i: -scores[i])
    for j, i in enumerate(order):
        ax.text(4.35, top - 0.4 - j * 0.32, f"{j + 1}.  d{i + 1}   {scores[i]:.3f}", family=MONO, fontsize=TEXT,
                va="center")
    ax.text(4.35, 1.2, f"k1 = 1.2, b = 0.75\naverage length {r.index.avg_doc_len:.2f}\nonly the postings of the\n"
                       "query stems are read", fontsize=NOTE, color=MUTED, va="top", linespacing=1.3)
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

    def short(p, n):
        t = title[p]
        return t if len(t) <= n else t[: n - 1].rstrip(" ,-") + "…"

    rh = 0.33
    fig, ax = canvas(4.75)
    half = (W - 0.3) / 2
    for x, name, lst, fc in [(0.05, "BM25", lb[:k_show], SPARSE), (0.25 + half, "Dense (MiniLM)", ld[:k_show], DENSE)]:
        ax.text(x, 4.6, name, fontsize=TITLE, fontweight="bold", va="center")
        for i, (p, s) in enumerate(lst):
            y = 4.27 - i * rh
            ax.add_patch(FancyBboxPatch((x, y - 0.13), half, 0.26, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=fc, ec=RULE, lw=0.7))
            ax.text(x + 0.07, y, f"{i + 1}", fontsize=TEXT, family=MONO, va="center", color=MUTED)
            ax.text(x + 0.25, y, short(p, 34), fontsize=TEXT, va="center")
    ax.plot([0.05, W - 0.05], [2.47, 2.47], color=RULE, lw=0.8)
    ax.text(0.05, 2.25, f"Fused with RRF, k = {k}", fontsize=TITLE, fontweight="bold", va="center")
    for i, (p, s) in enumerate(fused):
        y = 1.92 - i * rh
        ax.add_patch(FancyBboxPatch((0.05, y - 0.13), W - 0.1, 0.26, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=HYB, ec=RULE, lw=0.7))
        ax.text(0.12, y, f"{i + 1}", fontsize=TEXT, family=MONO, va="center", color=MUTED)
        ax.text(0.3, y, short(p, 42), fontsize=TEXT, va="center")
        parts = [f"1/({k}+{r})" for r in (rb.get(p), rd.get(p)) if r]
        ax.text(W - 0.12, y, " + ".join(parts) + f" = {s:.4f}", fontsize=CODE, family=MONO, color=BODY,
                va="center", ha="right")
    ax.text(0.05, 0.12, f"Query: “{query}”. A product earns 1/(k + rank) from each list that contains it.",
            fontsize=NOTE, color=MUTED, va="center")
    save(fig, "diagram_rrf.png")


# ---------------------------------------------------------------------------- 5. evaluation protocol

def protocol():
    fig, ax = canvas(3.0)
    box(ax, 0.05, 1.0, 1.2, 1.0, "1,000 queries", "split once,\nseed 42")
    box(ax, 1.6, 1.75, 1.3, 0.95, "300 dev", "tuning and\nablations", fc=SOFT)
    box(ax, 1.6, 0.3, 1.3, 0.95, "700 test", "scored once", fc=EVAL)
    arrow(ax, (1.25, 1.7), (1.6, 2.2)); arrow(ax, (1.25, 1.3), (1.6, 0.8))
    box(ax, 3.25, 1.75, 1.7, 0.95, "A. Re-rank", "order the judged\ncandidates (Task 1)", fc=EVAL)
    box(ax, 3.25, 0.3, 1.7, 0.95, "B. Retrieve", "rank all 19,876\nproducts", fc=EVAL)
    arrow(ax, (2.9, 0.95), (3.25, 2.1)); arrow(ax, (2.9, 0.75), (3.25, 0.75))
    box(ax, 5.3, 0.3, 1.15, 2.4, "Metrics", "P@10\nR@10, R@100\nMAP\nnDCG@10\nPR curve\nWilcoxon test")
    arrow(ax, (4.95, 2.2), (5.3, 2.0)); arrow(ax, (4.95, 0.75), (5.3, 0.95))
    save(fig, "diagram_protocol.png")


# ---------------------------------------------------------------------------- 6. web application request flow

def web_flow():
    fig, ax = canvas(4.4)
    lanes = [(0.75, "Browser", "app.js"), (2.4, "FastAPI", "backend/main.py"), (4.05, "Rankers", "src/models"),
             (5.7, "Index, caches", "in memory")]
    for x, name, sub in lanes:
        box(ax, x - 0.72, 3.65, 1.44, 0.7, name, sub)
        ax.plot([x, x], [0.05, 3.65], color=RULE, lw=1.0, ls="--")
    steps = [
        (0.75, 2.4, 3.3, "GET /api/search"),
        (2.4, 2.4, 2.9, "clean, tokenize, stem"),
        (2.4, 4.05, 2.5, "rank() or explain()"),
        (4.05, 5.7, 2.1, "postings or q · matrix"),
        (5.7, 4.05, 1.7, "scores, all products"),
        (4.05, 2.4, 1.3, "top-k, term scores"),
        (2.4, 2.4, 0.9, "titles, labels, nDCG@10"),
        (2.4, 0.75, 0.5, "JSON to the page"),
    ]
    for x0, x1, y, label in steps:
        if x0 == x1:
            ax.add_patch(FancyBboxPatch((x0 + 0.06, y - 0.14), 1.55, 0.28, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=SOFT, ec=RULE, lw=0.7))
            ax.text(x0 + 0.13, y, label, fontsize=NOTE, va="center")
        else:
            arrow(ax, (x0, y), (x1, y), lw=0.9)
            ax.text((x0 + x1) / 2, y + 0.05, label, fontsize=NOTE, ha="center", va="bottom")
    save(fig, "diagram_web.png")


if __name__ == "__main__":
    architecture()
    preprocessing_example()
    index_example()
    protocol()
    web_flow()
    if "--no-rrf" not in sys.argv:
        rrf_example()
