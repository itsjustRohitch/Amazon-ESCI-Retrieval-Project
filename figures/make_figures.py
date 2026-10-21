"""
figures/make_figures.py
----------------------
Draws every results figure in the report from the files written by
run_evaluation.py. Run from the project root:

    python figures/make_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "evaluation_results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid (RRF)"]
SHORT = {"TF-IDF": "TF-IDF", "BM25": "BM25", "Sentence Transformer": "Dense", "Hybrid (RRF)": "Hybrid RRF", "Random": "Random"}
COLOR = {"TF-IDF": "#2a78d6", "BM25": "#1baf7a", "Sentence Transformer": "#6250d6", "Hybrid (RRF)": "#e34948", "Random": "#a3a199"}
MARK = {"TF-IDF": "o", "BM25": "s", "Sentence Transformer": "^", "Hybrid (RRF)": "D", "Random": "x"}
LABEL_COLOR = {"E": "#2c6a45", "S": "#5b7fb5", "C": "#c9a646", "I": "#c98a86"}
INK, MUTED, RULE = "#1b1b19", "#77756e", "#e3e1d9"

plt.rcParams.update({
    "font.family": ["Calibri", "Segoe UI", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load():
    with open(os.path.join(RES, "run_summary.json")) as f:
        s = json.load(f)
    return s


def fig_dataset(s):
    df = pd.read_parquet(os.path.join(ROOT, "data", "benchmark_sample.parquet"))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), gridspec_kw={"wspace": 0.32})
    counts = df.esci_label.value_counts().reindex(list("ESCI"))
    names = ["Exact", "Substi-\ntute", "Comple-\nment", "Irrele-\nvant"]
    bars = ax[0].bar(names, counts.values, color=[LABEL_COLOR[l] for l in "ESCI"], width=0.62)
    for b, v in zip(bars, counts.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 150, f"{v:,}\n{v / len(df):.0%}", ha="center", va="bottom", fontsize=8.5)
    ax[0].set_ylim(0, counts.max() * 1.3); ax[0].set_title("(a) Judgments by label"); ax[0].grid(axis="x", visible=False)

    per_q = df.groupby("query_id").size()
    ax[1].hist(per_q.clip(upper=60), bins=np.arange(5, 62, 2), color="#2a78d6", rwidth=0.85)
    ax[1].set_title("(b) Judged products per query"); ax[1].set_xlabel("products"); ax[1].set_ylabel("queries")
    ax[1].grid(axis="x", visible=False)

    qlen = df.drop_duplicates("query_id")["query"].str.split().str.len().clip(upper=10)
    vc = qlen.value_counts().sort_index()
    ax[2].bar([str(i) if i < 10 else "10+" for i in vc.index], vc.values, color="#2a78d6", width=0.7)
    ax[2].set_title("(c) Query length"); ax[2].set_xlabel("words"); ax[2].set_ylabel("queries")
    ax[2].grid(axis="x", visible=False)
    save(fig, "fig_dataset.png")


def fig_pr():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    lv = np.linspace(0, 1, 11)
    for a, tag, title in [(ax[0], "rerank", "(a) Re-ranking judged candidates"), (ax[1], "retrieval", "(b) Full-catalog retrieval")]:
        pr = pd.read_csv(os.path.join(RES, f"pr_curve_{tag}.csv"), index_col=0)
        for m in [c for c in ["Random"] + MODELS if c in pr]:
            a.plot(lv, pr[m], color=COLOR[m], marker=MARK[m], label=SHORT[m],
                   ls=":" if m == "Random" else "-", markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        a.set_title(title); a.set_xlabel("Recall"); a.set_xlim(-0.02, 1.02); a.set_ylim(0, 1.02)
    ax[0].set_ylabel("Interpolated precision")
    ax[0].legend(loc="lower left")
    ax[1].legend(loc="upper right")
    save(fig, "fig_pr_curves.png")


def fig_metric_bars():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.15})
    for a, tag, title in [(ax[0], "rerank", "(a) Re-ranking"), (ax[1], "retrieval", "(b) Full catalog")]:
        t = pd.read_csv(os.path.join(RES, f"metrics_{tag}.csv"), index_col=0)
        rows = [m for m in ["Random"] + MODELS if m in t.index]
        metrics = list(t.columns)
        x = np.arange(len(metrics)); w = 0.8 / len(rows)
        for i, m in enumerate(rows):
            a.bar(x + (i - (len(rows) - 1) / 2) * w, t.loc[m, metrics], w * 0.88, color=COLOR[m], label=SHORT[m])
        a.set_xticks(x, metrics); a.set_ylim(0, 1); a.set_title(title); a.grid(axis="x", visible=False)
    ax[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.1))
    save(fig, "fig_metric_bars.png")


def fig_bm25_grid(s):
    g = pd.read_csv(os.path.join(RES, "dev_bm25_grid.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.25})
    for a, col, title in [(ax[0], "rerank", "(a) Re-ranking, nDCG@10"), (ax[1], "retrieval", "(b) Full catalog, nDCG@10")]:
        p = g.pivot(index="k1", columns="b", values=col)
        im = a.imshow(p.values, cmap="Blues", aspect="auto")
        a.set_xticks(range(len(p.columns)), p.columns); a.set_yticks(range(len(p.index)), p.index)
        a.set_xlabel("b"); a.set_ylabel("k1"); a.set_title(title); a.grid(False)
        lo, hi = p.values.min(), p.values.max()
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                       color="white" if (v - lo) / (hi - lo + 1e-12) > 0.6 else INK)
        bi = list(p.index).index(s["chosen"]["bm25_k1"]); bj = list(p.columns).index(s["chosen"]["bm25_b"])
        a.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=INK, lw=2))
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "fig_bm25_grid.png")


def fig_ablation():
    tv = pd.read_csv(os.path.join(RES, "dev_tfidf_variants.csv"))
    ab = pd.read_csv(os.path.join(RES, "dev_preprocessing_ablation.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"wspace": 0.2, "width_ratios": [1, 1.25]})

    x = np.arange(len(tv)); w = 0.38
    ax[0].bar(x - w / 2, tv.rerank, w * 0.9, color="#2a78d6", label="re-ranking")
    ax[0].bar(x + w / 2, tv.retrieval, w * 0.9, color="#184f95", label="full catalog")
    for i, (a_, b_) in enumerate(zip(tv.rerank, tv.retrieval)):
        ax[0].text(i - w / 2, a_ + 0.01, f"{a_:.3f}", ha="center", fontsize=7.5)
        ax[0].text(i + w / 2, b_ + 0.01, f"{b_:.3f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(x, tv.variant); ax[0].set_ylim(0, 1); ax[0].set_ylabel("nDCG@10 (dev)")
    ax[0].set_title("(a) TF-IDF weighting"); ax[0].legend(loc="upper right"); ax[0].grid(axis="x", visible=False)

    settings = list(dict.fromkeys(ab.setting))
    x = np.arange(len(settings)); w = 0.2
    for i, (m, proto) in enumerate([("TF-IDF", "rerank"), ("BM25", "rerank"), ("TF-IDF", "retrieval"), ("BM25", "retrieval")]):
        vals = ab[ab.model == m].set_index("setting").loc[settings, proto]
        ax[1].bar(x + (i - 1.5) * w, vals, w * 0.9, color=COLOR[m], alpha=1.0 if proto == "rerank" else 0.55,
                  label=f"{m}, {'re-ranking' if proto == 'rerank' else 'full catalog'}")
    vocab = ab[ab.model == "BM25"].set_index("setting").loc[settings, "vocabulary"]
    ax[1].set_xticks(x, [f"{s}\n({v:,} terms)" for s, v in zip(settings, vocab)])
    ax[1].set_ylim(0, 1); ax[1].set_title("(b) Preprocessing ablation"); ax[1].grid(axis="x", visible=False)
    ax[1].legend(ncol=2, loc="upper right", fontsize=8)
    save(fig, "fig_ablation.png")


def fig_rrf(s):
    rk = pd.read_csv(os.path.join(RES, "dev_rrf_k.csv"))
    rc = pd.read_csv(os.path.join(RES, "dev_rrf_combos.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.25})
    ax[0].plot(rk.k, rk.rerank, color=COLOR["Hybrid (RRF)"], marker="D", label="re-ranking")
    ax[0].plot(rk.k, rk.retrieval, color="#184f95", marker="o", label="full catalog")
    ax[0].set_xscale("log"); ax[0].set_xticks(rk.k, rk.k); ax[0].minorticks_off()
    ax[0].set_xlabel("RRF constant k"); ax[0].set_ylabel("nDCG@10 (dev)"); ax[0].set_title("(a) Effect of k, BM25 + Dense")
    ax[0].legend()
    y = np.arange(len(rc))
    ax[1].barh(y + 0.2, rc.rerank, 0.36, color=COLOR["Hybrid (RRF)"], label="re-ranking")
    ax[1].barh(y - 0.2, rc.retrieval, 0.36, color="#184f95", label="full catalog")
    for i, (a_, b_) in enumerate(zip(rc.rerank, rc.retrieval)):
        ax[1].text(a_ + 0.005, i + 0.2, f"{a_:.3f}", va="center", fontsize=8)
        ax[1].text(b_ + 0.005, i - 0.2, f"{b_:.3f}", va="center", fontsize=8)
    ax[1].set_yticks(y, rc.combo); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1)
    ax[1].set_title(f"(b) Which lists to fuse (k = {s['chosen']['rrf_k']})"); ax[1].grid(axis="y", visible=False)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig_rrf.png")


def per_query(tag):
    return pd.read_csv(os.path.join(RES, f"per_query_{tag}.csv")).set_index(["model", "query_id"])


def fig_query_length():
    feat = pd.read_csv(os.path.join(RES, "test_query_features.csv"), index_col=0)
    bucket = pd.cut(feat.n_words, [0, 2, 3, 4, 5, 7, 100], labels=["1-2", "3", "4", "5", "6-7", "8+"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=False, gridspec_kw={"wspace": 0.22})
    for a, tag, title in [(ax[0], "rerank", "(a) Re-ranking"), (ax[1], "retrieval", "(b) Full catalog")]:
        pq = per_query(tag)
        for m in MODELS:
            v = pq.loc[m, "nDCG@10"]
            means = v.groupby(bucket.reindex(v.index), observed=False).mean()
            a.plot(range(len(means)), means.values, color=COLOR[m], marker=MARK[m], label=SHORT[m],
                   markeredgecolor="white", markeredgewidth=0.8)
        counts = bucket.value_counts().reindex(means.index)
        a.set_xticks(range(len(means)), [f"{l}\nn={c}" for l, c in zip(means.index, counts)])
        a.set_xlabel("words in query"); a.set_title(title)
    ax[0].set_ylabel("mean nDCG@10")
    ax[1].legend(loc="best")
    save(fig, "fig_query_length.png")


def fig_label_position():
    lp = pd.read_csv(os.path.join(RES, "label_by_position.csv")).set_index(["model", "pos"])
    order = ["Random"] + MODELS
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, m in zip(ax, order):
        t = lp.loc[m].reindex(columns=list("ESCI"), fill_value=0)
        share = t.div(t.sum(axis=1), axis=0)
        bottom = np.zeros(len(share))
        for l in "ESCI":
            a.bar(share.index, share[l], bottom=bottom, color=LABEL_COLOR[l], width=0.86, label=l,
                  edgecolor="white", linewidth=0.6)
            bottom += share[l].values
        a.set_title(SHORT[m]); a.set_xticks(range(1, 11)); a.set_ylim(0, 1); a.grid(False)
        a.set_xlabel("rank position")
    ax[0].set_ylabel("share of results")
    ax[-1].legend(["Exact", "Substitute", "Complement", "Irrelevant"], loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig_label_position.png")


def fig_head_to_head():
    pq = per_query("retrieval")
    b, d, h = (pq.loc[m, "nDCG@10"] for m in ["BM25", "Sentence Transformer", "Hybrid (RRF)"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.1]})
    ax[0].scatter(b, d, s=12, color="#6250d6", alpha=0.45, linewidths=0)
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax[0].set_xlabel("BM25 nDCG@10"); ax[0].set_ylabel("Dense nDCG@10"); ax[0].set_aspect("equal")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.02)
    wins, loss = int((d > b + 1e-9).sum()), int((b > d + 1e-9).sum())
    ax[0].text(0.03, 0.95, f"Dense better: {wins}", fontsize=9, va="top")
    ax[0].text(0.97, 0.05, f"BM25 better: {loss}", fontsize=9, ha="right")
    ax[0].set_title("(a) Per-query, full catalog")

    rows = []
    for name, x, y in [("Dense vs BM25", d, b), ("Hybrid vs BM25", h, b), ("Hybrid vs Dense", h, d)]:
        rows.append((name, int((x > y + 1e-9).sum()), int((abs(x - y) <= 1e-9).sum()), int((y > x + 1e-9).sum())))
    names = [r[0] for r in rows]
    yy = np.arange(len(rows))
    w_, t_, l_ = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    ax[1].barh(yy, w_, color="#1baf7a", label="first model better", height=0.55)
    ax[1].barh(yy, t_, left=w_, color="#d6d4cc", label="tie", height=0.55)
    ax[1].barh(yy, l_, left=w_ + t_, color="#c98a86", label="second model better", height=0.55)
    for i in range(len(rows)):
        ax[1].text(w_[i] / 2, i, w_[i], ha="center", va="center", color="white", fontsize=9)
        ax[1].text(w_[i] + t_[i] + l_[i] / 2, i, l_[i], ha="center", va="center", color="white", fontsize=9)
    ax[1].set_yticks(yy, names); ax[1].invert_yaxis(); ax[1].grid(False)
    ax[1].set_xlabel("test queries"); ax[1].set_title("(b) Wins, ties and losses on nDCG@10")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    save(fig, "fig_head_to_head.png")


def fig_cost(s):
    lat = s["latency_ms_per_query"]
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    names = MODELS[::-1]
    vals = [lat[m] for m in names]
    ax.barh([SHORT[m] for m in names], vals, color=[COLOR[m] for m in names], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v * 1.02, i, f"{v:.1f} ms", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25); ax.set_xlabel(f"milliseconds per query, top 100 of {s['products']:,} products (CPU)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_latency.png")


def fig_index(s):
    dfc = pd.read_csv(os.path.join(RES, "term_document_frequency.csv"), index_col=0)["df"].values
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"wspace": 0.25})
    r = np.arange(1, len(dfc) + 1)
    ax[0].loglog(r, dfc, color="#2a78d6", lw=1.6)
    ax[0].set_xlabel("term rank"); ax[0].set_ylabel("document frequency")
    ax[0].set_title("(a) Rank vs document frequency (log-log)")
    bins = [1, 2, 3, 6, 11, 51, 101, 1001, 1e6]
    labels = ["1", "2", "3-5", "6-10", "11-50", "51-100", "101-1k", ">1k"]
    h, _ = np.histogram(dfc, bins=bins)
    ax[1].bar(labels, h, color="#2a78d6", width=0.7)
    for i, v in enumerate(h):
        ax[1].text(i, v + h.max() * 0.01, f"{v:,}", ha="center", fontsize=8)
    ax[1].set_xlabel("postings list length (df)"); ax[1].set_ylabel("terms"); ax[1].grid(axis="x", visible=False)
    ax[1].set_title(f"(b) Postings length, {len(dfc):,} terms")
    save(fig, "fig_index.png")


def fig_bm25_curves(s):
    k1c, bc = s["chosen"]["bm25_k1"], s["chosen"]["bm25_b"]
    f = np.arange(0, 11)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"wspace": 0.25})
    for k1, col in [(0.6, "#1baf7a"), (1.2, "#2a78d6"), (2.0, "#6250d6")]:
        sat = f * (k1 + 1) / (f + k1)
        ax[0].plot(f, sat, color=col, marker="o", markersize=4, label=f"k1 = {k1}" + ("  (ours)" if k1 == k1c else ""))
    ax[0].plot(f, f, color=MUTED, lw=1, ls=":", label="raw tf")
    ax[0].set_ylim(0, 4); ax[0].set_xlabel("occurrences of the term in the title, f(t,d)")
    ax[0].set_ylabel("tf factor (b = 0)"); ax[0].set_title("(a) Term-frequency saturation"); ax[0].legend(loc="upper left")
    r = np.linspace(0.25, 3, 60)
    for b, col in [(0.0, "#2a78d6"), (0.5, "#6250d6"), (1.0, "#1baf7a")]:
        val = 1 * (k1c + 1) / (1 + k1c * (1 - b + b * r))
        ax[1].plot(r, val, color=col, label=f"b = {b}" + ("  (ours)" if b == bc else ""))
    ax[1].axvline(1, color=MUTED, lw=1)
    ax[1].text(1.03, 0.25, "average\nlength", fontsize=8.5, color=MUTED)
    ax[1].set_xlabel("title length / average title length"); ax[1].set_ylabel(f"tf factor for f = 1 (k1 = {k1c})")
    ax[1].set_title("(b) Length normalisation"); ax[1].legend(loc="upper right"); ax[1].set_ylim(0, 2.4)
    save(fig, "fig_bm25_curves.png")


def fig_titles():
    from src.preprocessing import preprocess_text
    cat = pd.read_parquet(os.path.join(ROOT, "data", "catalog_cache.parquet"))
    words = cat.product_title.str.split().str.len().clip(upper=45)
    stems = pd.Series([len(preprocess_text(t)) for t in cat.text]).clip(upper=45)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, v, title in [(ax[0], words, "(a) Words per raw title"), (ax[1], stems, "(b) Stems per title after preprocessing")]:
        a.hist(v, bins=np.arange(0, 47, 2), color="#2a78d6", rwidth=0.85)
        a.axvline(v.median(), color=INK, lw=1)
        a.text(v.median() + 1, 0.92, f"median {int(v.median())}", fontsize=8.5, transform=a.get_xaxis_transform())
        a.set_title(title); a.set_xlabel("length (capped at 45)"); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("products")
    save(fig, "fig_titles.png")


def fig_embeddings():
    from sklearn.decomposition import PCA
    from src.models.dense_ranker import DenseRanker
    bench = pd.read_parquet(os.path.join(ROOT, "data", "benchmark_sample.parquet"))
    cat = pd.read_parquet(os.path.join(ROOT, "data", "catalog_cache.parquet"))
    emb = np.load(os.path.join(ROOT, "data", "embeddings_cache.npy"))
    pos = {p: i for i, p in enumerate(cat.product_id)}
    queries = ["wood handle induction pan", "go pro camera", "brown flats for women",
               "bluetooth speaker waterproof", "14 gallon trash can"]
    queries = [q for q in queries if q in set(bench["query"])]
    dense = DenseRanker().load_cached_embeddings(emb, cat.product_id.tolist())
    pts, labs, qv = [], [], []
    for q in queries:
        ex = bench[(bench["query"] == q) & (bench.esci_label.isin(["E", "S"]))].product_id
        for p in ex:
            pts.append(emb[pos[p]]); labs.append(q)
        qv.append(dense.model.encode([q], normalize_embeddings=True)[0])
    pca = PCA(n_components=2, random_state=0).fit(np.vstack(pts + qv))
    P, Q = pca.transform(np.vstack(pts)), pca.transform(np.vstack(qv))
    cols = ["#2a78d6", "#1baf7a", "#6250d6", "#e34948", "#eda100"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for q, c in zip(queries, cols):
        m = np.array(labs) == q
        ax.scatter(P[m, 0], P[m, 1], s=28, color=c, alpha=0.75, linewidths=0, label=q)
    for (x, y), q, c in zip(Q, queries, cols):
        ax.scatter([x], [y], s=150, marker="*", color=c, edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xlabel("first principal component"); ax.set_ylabel("second principal component")
    ax.set_title("MiniLM embeddings of Exact and Substitute products, stars are the queries")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    save(fig, "fig_embeddings.png")


if __name__ == "__main__":
    s = load()
    fig_dataset(s)
    fig_pr()
    fig_metric_bars()
    fig_bm25_grid(s)
    fig_ablation()
    fig_rrf(s)
    fig_query_length()
    fig_label_position()
    fig_head_to_head()
    fig_cost(s)
    fig_index(s)
    fig_bm25_curves(s)
    fig_titles()
    fig_embeddings()
