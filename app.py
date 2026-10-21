"""
app.py
------
Interactive Search Web Interface for Project 14: Product Search using Amazon ESCI.
Built with Streamlit.

Features:
- Live e-commerce search with 4 ranking paradigms:
  1. TF-IDF (Vector Space)
  2. BM25 (Probabilistic Sparse Retrieval)
  3. Sentence Transformer (Dense Semantic Retrieval using 384-D Embeddings)
  4. Hybrid (Reciprocal Rank Fusion - RRF)
- Side-by-Side 4-Model Comparison Mode for arbitrary queries.
- Live inspection of raw vs. scratch-preprocessed tokens.
- Interactive Benchmarking tab with Master Performance Comparison Table & Precision-Recall curves.
- Dataset EDA summary tab.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.preprocessing import clean_text, tokenize, remove_stopwords, stem_tokens, preprocess_text
from src.models.tfidf_ranker import TFIDFRanker
from src.models.bm25_ranker import BM25Ranker
from src.models.dense_ranker import DenseRanker
from src.models.hybrid_rrf import ReciprocalRankFusion

# Configure Streamlit page
st.set_page_config(
    page_title="Amazon ESCI Product Search Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .product-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.1s ease;
    }
    .product-card:hover {
        border-color: #94A3B8;
        background: #F1F5F9;
    }
    .badge-rank {
        background: #2563EB;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-tag {
        color: #059669;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading product catalog and initializing ranking models...")
def load_search_engine():
    """
    Load product catalog and fit all 4 ranking models once, caching in memory.
    """
    benchmark_path = os.path.join("data", "benchmark_sample.parquet")
    if not os.path.exists(benchmark_path):
        st.error(f"Dataset file {benchmark_path} not found. Please run run_evaluation.py first.")
        st.stop()

    df_data = pd.read_parquet(benchmark_path)
    catalog_df = df_data[["product_id", "product_title", "product_brand"]].drop_duplicates(subset=["product_id"]).copy()
    catalog_df["product_title"] = catalog_df["product_title"].fillna("Untitled Product")
    catalog_df["product_brand"] = catalog_df["product_brand"].fillna("Generic")

    corpus = catalog_df["product_title"].tolist()
    product_ids = catalog_df["product_id"].tolist()
    id_to_info = {
        row["product_id"]: {
            "title": row["product_title"],
            "brand": row["product_brand"]
        }
        for _, row in catalog_df.iterrows()
    }

    # Fit 4 models
    tfidf = TFIDFRanker().fit(corpus, product_ids)
    bm25 = BM25Ranker(k1=1.5, b=0.75).fit(corpus, product_ids)
    dense = DenseRanker().fit(corpus, product_ids, batch_size=128, show_progress_bar=False)
    hybrid = ReciprocalRankFusion(rankers=[bm25, dense], weights=[1.0, 1.0], k=60)

    return {
        "tfidf": tfidf,
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "id_to_info": id_to_info,
        "catalog_size": len(corpus)
    }


def main():
    engine = load_search_engine()
    id_to_info = engine["id_to_info"]

    st.markdown('<div class="main-header">🛒 Product Search Engine using Amazon ESCI</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Multi-Paradigm IR Benchmark evaluating <b>TF-IDF</b>, <b>BM25</b>, <b>Dense Sentence Transformers</b>, and <b>Hybrid RRF</b> over a catalog of <b>{engine["catalog_size"]:,}</b> products.</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("🔍 Search Configuration")

    model_choice = st.sidebar.selectbox(
        "Select Ranking Paradigm",
        [
            "Hybrid Model (RRF - Sparse + Dense)",
            "BM25 (Probabilistic Sparse Retrieval)",
            "Sentence Transformer (Dense Semantic Vectors)",
            "TF-IDF (Vector Space Model)",
            "⚡ Side-by-Side Model Comparison (All 4)"
        ],
        index=0
    )

    top_k = st.sidebar.slider("Top Results (Cutoff K)", min_value=3, max_value=20, value=10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Example Queries")
    sample_queries = [
        "wireless bluetooth earbuds with microphone",
        "apple iphone 13 silicone case",
        "men breathable running shoes",
        "gaming mouse rgb high dpi",
        "stainless steel 2 slice toaster",
        "smart android smartphone 128gb"
    ]
    selected_sample = st.sidebar.selectbox("Click to auto-fill query:", ["-- Select an example --"] + sample_queries)

    # Navigation tabs
    tab_search, tab_eval, tab_eda, tab_pipeline = st.tabs([
        "🔎 Live Search",
        "📊 Evaluation & Comparison Table",
        "📈 Dataset & EDA Stats",
        "⚙️ Text Preprocessing Pipeline"
    ])

    # ---------------------------------------------------------
    # TAB 1: LIVE SEARCH
    # ---------------------------------------------------------
    with tab_search:
        col_input, col_btn = st.columns([5, 1])
        initial_val = selected_sample if selected_sample != "-- Select an example --" else "wireless earbuds for running"
        query_input = col_input.text_input("Enter product search query:", value=initial_val, key="query_box")
        search_clicked = col_btn.button("Search 🚀", use_container_width=True)

        if query_input:
            # Display scratch preprocessing inspection
            clean_q = clean_text(query_input)
            toks = tokenize(clean_q)
            no_stop = remove_stopwords(toks)
            stemmed = stem_tokens(no_stop)

            with st.expander("🛠️ Inspect Query Preprocessing (From Scratch)", expanded=False):
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                col_p1.markdown(f"**1. Cleaned**<br>`{clean_q}`", unsafe_allow_html=True)
                col_p2.markdown(f"**2. Tokenized**<br>`{toks}`", unsafe_allow_html=True)
                col_p3.markdown(f"**3. Stop-removed**<br>`{no_stop}`", unsafe_allow_html=True)
                col_p4.markdown(f"**4. Porter Stemmed**<br>`{stemmed}`", unsafe_allow_html=True)

            t0 = time.time()

            if "Side-by-Side" in model_choice:
                st.subheader(f"⚡ Side-by-Side Model Comparison for: *'{query_input}'*")
                c1, c2, c3, c4 = st.columns(4)

                models_to_compare = [
                    ("TF-IDF", engine["tfidf"], c1, "#E67E22"),
                    ("BM25", engine["bm25"], c2, "#2980B9"),
                    ("Dense MiniLM", engine["dense"], c3, "#8E44AD"),
                    ("Hybrid RRF", engine["hybrid"], c4, "#27AE60")
                ]

                for name, model, col, color in models_to_compare:
                    with col:
                        st.markdown(f"<h4 style='color:{color}; text-align:center;'>{name}</h4>", unsafe_allow_html=True)
                        results = model.rank(query_input, top_k=top_k)
                        for rank, (pid, score) in enumerate(results, 1):
                            info = id_to_info.get(pid, {"title": "Unknown Title", "brand": "Generic"})
                            st.markdown(f"""
                            <div class="product-card">
                                <span class="badge-rank" style="background:{color};">#{rank}</span>
                                <span class="score-tag">Score: {score:.4f}</span><br/>
                                <b style="font-size:0.95rem;">{info['title'][:85]}...</b><br/>
                                <small style="color:#64748B;">Brand: {info['brand']} | ASIN: {pid}</small>
                            </div>
                            """, unsafe_allow_html=True)

            else:
                # Single model search
                if "Hybrid" in model_choice:
                    ranker = engine["hybrid"]
                    model_label = "Hybrid RRF (Sparse + Dense)"
                    color = "#27AE60"
                elif "BM25" in model_choice:
                    ranker = engine["bm25"]
                    model_label = "BM25 Probabilistic Retrieval"
                    color = "#2980B9"
                elif "Sentence Transformer" in model_choice:
                    ranker = engine["dense"]
                    model_label = "Sentence Transformer (all-MiniLM-L6-v2)"
                    color = "#8E44AD"
                else:
                    ranker = engine["tfidf"]
                    model_label = "TF-IDF Vector Space"
                    color = "#E67E22"

                results = ranker.rank(query_input, top_k=top_k)
                elapsed = (time.time() - t0) * 1000

                st.markdown(f"Retrieved **{len(results)}** results using **{model_label}** in **{elapsed:.1f} ms**")

                for rank, (pid, score) in enumerate(results, 1):
                    info = id_to_info.get(pid, {"title": "Unknown Title", "brand": "Generic"})
                    st.markdown(f"""
                    <div class="product-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span class="badge-rank" style="background:{color};">Rank #{rank}</span>
                            <span class="score-tag">Relevance Score: {score:.4f}</span>
                        </div>
                        <h4 style="margin:0 0 4px 0; color:#1E293B;">{info['title']}</h4>
                        <div style="font-size:0.85rem; color:#64748B;">
                            <b>Brand:</b> {info['brand']} &nbsp;|&nbsp; <b>ASIN Product ID:</b> <code>{pid}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: EVALUATION & COMPARISON TABLE
    # ---------------------------------------------------------
    with tab_eval:
        st.subheader("📊 Standardized IR Benchmark Performance")
        st.write("Evaluated over real Amazon customer queries against ground-truth ESCI judgments.")

        table_path = os.path.join("evaluation_results", "model_comparison_table.csv")
        plot_path = os.path.join("evaluation_results", "precision_recall_curves.png")

        if os.path.exists(table_path):
            df_metrics = pd.read_csv(table_path)
            st.markdown("### Master Performance Comparison Table")
            st.dataframe(
                df_metrics.style.highlight_max(axis=0, subset=["Precision@10", "Recall", "MAP", "nDCG"], color="#DCFCE7"),
                use_container_width=True
            )

            # Key metric summary cards
            c_p, c_r, c_m, c_n = st.columns(4)
            best_ndcg = df_metrics.loc[df_metrics["nDCG"].idxmax()]
            best_p10 = df_metrics.loc[df_metrics["Precision@10"].idxmax()]

            c_p.metric("Top Precision@10", f"{best_p10['Precision@10']:.4f}", best_p10["Model"])
            c_r.metric("Top Recall@10", f"{best_p10['Recall']:.4f}", best_p10["Model"])
            c_m.metric("Top MAP", f"{best_p10['MAP']:.4f}", best_p10["Model"])
            c_n.metric("Top Graded nDCG", f"{best_ndcg['nDCG']:.4f}", best_ndcg["Model"])

        if os.path.exists(plot_path):
            st.markdown("### Precision-Recall Curves (Across 4 Paradigms)")
            st.image(Image.open(plot_path), caption="11-Point Interpolated Precision-Recall Curves on Amazon ESCI Benchmark", use_container_width=True)

        st.info("""
        **Graded ESCI Relevance Gains**:
        - **Exact (E)** = 3.0: Target exact item.
        - **Substitute (S)** = 2.0: Functional alternative.
        - **Complement (C)** = 1.0: Accessory or add-on.
        - **Irrelevant (I)** = 0.0: Off-topic product.
        """)

    # ---------------------------------------------------------
    # TAB 3: DATASET & EDA
    # ---------------------------------------------------------
    with tab_eda:
        st.subheader("📈 Amazon ESCI Dataset Exploratory Analysis")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Raw Judgments", "2,621,288")
        col_m2.metric("English Task 1 Rows", "601,354")
        col_m3.metric("Unique Customer Queries", "29,844")
        col_m4.metric("Unique Catalog Products", "482,105")

        st.markdown("""
        #### ESCI Class Proportions:
        - **Exact (E)**: 43.5%
        - **Substitute (S)**: 35.1%
        - **Irrelevant (I)**: 16.9%
        - **Complement (C)**: 4.5%
        
        #### Query & Catalog Characteristics:
        - **Median Query Length**: 3 words (90% of queries $\\le$ 5 words).
        - **Average Candidate Depth**: 20.15 products judged per query.
        - **Product Title Availability**: >99.9% of catalog items contain rich descriptive titles.
        
        👉 For interactive cell outputs and visual plots, see: [`notebooks/01_eda_and_data_inspection.ipynb`](file:///C:/Users/rohit/Desktop/Information%20Retrival%20Project/notebooks/01_eda_and_data_inspection.ipynb).
        """)

    # ---------------------------------------------------------
    # TAB 4: PREPROCESSING & ARCHITECTURE PIPELINE
    # ---------------------------------------------------------
    with tab_pipeline:
        st.subheader("⚙️ Scratch-Built Preprocessing & Fusion Architecture")
        st.markdown("""
        ### Preprocessing Pipeline:
        1. **HTML & Entity Cleaning**: Strips `<p>`, `<b>`, `<div>`, unescapes `&amp;`, normalizes Unicode.
        2. **Alphanumeric Tokenization**: `\b[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*\b` preserving specs like `128gb`, `4k`, `g502`.
        3. **Stop-word Removal**: Prunes English stop words while keeping product intent.
        4. **Porter Stemming**: Collapses inflectional variants (`running` -> `run`, `headphones` -> `headphon`).

        ### Reciprocal Rank Fusion (RRF) Formula:
        $$RRF\\_Score(d) = \\sum_{m \\in M} \\frac{w_m}{k + r_m(d)}$$
        where $k = 60$, $r_m(d)$ is document rank in model $m$, and $w_m = 1.0$.
        """)


if __name__ == "__main__":
    main()
