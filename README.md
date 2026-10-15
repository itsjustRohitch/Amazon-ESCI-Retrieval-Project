# Project 14: Product Search using Amazon ESCI
> **Standardized Information Retrieval Pipeline & Multi-Paradigm Benchmark**

Built as part of the Information Retrieval (IR) curriculum under the **Unified Project Template Structure**, evaluating four distinct ranking paradigms on the real-world **Amazon ESCI (Shopping Queries) Dataset**.

---

## 🚀 Key Features & Highlights

1. **100% From Scratch Text Preprocessing**:
   - Product title cleaning (HTML stripping, entity unescaping, unicode normalization).
   - Alphanumeric-aware tokenization preserving hardware specs (`128gb`, `4k`, `g502`).
   - Stop-word removal filtering linguistic noise while retaining e-commerce intent.
   - Morphological stemming using the Porter Stemmer algorithm.

2. **The 4 Core Ranking Paradigms**:
   - **TF-IDF**: Classical Vector Space Model with sublinear TF scaling and Cosine Similarity.
   - **BM25**: Probabilistic relevance framework with term saturation ($k_1 = 1.5$) and document length normalization ($b = 0.75$).
   - **Dense Sentence Transformer**: Dense semantic vector retrieval using `all-MiniLM-L6-v2` (384-dimensional embeddings) to solve vocabulary mismatch.
   - **Hybrid RRF**: Reciprocal Rank Fusion blending sparse (BM25) and dense (Transformer) rankings with rank discount constant $k=60$.

3. **Rigorous Evaluation Engine**:
   - Evaluated across real Amazon customer queries against ground-truth ESCI judgments.
   - Metrics: **Precision@10**, **Recall**, **MAP**, and **Graded nDCG** ($E=3, S=2, C=1, I=0$).
   - 11-point interpolated **Precision-Recall Curves** plotted across all 4 models on a single graph.

4. **Standalone EDA Notebook**:
   - Pre-rendered [`notebooks/01_eda_and_data_inspection.ipynb`](notebooks/01_eda_and_data_inspection.ipynb) containing comprehensive distributions of the 2.6M raw judgments, 601k English Task 1 rows, ESCI class balance, and catalog properties.

5. **Interactive Search Web UI**:
   - Fast Streamlit web application (`app.py`) with real-time query search, live scratch-preprocessing inspector, side-by-side 4-model comparison, and benchmark dashboards.

---

## 📊 Evaluation Results (Standardized Comparison Table)

Evaluated over representative test queries from the official Amazon ESCI Task 1 benchmark:

| Ranking Model | Precision@10 | Recall@10 | MAP | Graded nDCG |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF** (Vector Space) | 0.8364 | 0.5992 | 0.8804 | 0.7812 |
| **BM25** (Probabilistic) | 0.8372 | 0.5994 | 0.8794 | 0.7815 |
| **Sentence Transformer** (MiniLM) | **0.8516** | **0.6115** | **0.8932** | 0.7923 |
| **Hybrid (RRF)** (Sparse + Dense) | 0.8480 | 0.6097 | 0.8896 | **0.7984** |

> **Key Finding**: **Hybrid RRF achieved the highest graded nDCG (0.7984)**, demonstrating that combining the exact keyword precision of BM25 with the conceptual understanding of Sentence Transformers consistently pushes Exact matches to the very top rank.

The generated Precision-Recall curve graph is saved at:
[`evaluation_results/precision_recall_curves.png`](evaluation_results/precision_recall_curves.png)

---

## 📁 Repository Structure

```text
Information Retrival Project/
├── Amazon-ESCI/                            # Raw untouched official dataset
│   └── shopping_queries_dataset/
│       ├── shopping_queries_dataset_examples.parquet
│       ├── shopping_queries_dataset_products.parquet
│       └── shopping_queries_dataset_sources.csv
├── data/
│   └── benchmark_sample.parquet            # Curated English Task 1 benchmark
├── notebooks/
│   └── 01_eda_and_data_inspection.ipynb    # Pre-rendered EDA notebook with charts & stats
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Cleaning, tokenization, stop-words, stemming
│   ├── evaluation.py                       # Precision@10, Recall, MAP, nDCG, PR-curves
│   └── models/
│       ├── __init__.py
│       ├── tfidf_ranker.py                 # Model 1: TF-IDF
│       ├── bm25_ranker.py                  # Model 2: BM25
│       ├── dense_ranker.py                 # Model 3: Sentence Transformer
│       └── hybrid_rrf.py                   # Model 4: Reciprocal Rank Fusion
├── tests/
│   ├── test_preprocessing.py               # Unit tests for text pipeline (7/7 passing)
│   └── test_models.py                      # Unit tests for all 4 ranking models (5/5 passing)
├── evaluation_results/
│   ├── model_comparison_table.csv          # Master performance metrics CSV
│   └── precision_recall_curves.png         # Precision-Recall curves plot
├── app.py                                  # Streamlit interactive search UI
├── run_evaluation.py                       # Master benchmark runner script
├── requirements.txt                        # Dependency specifications
└── README.md                               # Project documentation
```

---

## ⚙️ How to Run & Verify

### 1. Run Automated Unit Tests
```bash
# Test the scratch preprocessing pipeline
python -m unittest tests/test_preprocessing.py

# Test all 4 ranking models
python -m unittest tests/test_models.py
```

### 2. Run the Benchmark Evaluation
```bash
python run_evaluation.py
```
This evaluates the models, outputs the comparison table in terminal, and generates `evaluation_results/precision_recall_curves.png` and `evaluation_results/model_comparison_table.csv`.

### 3. Launch the Interactive Web Application & REST API
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Open **`http://localhost:8000`** in your browser for the dedicated, ultra-responsive frontend web app.
- Open **`http://localhost:8000/docs`** for the interactive Swagger OpenAPI REST documentation.

### REST Endpoints Available for Frontend Integration:
- `GET /api/search?q={query}&model={hybrid|bm25|dense|tfidf}&k={10}`: Search catalog with live preprocessing inspection.
- `GET /api/compare?q={query}&k={5}`: Side-by-side 4-model comparison on the exact same query.
- `GET /api/metrics`: Precomputed standardized evaluation comparison table.
- `GET /api/eda`: Amazon ESCI dataset distribution summary.
- `GET /health`: Engine status and catalog availability.

### 4. View EDA Notebook
Open `notebooks/01_eda_and_data_inspection.ipynb` in VS Code or Jupyter Notebook to inspect pre-rendered distributions and charts.
