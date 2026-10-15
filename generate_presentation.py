"""
generate_presentation.py
------------------------
Generates a complete, high-quality PowerPoint presentation (.pptx)
for Project 14: Product Search using Amazon ESCI.
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    PRIMARY = RGBColor(30, 27, 75)      # Deep Indigo #1E1B4B
    SECONDARY = RGBColor(79, 70, 229)   # Indigo #4F46E5
    TEXT_DARK = RGBColor(15, 23, 42)    # Slate 900
    TEXT_MUTED = RGBColor(100, 116, 139)# Slate 500
    ACCENT_GREEN = RGBColor(16, 185, 129)# Emerald
    BG_LIGHT = RGBColor(248, 250, 252)  # Slate 50
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(226, 232, 240)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="PROJECT 14: E-COMMERCE SEARCH"):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = SECONDARY

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = PRIMARY

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    
    # Title text box
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "PROJECT 14: E-COMMERCE SEARCH"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = SECONDARY
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "Product Search using Amazon ESCI"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = PRIMARY
    p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "Multi-Paradigm IR Benchmark: TF-IDF, BM25, Sentence Transformers, and Hybrid RRF"
    p2.font.size = Pt(18)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_after = Pt(28)

    p3 = tf1.add_paragraph()
    p3.text = "Information Retrieval Course Project  |  Evaluation: Precision@10, Recall, MAP, nDCG"
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 2: Problem Statement & ESCI Graded Relevance
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Goal & Dataset Overview: Amazon ESCI Benchmark")

    # Left card: Goal
    left_box = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0))
    tf_l = left_box.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Problem Statement"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    bullets_l = [
        "In e-commerce search, keyword matching alone fails to capture customer intent.",
        "Amazon ESCI (Shopping Queries Dataset) is the gold standard benchmark released for KDD Cup 2022.",
        "Contains difficult customer search queries paired with human relevance judgments.",
        "Our mission: Build an end-to-end IR system evaluated under 4 standardized retrieval paradigms from raw data."
    ]
    for b in bullets_l:
        p = tf_l.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # Right: The 4 ESCI Levels
    right_box = s2.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_r = right_box.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "The 4 Graded Relevance Levels (ESCI)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    esci_cards = [
        ("[E] Exact Match (Gain: 3.0)", "The item is exactly what the customer searched for. (e.g., 'Nike shoes 10' -> exact shoe & size)"),
        ("[S] Substitute (Gain: 2.0)", "A functional replacement if the exact item is unavailable. (e.g., 'iPhone 13' -> iPhone 14)"),
        ("[C] Complement (Gain: 1.0)", "An accessory or add-on product. (e.g., 'iPhone 13' -> iPhone silicone case or charger)"),
        ("[I] Irrelevant (Gain: 0.0)", "Completely off-topic product that fails user intent. (e.g., 'iPhone 13' -> cat food)")
    ]
    for title, desc in esci_cards:
        p_t = tf_r.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = SECONDARY
        p_d = tf_r.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11.5)
        p_d.font.color.rgb = TEXT_DARK
        p_d.space_after = Pt(6)

    # =========================================================================
    # SLIDE 3: Exploratory Data Analysis (EDA) Insights
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Exploratory Data Analysis: Real-World Dataset Scale")

    # Metrics grid on top
    stat_cards = [
        ("Total Raw Judgments", "2,621,288", "Spans US, Japan, Spain"),
        ("English Task 1 Benchmark", "601,354", "small_version == 1 & us"),
        ("Unique Search Queries", "29,844", "419k Train / 181k Test"),
        ("Unique Catalog Products", "482,105", ">99.9% have rich titles")
    ]
    for idx, (label, val, sub) in enumerate(stat_cards):
        left_pos = Inches(0.8 + idx * 2.95)
        shape = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.8), Inches(2.8), Inches(1.3))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = BORDER_COLOR
        
        tf_s = shape.text_frame
        tf_s.word_wrap = True
        p = tf_s.paragraphs[0]
        p.text = val
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = SECONDARY
        p.alignment = PP_ALIGN.CENTER

        p_sub = tf_s.add_paragraph()
        p_sub.text = label
        p_sub.font.size = Pt(11)
        p_sub.font.bold = True
        p_sub.font.color.rgb = TEXT_DARK
        p_sub.alignment = PP_ALIGN.CENTER

    # Bottom analysis text
    bot_box = s3.shapes.add_textbox(Inches(0.8), Inches(3.4), Inches(11.7), Inches(3.5))
    tf_b = bot_box.text_frame
    tf_b.word_wrap = True

    eda_points = [
        ("ESCI Class Distribution", "Exact (43.5%) and Substitute (35.1%) make up ~78.6% of annotations. Irrelevant is 16.9%, Complement is 4.5%. Graded nDCG is vital to reward placing Exact over Substitute."),
        ("Search Query Characteristics", "Customer queries are short and concise: Median = 3 words (90% of queries <= 5 words). Exact keyword overlap alone leaves a semantic gap that dense models must bridge."),
        ("Product Catalog Richness", "Product titles average ~15 words and contain critical brand names, model numbers, and technical specifications, making them the ideal primary feature for high-speed indexing."),
        ("Dedicated Jupyter Notebook", "All distributions, quartiles, and Seaborn charts are pre-rendered in 'notebooks/01_eda_and_data_inspection.ipynb' for full reproducibility.")
    ]
    for heading, detail in eda_points:
        p = tf_b.add_paragraph()
        p.text = "• " + heading + ": "
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = PRIMARY
        run = p.add_run()
        run.text = detail
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 4: Text Preprocessing Pipeline (From Scratch)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Text Preprocessing Pipeline (Built 100% From Scratch)")

    steps = [
        ("1. HTML & Noise Stripping", "Strips tags (<p>, <b>, <div>), unescapes entities (&amp; -> &), normalizes Unicode NFKD, lowercases, cleans delimiters."),
        ("2. Alphanumeric Tokenizer", "Regex-based token extraction that preserves hardware specs and dimensions ('128gb', '4k', '12.9', 'g502')."),
        ("3. Stop-Word Removal", "Prunes high-frequency English noise tokens ('the', 'for', 'in') while preserving e-commerce search intent."),
        ("4. Porter Stemming", "Applies Porter's algorithmic suffix-stripping rules: 'running' -> 'run', 'headphones' -> 'headphon', 'tested' -> 'test'.")
    ]
    for idx, (s_title, s_desc) in enumerate(steps):
        top_pos = Inches(1.8 + idx * 1.05)
        shape = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_pos, Inches(11.7), Inches(0.95))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = BORDER_COLOR
        
        tf_st = shape.text_frame
        tf_st.word_wrap = True
        p = tf_st.paragraphs[0]
        p.text = s_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = SECONDARY
        
        p2 = tf_st.add_paragraph()
        p2.text = s_desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_DARK

    # Verification banner
    v_box = s4.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8))
    tf_v = v_box.text_frame
    p = tf_v.paragraphs[0]
    p.text = "✔ Automated Test Suite: 7 unit tests in tests/test_preprocessing.py pass with 100% OK in 0.003s."
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    # =========================================================================
    # SLIDE 5: The 4 Core Ranking Paradigms
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "The Four Core Ranking Paradigms Evaluated")

    model_cards = [
        ("Model 1: TF-IDF (Vector Space)", "Classical VSM with Cosine Similarity", [
            "Computes Term Frequency * Inverse Document Frequency.",
            "Sublinear scaling: tf = 1 + log(tf).",
            "Cosine similarity against high-dimensional sparse query vector.",
            "Fast, strict baseline for exact keyword matches."
        ]),
        ("Model 2: BM25 (Probabilistic)", "Okapi BM25 Ranking Framework", [
            "Term frequency saturation with constant k1 = 1.5.",
            "Document length normalization with constant b = 0.75.",
            "Penalizes bloated titles while rewarding rare query terms.",
            "Gold standard sparse IR baseline."
        ]),
        ("Model 3: Sentence Transformer", "Dense Semantic Vector Retrieval", [
            "Encodes text using 'all-MiniLM-L6-v2' into 384-dim space.",
            "L2-normalized vectors (inner product = cosine similarity).",
            "Bridges vocabulary mismatch (e.g. 'jogging' matches 'running').",
            "Infers category semantics without exact lexical overlap."
        ]),
        ("Model 4: Hybrid RRF", "Reciprocal Rank Fusion (Sparse + Dense)", [
            "Formula: RRF_Score(d) = sum [ w_m / (k + rank_m(d)) ].",
            "Smoothing constant k = 60; weights = 1.0 each for BM25 and Dense.",
            "Solves score-mismatch problem (BM25 log-odds vs Cosine).",
            "Combines keyword precision with semantic depth."
        ])
    ]

    for idx, (m_name, m_sub, bullets) in enumerate(model_cards):
        col_idx = idx % 2
        row_idx = idx // 2
        left_pos = Inches(0.8 + col_idx * 5.9)
        top_pos = Inches(1.8 + row_idx * 2.5)

        shape = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos, Inches(5.7), Inches(2.35))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = BORDER_COLOR

        tf_m = shape.text_frame
        tf_m.word_wrap = True
        p = tf_m.paragraphs[0]
        p.text = m_name
        p.font.size = Pt(13.5)
        p.font.bold = True
        p.font.color.rgb = SECONDARY

        p_sub = tf_m.add_paragraph()
        p_sub.text = m_sub
        p_sub.font.size = Pt(10.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.space_after = Pt(4)

        for b in bullets:
            pb = tf_m.add_paragraph()
            pb.text = "• " + b
            pb.font.size = Pt(10)
            pb.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 6: Master Performance Comparison Table
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Quantitative Benchmark: Standardized Comparison Table")

    # Table creation
    rows = 5
    cols = 5
    left = Inches(0.8)
    top = Inches(1.8)
    width = Inches(11.7)
    height = Inches(2.5)

    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table_data = [
        ["Ranking Model Paradigm", "Precision@10", "Recall@10", "MAP", "Graded nDCG"],
        ["TF-IDF (Vector Space)", "0.8364", "0.5992", "0.8804", "0.7812"],
        ["BM25 (Probabilistic Okapi)", "0.8372", "0.5994", "0.8794", "0.7815"],
        ["Sentence Transformer (MiniLM)", "0.8516 (Best)", "0.6115 (Best)", "0.8932 (Best)", "0.7923"],
        ["Hybrid (RRF - Sparse + Dense)", "0.8480", "0.6097", "0.8896", "0.7984 (Best)"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = PRIMARY
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(241, 245, 249)
            elif "Best" in val:
                p.font.bold = True
                p.font.color.rgb = RGBColor(21, 128, 61) # Green
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(220, 252, 231)
            else:
                p.font.color.rgb = TEXT_DARK

    # Insights box below
    ins_box = s6.shapes.add_textbox(Inches(0.8), Inches(4.7), Inches(11.7), Inches(2.2))
    tf_ins = ins_box.text_frame
    tf_ins.word_wrap = True

    p = tf_ins.paragraphs[0]
    p.text = "Key Empirical Takeaways:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(6)

    takeaways = [
        "Hybrid RRF wins on Graded nDCG (0.7984): Fusing lexical precision (BM25) with semantic depth puts Exact matches (E=3) at top rank.",
        "Sentence Transformer dominates Precision@10 (0.8516) and MAP (0.8932): Overcomes synonymy and complex phrasing.",
        "BM25 outperforms TF-IDF on precision and nDCG: Document length normalization avoids penalizing descriptive product titles.",
        "Strict ESCI Gain Alignment: Relevance scheme (E=3, S=2, C=1, I=0) rewards models that prioritize exact replacements."
    ]
    for t in takeaways:
        p = tf_ins.add_paragraph()
        p.text = "✔ " + t
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 7: Precision-Recall Curves
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "11-Point Interpolated Precision-Recall Curves")

    # Check if plot image exists and embed
    pr_img_path = os.path.join("evaluation_results", "precision_recall_curves.png")
    if os.path.exists(pr_img_path):
        s7.shapes.add_picture(pr_img_path, Inches(0.8), Inches(1.8), Inches(6.8), Inches(5.0))

    # Right explanation box
    r_curve_box = s7.shapes.add_textbox(Inches(7.8), Inches(1.8), Inches(4.7), Inches(5.0))
    tf_c = r_curve_box.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Curve Diagnostics"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)

    c_points = [
        ("High-Precision Zone (Recall 0.0 - 0.4)", "Dense Sentence Transformer maintains precision > 0.88 by understanding contextual synonyms immediately."),
        ("Mid-Recall Cross-over (Recall 0.4 - 0.7)", "Hybrid RRF tracks the upper envelope of both sparse and dense paradigms, avoiding lexical drop-offs."),
        ("Tail Recall Behavior", "All four paradigms converge smoothly as recall approaches 1.0, proving consistent retrieval stability across queries."),
        ("Standardized Protocol", "Computed using standard 11-point interpolation: p_interp(r) = max_{r' >= r} p(r').")
    ]
    for h, d in c_points:
        p = tf_c.add_paragraph()
        p.text = "• " + h + ":"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = SECONDARY
        p_sub = tf_c.add_paragraph()
        p_sub.text = d
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = TEXT_DARK
        p_sub.space_after = Pt(6)

    # =========================================================================
    # SLIDE 8: System Architecture & Decoupled Web App
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Production Architecture: Decoupled FastAPI + SPA")

    arch_left = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_al = arch_left.text_frame
    tf_al.word_wrap = True

    p = tf_al.paragraphs[0]
    p.text = "Backend: FastAPI REST Engine"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)

    backend_features = [
        "Zero Reload Overhead: Precomputed 384-dim embedding cache (data/embeddings_cache.npy) boots in 0.02s.",
        "Clean REST Contracts: /api/search, /api/compare, /api/metrics, /api/eda.",
        "Interactive Swagger Docs: Built-in OpenAPI explorer at /docs for team collaboration.",
        "CORS Enabled: Ready for any frontend stack (React, Next.js, Vue, or Vanilla JS)."
    ]
    for bf in backend_features:
        p = tf_al.add_paragraph()
        p.text = "• " + bf
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    arch_right = s8.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_ar = arch_right.text_frame
    tf_ar.word_wrap = True

    p = tf_ar.paragraphs[0]
    p.text = "Frontend: Ultra-Responsive Web UI"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)

    frontend_features = [
        "Live Search & Model Switcher: Instant query testing across all 4 paradigms.",
        "Side-by-Side 4-Model Comparison: Evaluates all 4 paradigms concurrently in parallel columns.",
        "Scratch Preprocessing Inspector: Live visualization of HTML cleaning, tokens, stop-words, and Porter stems.",
        "Sub-second Latency: Instant rendering with zero build-tool bloat."
    ]
    for ff in frontend_features:
        p = tf_ar.add_paragraph()
        p.text = "• " + ff
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 9: Conclusion & Summary
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Conclusion & Project Deliverables Summary")

    conc_box = s9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_cn = conc_box.text_frame
    tf_cn.word_wrap = True

    p = tf_cn.paragraphs[0]
    p.text = "All Teacher Requirements Fully Satisfied:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    summary_items = [
        ("True Official Dataset", "Ingested untouched Amazon ESCI dataset with 2.6M raw judgments."),
        ("Dedicated EDA Notebook", "Pre-rendered 01_eda_and_data_inspection.ipynb with class balance & distributions."),
        ("100% From-Scratch Preprocessing", "Custom cleaning, tokenization, stop-words, and Porter stemming (7/7 unit tests passing)."),
        ("4 Distinct Ranking Paradigms", "TF-IDF, BM25, Sentence Transformer, and Hybrid RRF (5/5 unit tests passing)."),
        ("Standardized IR Evaluation", "Precision@10, Recall, MAP, nDCG, and 11-point Precision-Recall curve graph."),
        ("Interactive Search Interface", "Decoupled FastAPI + lightweight Web UI with side-by-side model comparison.")
    ]
    for title, detail in summary_items:
        p = tf_cn.add_paragraph()
        p.text = "✔ " + title + ": "
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = SECONDARY
        r = p.add_run()
        r.text = detail
        r.font.bold = False
        r.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    output_pptx = "Project14_Amazon_ESCI_Presentation.pptx"
    prs.save(output_pptx)
    print(f"PowerPoint Presentation saved to: {os.path.abspath(output_pptx)}")


if __name__ == "__main__":
    create_deck()
