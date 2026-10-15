// frontend/app.js
const API_BASE = ""; // Relative path to FastAPI server

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabViews = document.querySelectorAll(".tab-view");

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      navButtons.forEach(b => b.classList.remove("active"));
      tabViews.forEach(v => v.style.display = "none");

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).style.display = "block";

      if (targetId === "tab-eval") {
        loadEvaluationMetrics();
      } else if (targetId === "tab-eda") {
        loadEDAMetrics();
      }
    });
  });

  // Search execution
  const searchInput = document.getElementById("search-input");
  const searchBtn = document.getElementById("search-btn");
  const modelSelect = document.getElementById("model-select");
  const topkSelect = document.getElementById("topk-select");

  searchBtn.addEventListener("click", () => triggerSearch());
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") triggerSearch();
  });

  // Sample Query Pills
  document.querySelectorAll(".sample-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      searchInput.value = pill.textContent.trim();
      triggerSearch();
    });
  });

  // Compare Tab Button
  const compareBtn = document.getElementById("compare-btn");
  const compareInput = document.getElementById("compare-input");
  compareBtn.addEventListener("click", () => triggerCompare());
  compareInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") triggerCompare();
  });

  // Initial load
  triggerSearch();
});

async function triggerSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;

  const model = document.getElementById("model-select").value;
  const top_k = document.getElementById("topk-select").value;
  const resultsContainer = document.getElementById("results-list");
  const prepContainer = document.getElementById("prep-drawer");
  const latencyBadge = document.getElementById("latency-badge");
  const countBadge = document.getElementById("count-badge");

  resultsContainer.innerHTML = `<div style="text-align:center; padding:30px; color:#64748b;">Searching catalog...</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}&model=${encodeURIComponent(model)}&k=${top_k}`);
    if (!res.ok) throw new Error("Search request failed");
    const data = await res.json();

    // Render Preprocessing
    document.getElementById("prep-cleaned").textContent = data.preprocessing.cleaned || "(empty)";
    document.getElementById("prep-tokenized").textContent = JSON.stringify(data.preprocessing.tokenized);
    document.getElementById("prep-nostop").textContent = JSON.stringify(data.preprocessing.stopwords_removed);
    document.getElementById("prep-stemmed").textContent = JSON.stringify(data.preprocessing.stemmed);
    prepContainer.style.display = "block";

    // Update Status
    latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
    countBadge.textContent = `Showing top ${data.total_results} results (${data.model})`;

    // Render Products
    if (data.results.length === 0) {
      resultsContainer.innerHTML = `<div style="text-align:center; padding:30px; color:#64748b;">No matching products found.</div>`;
      return;
    }

    resultsContainer.innerHTML = data.results.map(item => `
      <div class="product-card">
        <div class="card-top">
          <span class="rank-badge">Rank #${item.rank}</span>
          <span class="score-badge">Score: ${item.score.toFixed(4)}</span>
        </div>
        <div class="product-title">${escapeHtml(item.title)}</div>
        <div class="product-meta">
          <span>Brand: <b>${escapeHtml(item.brand)}</b></span> &nbsp;•&nbsp;
          <span>ASIN: <code>${item.product_id}</code></span>
        </div>
      </div>
    `).join("");

  } catch (err) {
    resultsContainer.innerHTML = `<div style="color:#ef4444; padding:20px;">Error: ${err.message}</div>`;
  }
}

async function triggerCompare() {
  const query = document.getElementById("compare-input").value.trim();
  if (!query) return;

  const compareGrid = document.getElementById("compare-grid");
  const latencySpan = document.getElementById("compare-latency");
  compareGrid.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:30px; color:#64748b;">Evaluating all 4 ranking models...</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/compare?q=${encodeURIComponent(query)}&k=5`);
    if (!res.ok) throw new Error("Comparison request failed");
    const data = await res.json();

    latencySpan.textContent = `Completed all 4 models in ${data.latency_ms} ms`;

    const modelConfigs = [
      { key: "TF-IDF", color: "#e67e22" },
      { key: "BM25", color: "#2980b9" },
      { key: "Sentence Transformer", color: "#8e44ad" },
      { key: "Hybrid (RRF)", color: "#10b981" }
    ];

    compareGrid.innerHTML = modelConfigs.map(cfg => {
      const items = data.rankings[cfg.key] || [];
      const cardHtml = items.map(item => `
        <div style="background:white; border:1px solid #e2e8f0; border-radius:8px; padding:10px; margin-bottom:8px; font-size:0.85rem;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="background:${cfg.color}; color:white; padding:1px 6px; border-radius:4px; font-weight:700; font-size:0.75rem;">#${item.rank}</span>
            <span style="color:#059669; font-weight:700; font-size:0.75rem;">${item.score.toFixed(4)}</span>
          </div>
          <div style="font-weight:600; color:#1e293b; line-height:1.3; margin-bottom:4px;">${escapeHtml(item.title.substring(0, 75))}...</div>
          <div style="color:#64748b; font-size:0.75rem;">${escapeHtml(item.brand)} | ${item.product_id}</div>
        </div>
      `).join("");

      return `
        <div class="compare-col">
          <div class="col-header" style="color:${cfg.color}; border-color:${cfg.color};">${cfg.key}</div>
          ${cardHtml}
        </div>
      `;
    }).join("");

  } catch (err) {
    compareGrid.innerHTML = `<div style="grid-column:1/-1; color:#ef4444; padding:20px;">Error: ${err.message}</div>`;
  }
}

async function loadEvaluationMetrics() {
  const tableContainer = document.getElementById("metrics-table-container");
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    const data = await res.json();
    const rows = data.metrics_table;

    // Find max values
    const maxP10 = Math.max(...rows.map(r => r["Precision@10"]));
    const maxRec = Math.max(...rows.map(r => r["Recall"]));
    const maxMap = Math.max(...rows.map(r => r["MAP"]));
    const maxNdcg = Math.max(...rows.map(r => r["nDCG"]));

    let html = `
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Ranking Model</th>
            <th>Precision@10</th>
            <th>Recall@10</th>
            <th>MAP</th>
            <th>Graded nDCG</th>
          </tr>
        </thead>
        <tbody>
    `;

    rows.forEach(r => {
      html += `
        <tr>
          <td><b>${escapeHtml(r.Model)}</b></td>
          <td><span class="${r['Precision@10'] === maxP10 ? 'best-val' : ''}">${r['Precision@10'].toFixed(4)}</span></td>
          <td><span class="${r['Recall'] === maxRec ? 'best-val' : ''}">${r['Recall'].toFixed(4)}</span></td>
          <td><span class="${r['MAP'] === maxMap ? 'best-val' : ''}">${r['MAP'].toFixed(4)}</span></td>
          <td><span class="${r['nDCG'] === maxNdcg ? 'best-val' : ''}">${r['nDCG'].toFixed(4)}</span></td>
        </tr>
      `;
    });

    html += `</tbody></table>`;
    tableContainer.innerHTML = html;
  } catch (err) {
    tableContainer.innerHTML = `<p style="color:#ef4444;">Failed to load metrics table.</p>`;
  }
}

async function loadEDAMetrics() {
  const edaContainer = document.getElementById("eda-stats-container");
  try {
    const res = await fetch(`${API_BASE}/api/eda`);
    const data = await res.json();
    edaContainer.innerHTML = `
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin-bottom:24px;">
        <div style="background:white; padding:16px; border-radius:10px; border:1px solid #e2e8f0; text-align:center;">
          <div style="color:#64748b; font-size:0.85rem;">Total Raw Judgments</div>
          <div style="font-size:1.6rem; font-weight:800; color:#1e1b4b;">${data.raw_total_judgments.toLocaleString()}</div>
        </div>
        <div style="background:white; padding:16px; border-radius:10px; border:1px solid #e2e8f0; text-align:center;">
          <div style="color:#64748b; font-size:0.85rem;">English Task 1 Rows</div>
          <div style="font-size:1.6rem; font-weight:800; color:#1e1b4b;">${data.english_task1_judgments.toLocaleString()}</div>
        </div>
        <div style="background:white; padding:16px; border-radius:10px; border:1px solid #e2e8f0; text-align:center;">
          <div style="color:#64748b; font-size:0.85rem;">Unique Queries</div>
          <div style="font-size:1.6rem; font-weight:800; color:#1e1b4b;">${data.unique_queries.toLocaleString()}</div>
        </div>
        <div style="background:white; padding:16px; border-radius:10px; border:1px solid #e2e8f0; text-align:center;">
          <div style="color:#64748b; font-size:0.85rem;">Unique Products</div>
          <div style="font-size:1.6rem; font-weight:800; color:#1e1b4b;">${data.unique_products.toLocaleString()}</div>
        </div>
      </div>
    `;
  } catch (err) {
    edaContainer.innerHTML = `<p style="color:#ef4444;">Failed to load EDA stats.</p>`;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
