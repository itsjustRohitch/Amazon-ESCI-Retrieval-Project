// Shelf: search interface for the ESCI project.
// Plain JS, no build step. Routes: #/search, #/compare, #/evaluation, #/index

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));

const MODELS = { tfidf: "TF-IDF", bm25: "BM25", dense: "Sentence Transformer", hybrid: "Hybrid (RRF)" };
const SERIES = { "TF-IDF": "tfidf", "BM25": "bm25", "Sentence Transformer": "dense", "Hybrid (RRF)": "hybrid", "Random": "random" };
const LABEL = { E: "Exact", S: "Substitute", C: "Complement", I: "Irrelevant" };

const st = { view: "search", q: "", model: "bm25", evalProto: "rerank", evalCache: null, indexCache: null };

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function lab(l) {
  return l ? `<span class="lab lab-${l}" title="${LABEL[l]}">${l}</span>`
           : `<span class="lab lab-none" title="Not judged for this query">·</span>`;
}

// ---------------------------------------------------------------- routing

function readHash() {
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const p = new URLSearchParams(qs || "");
  st.view = ["search", "compare", "evaluation", "index"].includes(path) ? path : "search";
  st.q = p.get("q") || "";
  if (MODELS[p.get("m")]) st.model = p.get("m");
  if (["light", "dark"].includes(p.get("theme"))) document.documentElement.dataset.theme = p.get("theme");
}

function writeHash() {
  const p = new URLSearchParams();
  if (st.q) p.set("q", st.q);
  if (st.view === "search" && st.q) p.set("m", st.model);
  const h = `#/${st.view}${p.toString() ? "?" + p : ""}`;
  if (location.hash !== h) history.pushState(null, "", h);
}

function render() {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === `view-${st.view}`));
  document.querySelectorAll(".nav a").forEach((a) => {
    a.classList.toggle("on", a.dataset.view === st.view);
    // carry the current query between Search and Compare
    const keep = st.q && (a.dataset.view === "search" || a.dataset.view === "compare");
    a.href = `#/${a.dataset.view}${keep ? `?q=${encodeURIComponent(st.q)}` : ""}`;
  });
  document.querySelectorAll(".models button").forEach((b) => b.classList.toggle("on", b.dataset.model === st.model));
  document.body.classList.toggle("hide-query", st.view === "evaluation" || st.view === "index");
  $("#q").value = st.q;
  ({ search: renderSearch, compare: renderCompare, evaluation: renderEvaluation, index: renderIndex })[st.view]();
}

// ---------------------------------------------------------------- search

async function renderSearch() {
  if (!st.q) return renderEmpty();
  $("#empty").innerHTML = "";
  const q = st.q, model = st.model;
  let d;
  try { d = await api(`/api/search?q=${encodeURIComponent(q)}&model=${model}&k=20`); }
  catch (e) { $("#hits").innerHTML = `<li class="err">${esc(e.message)}</li>`; return; }
  if (q !== st.q || model !== st.model) return; // a newer request has started

  const judged = d.judged
    ? ` · judged query, ${d.judged} labelled products · nDCG@10 <b>${fmt(d.ndcg10)}</b>`
    : " · not a benchmark query, no labels";
  $("#meta").innerHTML = `${d.results.length} results in <b>${fmt(d.latency_ms, 1)} ms</b>${judged}`;

  const maxTerm = Math.max(1e-9, ...d.results.flatMap((r) => Object.values(r.why?.terms || {})));
  $("#hits").innerHTML = d.results.map((r) => `
    <li class="hit">
      <span class="rank">${String(r.rank).padStart(2, "0")}</span>
      <div>
        <div class="title">${esc(r.title)}</div>
        <div class="sub">${r.brand ? esc(r.brand) + " · " : ""}<span class="pid">${r.product_id}</span></div>
        ${why(r, maxTerm)}
      </div>
      <div class="side">${d.judged ? lab(r.label) : ""}<span class="score">${fmt(r.score, model === "hybrid" ? 5 : 3)}</span></div>
    </li>`).join("");

  $("#aside").innerHTML = asideHtml(d.preprocessing, d.params, model);
}

function why(r, maxTerm) {
  if (!r.why) return "";
  if (r.why.terms) {
    return `<div class="why">${Object.entries(r.why.terms).map(([t, v]) => `
      <span class="t${v ? "" : " zero"}">${esc(t)}<span class="bar" style="width:${Math.round(48 * v / maxTerm)}px"></span>${fmt(v, 2)}</span>`).join("")}</div>`;
  }
  const b = r.why.bm25_rank, dn = r.why.dense_rank;
  return `<div class="why"><span class="t">BM25 rank ${b ?? "n/a"}</span><span class="t">Dense rank ${dn ?? "n/a"}</span></div>`;
}

function asideHtml(p, params, model) {
  const tokens = p.tokens.map((t) => `<span class="chip${t.removed ? " off" : ""}">${esc(t.text)}</span>`).join("") || `<span class="note">none</span>`;
  const stems = p.stems.map((t) => `<span class="chip">${esc(t)}</span>`).join("") || `<span class="note">none</span>`;
  const terms = p.terms.map((t) => `
    <tr class="${t.df ? "" : "miss"}"><td>${esc(t.term)}</td><td class="n">${t.df.toLocaleString()}</td><td class="n">${t.df ? fmt(t.idf, 2) : "not in index"}</td></tr>`).join("");
  const setting = {
    tfidf: `<dt>tf</dt><dd>${params.tfidf.sublinear_tf ? "1 + ln f" : "raw count"}</dd><dt>idf</dt><dd>${params.tfidf.idf ? "ln((1+N)/(1+df)) + 1" : "off"}</dd><dt>score</dt><dd>cosine</dd>`,
    bm25: `<dt>k1</dt><dd>${params.bm25.k1}</dd><dt>b</dt><dd>${params.bm25.b}</dd><dt>idf</dt><dd>ln(1 + (N−df+.5)/(df+.5))</dd>`,
    dense: `<dt>encoder</dt><dd>${esc(params.dense.model.split("/").pop())}</dd><dt>dim</dt><dd>384</dd><dt>score</dt><dd>cosine</dd>`,
    hybrid: `<dt>fuses</dt><dd>BM25 + Dense</dd><dt>k</dt><dd>${params.rrf.k}</dd><dt>score</dt><dd>Σ 1 / (k + rank)</dd>`,
  }[model];
  return `
    <section><h3>How the query was read</h3>
      <div class="step"><span>Cleaned</span><span class="mono">${esc(p.cleaned) || "–"}</span></div>
      <div class="step"><span>Tokens</span><div class="chips">${tokens}</div></div>
      <div class="step"><span>Stems</span><div class="chips">${stems}</div></div>
      <p class="note">Struck-out tokens are stop-words or single characters and are dropped before indexing.</p>
    </section>
    <section><h3>Postings</h3>
      <table class="t"><tr><th>term</th><th class="n">df</th><th class="n">idf</th></tr>${terms}</table>
      <p class="note">df is the number of products whose title contains the stem.</p>
    </section>
    <section><h3>${MODELS[model]}</h3><dl class="kv">${setting}</dl></section>`;
}

async function renderEmpty() {
  $("#hits").innerHTML = ""; $("#meta").innerHTML = ""; $("#aside").innerHTML = "";
  const { queries } = await api("/api/queries?n=12");
  $("#empty").innerHTML = `
    <h2>Search about twenty thousand Amazon product titles.</h2>
    <p>Type anything, or pick one of the benchmark queries below. For those the results are marked with the human ESCI judgment and the list gets an nDCG@10 score.</p>
    <ul class="qlist">${queries.map((q) => `<li><a href="#/search?q=${encodeURIComponent(q)}&m=${st.model}">${esc(q)}</a></li>`).join("")}</ul>
    <p style="margin-top:16px"><button class="linkbtn" id="more">Show other queries</button></p>`;
  $("#more").onclick = renderEmpty;
}

// ---------------------------------------------------------------- compare

async function renderCompare() {
  const grid = $("#grid4");
  if (!st.q) {
    const { queries } = await api("/api/queries?n=1");
    st.q = queries[0]; writeHash(); $("#q").value = st.q;
  }
  const q = st.q;
  let d;
  try { d = await api(`/api/compare?q=${encodeURIComponent(q)}&k=10`); }
  catch (e) { grid.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  if (q !== st.q) return;

  const seen = {};
  d.runs.forEach((run) => run.results.forEach((r) => (seen[r.product_id] = (seen[r.product_id] || 0) + 1)));
  const anyJudged = d.runs[0].judged > 0;

  grid.innerHTML = d.runs.map((run) => `
    <div class="col">
      <div class="col-h"><b>${run.model === "dense" ? "Dense" : run.name}</b></div>
      <div class="col-m">${anyJudged ? `nDCG@10 ${fmt(run.ndcg10)} · ` : ""}${fmt(run.latency_ms, 1)} ms</div>
      <ol>${run.results.map((r) => `
        <li class="ci" data-pid="${r.product_id}" title="${esc(r.title)}">
          <span class="r">${r.rank}</span>
          <span class="x">${esc(r.title)}${seen[r.product_id] > 1 ? `<span class="shared">in ${seen[r.product_id]} of 4 lists</span>` : ""}</span>
          ${anyJudged ? lab(r.label) : "<span></span>"}
        </li>`).join("")}</ol>
    </div>`).join("");

  $("#compare-lede").textContent = anyJudged
    ? `“${q}” is a benchmark query with ${d.runs[0].judged} judged products. Hover a product to find it in the other lists.`
    : `The same query through all four models. Hover a product to find it in the other lists.`;

  grid.onmouseover = (e) => {
    const li = e.target.closest(".ci");
    grid.querySelectorAll(".ci.hl").forEach((x) => x.classList.remove("hl"));
    if (li) grid.querySelectorAll(`.ci[data-pid="${li.dataset.pid}"]`).forEach((x) => x.classList.add("hl"));
  };
  grid.onmouseleave = () => grid.querySelectorAll(".ci.hl").forEach((x) => x.classList.remove("hl"));
}

// ---------------------------------------------------------------- evaluation

async function renderEvaluation() {
  const el = $("#evaluation");
  if (!st.evalCache) {
    try { st.evalCache = await api("/api/evaluation"); }
    catch (e) { el.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  }
  const d = st.evalCache, proto = st.evalProto;
  const rows = d[proto];
  if (!rows.length) { el.innerHTML = `<p class="err">No results yet. Run <code>python run_evaluation.py</code>.</p>`; return; }

  const cols = Object.keys(rows[0]).filter((c) => c !== "Model");
  const real = rows.filter((r) => r.Model !== "Random");
  const best = Object.fromEntries(cols.map((c) => [c, Math.max(...real.map((r) => r[c]))]));
  const s = d.summary;

  el.innerHTML = `
    <div class="block">
      <div class="block-h">
        <h2>Test set, ${s.test_queries} queries</h2>
        <div class="seg">
          <button data-p="rerank" class="${proto === "rerank" ? "on" : ""}">Re-rank judged candidates</button>
          <button data-p="retrieval" class="${proto === "retrieval" ? "on" : ""}">Full catalog</button>
        </div>
      </div>
      <table class="m">
        <tr><th>Model</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>
        ${rows.map((r) => `<tr class="${r.Model === "Random" ? "base" : ""}">
          <td><span class="sw" style="background:var(--m-${SERIES[r.Model]})"></span>${r.Model}</td>
          ${cols.map((c) => `<td class="${r.Model !== "Random" && r[c] === best[c] ? "best" : ""}">${fmt(r[c], 4)}</td>`).join("")}
        </tr>`).join("")}
      </table>
      <p class="caption">${proto === "rerank"
        ? "Each model orders only the products that were judged for the query (on average 20), as in ESCI Task 1. Binary metrics count Exact as relevant; nDCG uses gains E 3, S 2, C 1, I 0. Underlined: best per column."
        : `Each model ranks all ${s.products.toLocaleString()} products. Unjudged products count as not relevant, so these numbers are a lower bound. Recall@100 shows how many Exact products make it into the first hundred.`}</p>
    </div>

    <div class="block two">
      <div>
        <div class="block-h"><h2>Precision and recall</h2></div>
        <div class="legend">${Object.keys(d[`pr_${proto}`]).map((m) => `<span><span class="sw" style="background:var(--m-${SERIES[m]})"></span>${m}</span>`).join("")}</div>
        <div class="chart" id="pr"></div>
        <p class="caption">11-point interpolated precision, averaged over test queries.</p>
      </div>
      <div>
        <div class="block-h"><h2>BM25 on dev queries</h2></div>
        <div id="heat"></div>
        <p class="caption">Mean nDCG@10 of the two protocols on the ${s.dev_queries} dev queries, for each k1 and b. The outlined cell is the setting used everywhere else: k1 = ${s.chosen.bm25_k1}, b = ${s.chosen.bm25_b}.</p>
      </div>
    </div>`;

  el.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => { st.evalProto = b.dataset.p; renderEvaluation(); }));
  drawPR($("#pr"), d[`pr_${proto}`]);
  drawHeat($("#heat"), d.bm25_grid, s.chosen);
}

function drawPR(el, curves) {
  const W = 560, H = 360, m = { l: 46, r: 16, t: 12, b: 40 };
  const x = (v) => m.l + v * (W - m.l - m.r), y = (v) => H - m.b - v * (H - m.t - m.b);
  const levels = [...Array(11).keys()].map((i) => i / 10);
  const grid = levels.filter((_, i) => i % 2 === 0).map((v) =>
    `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("");
  const ticks = levels.filter((_, i) => i % 2 === 0).map((v) =>
    `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text><text x="${x(v)}" y="${H - m.b + 18}" text-anchor="middle">${v.toFixed(1)}</text>`).join("");
  const series = Object.entries(curves).map(([name, vals]) => {
    const pts = vals.map((v, i) => `${x(levels[i])},${y(v)}`).join(" L");
    const dots = vals.map((v, i) => `<circle cx="${x(levels[i])}" cy="${y(v)}" r="3" fill="var(--m-${SERIES[name]})" stroke="var(--paper)" stroke-width="2"/>`).join("");
    return `<g class="series"><path d="M${pts}" stroke="var(--m-${SERIES[name]})" ${name === "Random" ? 'stroke-dasharray="2 4"' : ""}/>${dots}</g>`;
  }).join("");
  const hits = levels.map((v, i) => `<rect class="hit-area" data-i="${i}" x="${x(v) - (W - m.l - m.r) / 20}" y="${m.t}" width="${(W - m.l - m.r) / 10}" height="${H - m.t - m.b}"/>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision-recall curves">
    <g class="grid">${grid}</g><g class="axis">${ticks}</g>
    <text class="axis-label" x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">Recall</text>
    <text class="axis-label" transform="translate(12 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">Precision</text>
    ${series}${hits}</svg>`;
  const tip = $("#tip");
  el.querySelectorAll(".hit-area").forEach((r) => {
    r.onmousemove = (e) => {
      const i = +r.dataset.i;
      tip.innerHTML = `recall ${levels[i].toFixed(1)}<br>` + Object.entries(curves)
        .sort((a, b) => b[1][i] - a[1][i]).map(([n, v]) => `${n.padEnd(20, " ").replace(/ /g, "&nbsp;")}${v[i].toFixed(3)}`).join("<br>");
      tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
    };
    r.onmouseleave = () => (tip.style.display = "none");
  });
}

function drawHeat(el, grid, chosen) {
  if (!grid.length) return;
  const k1s = [...new Set(grid.map((g) => g.k1))], bs = [...new Set(grid.map((g) => g.b))];
  const vals = grid.map((g) => g.both), lo = Math.min(...vals), hi = Math.max(...vals);
  const cell = (k1, b) => grid.find((g) => g.k1 === k1 && g.b === b);
  const mix = (t) => `color-mix(in oklab, var(--heat-1) ${Math.round(t * 100)}%, var(--heat-0))`;
  el.innerHTML = `<table class="heat"><tr><th>k1 \\ b</th>${bs.map((b) => `<th>${b}</th>`).join("")}</tr>
    ${k1s.map((k1) => `<tr><th>${k1}</th>${bs.map((b) => {
      const g = cell(k1, b), t = (g.both - lo) / (hi - lo || 1);
      const pick = k1 === chosen.bm25_k1 && b === chosen.bm25_b;
      return `<td class="${pick ? "pick" : ""}" style="background:${mix(t)};color:${t > 0.55 ? "var(--paper)" : "var(--ink)"}">${g.both.toFixed(3)}</td>`;
    }).join("")}</tr>`).join("")}</table>`;
}

// ---------------------------------------------------------------- index

async function renderIndex() {
  const el = $("#index");
  if (!st.indexCache) {
    try { st.indexCache = await api("/api/index"); }
    catch (e) { el.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  }
  const { stats, top_terms, dataset } = st.indexCache;
  const labels = dataset.label_counts || {};
  const total = Object.values(labels).reduce((a, b) => a + b, 0) || 1;
  const maxDf = top_terms[0]?.df || 1;
  el.innerHTML = `
    <div class="figs">
      <div class="fig"><div class="v">${stats.documents.toLocaleString()}</div><div class="k">products indexed</div></div>
      <div class="fig"><div class="v">${stats.vocabulary.toLocaleString()}</div><div class="k">distinct stems</div></div>
      <div class="fig"><div class="v">${stats.postings.toLocaleString()}</div><div class="k">postings</div></div>
      <div class="fig"><div class="v">${stats.avg_doc_len}</div><div class="k">stems per title</div></div>
      <div class="fig"><div class="v">${Math.round(100 * stats.singleton_terms / stats.vocabulary)}%</div><div class="k">stems in one title only</div></div>
    </div>
    <div class="block two">
      <div>
        <div class="block-h"><h2>Most common stems</h2></div>
        ${top_terms.map((t) => `<div class="hbar"><span class="mono">${esc(t.term)}</span><span class="track"><span class="fill" style="display:block;width:${100 * t.df / maxDf}%"></span></span><span class="n">${t.df.toLocaleString()}</span></div>`).join("")}
        <p class="caption">Document frequency of the fifteen most common stems. Their low idf is why words like “pack” or “black” barely move a BM25 score.</p>
      </div>
      <div>
        <div class="block-h"><h2>Judgments</h2></div>
        ${["E", "S", "C", "I"].map((l) => `<div class="hbar"><span>${lab(l)} ${LABEL[l]}</span><span class="track"><span class="fill" style="display:block;width:${100 * (labels[l] || 0) / total}%"></span></span><span class="n">${(labels[l] || 0).toLocaleString()}</span></div>`).join("")}
        <dl class="kv" style="margin-top:22px">
          <dt>queries</dt><dd>${dataset.queries} (${dataset.dev_queries} dev, ${dataset.test_queries} test)</dd>
          <dt>judgments</dt><dd>${dataset.judgments?.toLocaleString()}</dd>
          <dt>locale</dt><dd>us, ESCI small version, test split</dd>
        </dl>
      </div>
    </div>`;
}

// ---------------------------------------------------------------- events

$("#query-form").addEventListener("submit", (e) => {
  e.preventDefault();
  st.q = $("#q").value.trim();
  writeHash(); render();
});
document.querySelectorAll(".models button").forEach((b) => b.addEventListener("click", () => {
  st.model = b.dataset.model; writeHash(); render();
}));
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("#q") && !document.body.classList.contains("hide-query")) {
    e.preventDefault(); $("#q").focus(); $("#q").select();
  }
});
window.addEventListener("popstate", () => { readHash(); render(); });
window.addEventListener("hashchange", () => { readHash(); render(); });

readHash();
render();
