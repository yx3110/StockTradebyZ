// Data Explorer frontend controller.
// Modules wired below: schemaBrowser, sqlEditor (Task 12), queryRunner (Task 13),
// chart (Task 13), builder (Task 14), saved (Task 15).

import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { sql } from "@codemirror/lang-sql";

const state = {
  schema: null,
  editor: null,
  lastResult: null,
  currentHint: null,
  currentChart: null,
  savedQueries: [],
};

// ---- Schema Browser ---------------------------------------------------------

async function loadSchema() {
  const resp = await fetch("/api/explorer/schema");
  const body = await resp.json();
  if (!body.success) {
    window.showToast("加载 schema 失败", "error");
    return;
  }
  state.schema = body.schema;
  renderSchema();
}

function renderSchema() {
  const tree = document.getElementById("schema-tree");
  tree.innerHTML = "";

  const orderedCats = [
    "raw", "technical", "feature_cache", "factor",
    "market_state", "moneyflow", "meta", "backtest", "other",
  ];
  for (const cat of orderedCats) {
    const tables = state.schema[cat];
    if (!tables || tables.length === 0) continue;

    const catDiv = document.createElement("div");
    catDiv.className = "category";
    catDiv.innerHTML = `<i class="bi bi-caret-down-fill"></i> <b>${cat}</b> <span class="category-count">(${tables.length})</span>`;
    const ul = document.createElement("ul");
    ul.className = "tables";

    for (const t of tables.sort((a, b) => a.table.localeCompare(b.table))) {
      const li = document.createElement("li");
      li.textContent = t.table;
      li.title = `${t.row_count.toLocaleString()} 行` +
        (t.date_range ? ` · ${t.date_range[0]} → ${t.date_range[1]}` : "");
      if (t.has_features_json) li.classList.add("has-json");
      li.addEventListener("click", () => insertSampleQuery(t));
      ul.appendChild(li);
    }

    catDiv.addEventListener("click", () => {
      // toggle collapse
      const icon = catDiv.querySelector(".bi");
      if (ul.style.display === "none") {
        ul.style.display = "";
        icon.classList.replace("bi-caret-right-fill", "bi-caret-down-fill");
      } else {
        ul.style.display = "none";
        icon.classList.replace("bi-caret-down-fill", "bi-caret-right-fill");
      }
    });

    tree.appendChild(catDiv);
    tree.appendChild(ul);
  }

  // Search box filters
  document.getElementById("schema-search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    tree.querySelectorAll("ul.tables li").forEach((li) => {
      li.style.display = li.textContent.toLowerCase().includes(q) ? "" : "none";
    });
  });

  populateBuilder();
}

function insertSampleQuery(tableInfo) {
  const hasTradeDate = tableInfo.columns.some((c) => c.name === "trade_date");
  let sample;
  if (hasTradeDate) {
    sample = `SELECT *\nFROM ${tableInfo.table}\nWHERE trade_date = (SELECT MAX(trade_date) FROM ${tableInfo.table})\nLIMIT 100;`;
  } else {
    sample = `SELECT *\nFROM ${tableInfo.table}\nLIMIT 100;`;
  }
  setEditorContent(sample);
}

// ---- SQL Editor (CodeMirror 6) ---------------------------------------------

function setEditorContent(content) {
  state.editor.dispatch({
    changes: { from: 0, to: state.editor.state.doc.length, insert: content },
  });
}

function getEditorContent() {
  return state.editor.state.doc.toString();
}

function initEditor() {
  const container = document.getElementById("sql-editor-container");
  const runKey = {
    key: "Mod-Enter",
    run: () => { runQuery(); return true; },
  };
  state.editor = new EditorView({
    state: EditorState.create({
      doc: "-- 点击左侧表名自动生成 sample SQL, 或直接写:\nSELECT 1;",
      extensions: [
        lineNumbers(),
        history(),
        sql(),
        keymap.of([runKey, ...defaultKeymap, ...historyKeymap]),
      ],
    }),
    parent: container,
  });
}

// ---- Query Runner -----------------------------------------------------------

async function runQuery() {
  const sqlText = getEditorContent().trim();
  if (!sqlText) {
    window.showToast("SQL 为空", "warning");
    return;
  }
  const expandFeatures = document.getElementById("expand-features").checked;
  const runBtn = document.getElementById("btn-run");
  runBtn.disabled = true;
  document.getElementById("query-summary").textContent = "执行中...";
  document.getElementById("warning-area").innerHTML = "";

  try {
    const resp = await fetch("/api/explorer/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql: sqlText, expand_features: expandFeatures }),
    });
    const body = await resp.json();
    if (!body.success) {
      window.showToast(body.error || "查询失败", "error");
      document.getElementById("query-summary").textContent =
        `错误 (${body.code}): ${body.error}`;
      return;
    }
    state.lastResult = body;
    state.currentHint = body.chart_hint;
    renderResultTable(body);
    renderWarnings(body.warnings, body.truncated);
    document.getElementById("query-summary").textContent =
      `${body.row_count.toLocaleString()} 行 · ${body.took_ms} ms` +
      (body.truncated ? " (截断)" : "");
    renderChart(body, "auto");
  } catch (e) {
    window.showToast("网络错误: " + e.message, "error");
  } finally {
    runBtn.disabled = false;
  }
}

function renderWarnings(warnings, truncated) {
  const area = document.getElementById("warning-area");
  area.innerHTML = "";
  if (truncated) {
    const banner = document.createElement("div");
    banner.className = "warning-banner";
    banner.textContent = "⚠️ 结果已截断至上限 10 000 行 — 请加 LIMIT 或 WHERE 缩小范围";
    area.appendChild(banner);
  }
  for (const w of warnings || []) {
    const div = document.createElement("div");
    div.className = "warning-banner";
    div.textContent = w;
    area.appendChild(div);
  }
}

function renderResultTable(body) {
  const wrap = document.getElementById("result-table-wrap");
  wrap.innerHTML = "";
  if (body.row_count === 0) {
    wrap.innerHTML = '<div class="text-muted small">0 rows</div>';
    return;
  }
  const table = document.createElement("table");
  table.className = "table table-sm table-striped";
  table.style.width = "100%";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr>" + body.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr>";
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of body.rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = row.map((v) => `<td>${v === null ? "<span class='text-muted'>·</span>" : escapeHtml(String(v))}</td>`).join("");
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);

  // DataTables for sort + paging (library loaded in base.html)
  $(table).DataTable({
    pageLength: 25,
    deferRender: true,
    scrollX: true,
    order: [],
  });
  document.getElementById("table-meta").textContent =
    `${body.row_count.toLocaleString()} rows × ${body.columns.length} cols`;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ---- Chart ------------------------------------------------------------------

function renderChart(body, typeOverride) {
  const hint = body.chart_hint;
  const label = document.getElementById("chart-hint-label");
  const container = document.getElementById("result-chart");
  container.innerHTML = "";

  // Determine type
  let type = typeOverride;
  if (type === "auto") type = hint ? hint.type : "none";
  if (!type || type === "none") {
    label.textContent = hint ? "" : "无合适图表";
    return;
  }
  label.textContent = (typeOverride === "auto" && hint) ? `✨ auto: ${hint.type}` : "";

  // Build series depending on type. Use first numeric as Y if hint missing.
  const numericCols = body.columns.filter((c, i) =>
    body.rows.every((r) => r[i] === null || typeof r[i] === "number")
  );
  const firstNumeric = numericCols[0];
  const xCol = (hint && hint.x) || "code";
  const yCol = (hint && hint.y) || firstNumeric;
  if (!yCol) { label.textContent = "无数值列可画"; return; }

  const xIdx = body.columns.indexOf(xCol);
  const yIdx = body.columns.indexOf(yCol);
  if (yIdx < 0) { label.textContent = "指定列不存在"; return; }

  let options;
  if (type === "line") {
    options = {
      chart: { type: "line", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: body.rows.map((r) => r[yIdx]) }],
      xaxis: { categories: body.rows.map((r) => r[xIdx]), title: { text: xCol } },
      yaxis: { title: { text: yCol } },
      stroke: { width: 2 },
    };
  } else if (type === "scatter") {
    options = {
      chart: { type: "scatter", height: 280, animations: { enabled: false } },
      series: [{
        name: `${xCol} vs ${yCol}`,
        data: body.rows.map((r) => [r[xIdx], r[yIdx]]).filter((p) => p[0] !== null && p[1] !== null),
      }],
      xaxis: { title: { text: xCol } },
      yaxis: { title: { text: yCol } },
      title: { text: hint && hint.annotations ? `r = ${hint.annotations.pearson_r}` : "" },
    };
  } else if (type === "bar") {
    // sort descending by Y
    const sorted = [...body.rows].sort((a, b) => (b[yIdx] ?? 0) - (a[yIdx] ?? 0));
    options = {
      chart: { type: "bar", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: sorted.map((r) => r[yIdx]) }],
      xaxis: { categories: sorted.map((r) => r[xIdx]), title: { text: xCol } },
      yaxis: { title: { text: yCol } },
    };
  } else if (type === "histogram") {
    const values = body.rows.map((r) => r[yIdx]).filter((v) => typeof v === "number");
    const bins = histogram(values, 20);
    options = {
      chart: { type: "bar", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: bins.counts }],
      xaxis: {
        categories: bins.edges.map((v) => v.toFixed(3)),
        title: { text: yCol },
      },
      yaxis: { title: { text: "count" } },
    };
  }

  if (state.currentChart) {
    state.currentChart.destroy();
    state.currentChart = null;
  }
  if (options) {
    state.currentChart = new ApexCharts(container, options);
    state.currentChart.render();
  }
}

function histogram(values, nBins) {
  if (values.length === 0) return { edges: [], counts: [] };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const step = (hi - lo) / nBins || 1;
  const counts = new Array(nBins).fill(0);
  const edges = [];
  for (let i = 0; i < nBins; i++) edges.push(lo + i * step);
  for (const v of values) {
    const idx = Math.min(nBins - 1, Math.floor((v - lo) / step));
    counts[idx]++;
  }
  return { edges, counts };
}

// ---- Bootstrap --------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initEditor();
  loadSchema();
  document.getElementById("btn-run").addEventListener("click", runQuery);
});

// Chart type dropdown
document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("chart-type-select");
  sel.addEventListener("change", () => {
    if (state.lastResult) renderChart(state.lastResult, sel.value);
  });
});

// ---- Visual Builder ---------------------------------------------------------

function getAllTablesFlat() {
  const out = [];
  for (const cat of Object.values(state.schema || {})) {
    for (const t of cat) out.push(t);
  }
  return out.sort((a, b) => a.table.localeCompare(b.table));
}

function populateBuilder() {
  const tables = getAllTablesFlat();
  const tSel = document.getElementById("b-table");
  tSel.innerHTML = tables.map((t) => `<option value="${t.table}">${t.table}</option>`).join("");

  tSel.addEventListener("change", () => refreshBuilderColumns());
  refreshBuilderColumns();

  document.getElementById("b-add-filter").addEventListener("click", () => addFilterRow());
  document.getElementById("b-build-sql").addEventListener("click", () => buildSqlFromBuilder());
}

function findTable(name) {
  return getAllTablesFlat().find((t) => t.table === name);
}

function refreshBuilderColumns() {
  const name = document.getElementById("b-table").value;
  const t = findTable(name);
  const cols = t ? t.columns.map((c) => c.name) : [];
  document.getElementById("b-columns").innerHTML =
    cols.map((c) => `<option value="${c}" selected>${c}</option>`).join("");
  document.getElementById("b-order-col").innerHTML =
    `<option value="">(none)</option>` +
    cols.map((c) => `<option value="${c}">${c}</option>`).join("");
  document.getElementById("b-filters-rows").innerHTML = "";
}

function addFilterRow() {
  const name = document.getElementById("b-table").value;
  const t = findTable(name);
  const cols = t ? t.columns.map((c) => c.name) : [];
  const row = document.createElement("div");
  row.className = "input-group input-group-sm mb-1";
  row.innerHTML = `
    <select class="form-select form-select-sm b-f-col">
      ${cols.map((c) => `<option>${c}</option>`).join("")}
    </select>
    <select class="form-select form-select-sm b-f-op" style="max-width:90px">
      <option>=</option><option>!=</option><option>&gt;</option><option>&lt;</option>
      <option>&gt;=</option><option>&lt;=</option><option>LIKE</option><option>IN</option>
    </select>
    <input class="form-control form-control-sm b-f-val" placeholder="value">
    <button class="btn btn-outline-danger btn-sm b-f-del">&times;</button>
  `;
  row.querySelector(".b-f-del").addEventListener("click", () => row.remove());
  document.getElementById("b-filters-rows").appendChild(row);
}

function buildSqlFromBuilder() {
  const table = document.getElementById("b-table").value;
  const cols = [...document.getElementById("b-columns").selectedOptions].map((o) => o.value);
  const limit = parseInt(document.getElementById("b-limit").value, 10) || 100;
  const orderCol = document.getElementById("b-order-col").value;
  const orderDir = document.getElementById("b-order-dir").value;

  const selectList = cols.length ? cols.join(", ") : "*";
  let sql = `SELECT ${selectList}\nFROM ${table}`;

  const filterRows = document.querySelectorAll("#b-filters-rows .input-group");
  const clauses = [];
  for (const r of filterRows) {
    const col = r.querySelector(".b-f-col").value;
    const op = r.querySelector(".b-f-op").value;
    const valRaw = r.querySelector(".b-f-val").value.trim();
    if (!valRaw) continue;
    let val = valRaw;
    if (op === "IN") {
      val = "(" + valRaw.split(",").map((v) => `'${v.trim()}'`).join(", ") + ")";
    } else if (isNaN(Number(valRaw))) {
      val = `'${valRaw.replace(/'/g, "''")}'`;
    }
    clauses.push(`  ${col} ${op} ${val}`);
  }
  if (clauses.length) sql += `\nWHERE\n${clauses.join("\n  AND\n")}`;
  if (orderCol) sql += `\nORDER BY ${orderCol} ${orderDir}`;
  sql += `\nLIMIT ${limit};`;

  setEditorContent(sql);
  toggleInputMode("sql");
}

// Input mode toggle (SQL <-> Builder)
function toggleInputMode(mode) {
  const sqlC = document.getElementById("sql-editor-container");
  const bC = document.getElementById("builder-container");
  const toggle = document.getElementById("input-mode-toggle");
  if (mode === "builder") {
    sqlC.classList.add("d-none");
    bC.classList.remove("d-none");
    toggle.querySelector('[data-input="sql"]').classList.replace("btn-primary", "btn-outline-secondary");
    toggle.querySelector('[data-input="builder"]').classList.replace("btn-outline-secondary", "btn-primary");
  } else {
    sqlC.classList.remove("d-none");
    bC.classList.add("d-none");
    toggle.querySelector('[data-input="sql"]').classList.replace("btn-outline-secondary", "btn-primary");
    toggle.querySelector('[data-input="builder"]').classList.replace("btn-primary", "btn-outline-secondary");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#input-mode-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => toggleInputMode(btn.dataset.input));
  });
});

// ---- Saved queries ---------------------------------------------------------

async function loadSavedQueries() {
  const resp = await fetch("/api/explorer/saved");
  const body = await resp.json();
  if (!body.success) return;
  state.savedQueries = body.queries;
  document.getElementById("saved-count").textContent = body.queries.length;
  renderSavedList();
}

function renderSavedList() {
  const host = document.getElementById("saved-query-list");
  host.innerHTML = "";
  const filter = (document.getElementById("saved-tag-filter").value || "").trim().toLowerCase();
  const rows = state.savedQueries.filter((q) =>
    !filter || (q.tags || "").toLowerCase().includes(filter)
  );
  if (rows.length === 0) {
    host.innerHTML = '<div class="text-muted small p-2">无保存查询</div>';
    return;
  }
  for (const q of rows) {
    const d = document.createElement("div");
    d.className = "saved-row";
    d.innerHTML = `
      <div class="d-flex align-items-center">
        <b class="me-2">${escapeHtml(q.name)}</b>
        <span class="text-muted small">${q.tags ? "[" + escapeHtml(q.tags) + "]" : ""}</span>
        <span class="ms-auto text-muted small">run ${q.run_count}×</span>
        <button class="btn btn-sm btn-outline-primary ms-2 b-load">加载</button>
        <button class="btn btn-sm btn-outline-danger ms-1 b-del">×</button>
      </div>
      <div class="text-muted small mt-1">${escapeHtml(q.description || "")}</div>
      <pre class="small mb-0" style="white-space:pre-wrap">${escapeHtml(q.sql)}</pre>
    `;
    d.querySelector(".b-load").addEventListener("click", () => {
      setEditorContent(q.sql);
      window.location.hash = "#sql";
    });
    d.querySelector(".b-del").addEventListener("click", async () => {
      if (!confirm(`删除 "${q.name}"?`)) return;
      await fetch(`/api/explorer/saved/${q.id}`, { method: "DELETE" });
      loadSavedQueries();
    });
    host.appendChild(d);
  }
}

async function openSaveModal() {
  document.getElementById("save-name").value = "";
  document.getElementById("save-tags").value = "user";
  document.getElementById("save-description").value = "";
  new bootstrap.Modal(document.getElementById("save-modal")).show();
}

async function confirmSave() {
  const name = document.getElementById("save-name").value.trim();
  const tags = document.getElementById("save-tags").value.trim();
  const description = document.getElementById("save-description").value.trim();
  const sqlText = getEditorContent().trim();
  if (!name) { window.showToast("请输入名称", "warning"); return; }
  const resp = await fetch("/api/explorer/saved", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, sql: sqlText, tags, description }),
  });
  const body = await resp.json();
  if (!body.success) {
    window.showToast(body.error || "保存失败", "error");
    return;
  }
  bootstrap.Modal.getInstance(document.getElementById("save-modal")).hide();
  window.showToast(`已保存: ${name}`, "success");
  loadSavedQueries();
}

// ---- Hash routing (Mode tabs) ---------------------------------------------

const MODE_PRESET_NAMES = {
  stock:   "preset: single-stock all-features",
  cross:   "preset: cross-section pred_10d top50",
  compare: "preset: model compare ng101 vs ng110",
};

async function applyMode(mode) {
  // Update tab UI
  document.querySelectorAll(".mode-tabs .nav-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.mode === mode);
  });

  const main = document.getElementById("explorer-main");
  const saved = document.getElementById("saved-panel");
  if (mode === "saved") {
    main.classList.add("d-none");
    saved.classList.remove("d-none");
    await loadSavedQueries();
    return;
  }
  main.classList.remove("d-none");
  saved.classList.add("d-none");

  if (mode in MODE_PRESET_NAMES) {
    const presetName = MODE_PRESET_NAMES[mode];
    const preset = state.savedQueries.find((q) => q.name === presetName);
    if (preset) {
      setEditorContent(preset.sql);
    } else {
      window.showToast(
        `preset "${presetName}" missing — run saved_seed or re-check`,
        "warning"
      );
    }
  }
}

function currentMode() {
  return (window.location.hash || "#sql").slice(1);
}

window.addEventListener("hashchange", () => applyMode(currentMode()));

// ---- CSV download -----------------------------------------------------------

function downloadCsv() {
  if (!state.lastResult) {
    window.showToast("先运行一条查询", "warning");
    return;
  }
  const { columns, rows } = state.lastResult;
  const lines = [columns.join(",")];
  for (const r of rows) {
    lines.push(r.map((v) => {
      if (v === null) return "";
      const s = String(v);
      return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `explorer_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

// ---- Final wiring ----------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-save").addEventListener("click", openSaveModal);
  document.getElementById("save-modal-confirm").addEventListener("click", confirmSave);
  document.getElementById("btn-csv").addEventListener("click", downloadCsv);
  document.getElementById("saved-tag-filter").addEventListener("input", renderSavedList);

  // Load saved queries once so presets are available when the user switches modes
  loadSavedQueries().then(() => applyMode(currentMode()));
});
