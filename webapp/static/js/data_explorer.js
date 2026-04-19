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
  thead.innerHTML = "<tr>" + body.columns.map((c) => `<th>${c}</th>`).join("") + "</tr>";
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

// ---- Chart (stub — filled by Task 13) --------------------------------------

function renderChart(body, typeOverride) {
  // Stub — Task 13 fills this in
  const label = document.getElementById("chart-hint-label");
  if (label) label.textContent = "(chart rendered in Task 13)";
}

// ---- Bootstrap --------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initEditor();
  loadSchema();
  document.getElementById("btn-run").addEventListener("click", runQuery);
});
