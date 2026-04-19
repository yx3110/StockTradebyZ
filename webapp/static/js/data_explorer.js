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

// ---- Query Runner (filled in Task 13) --------------------------------------

async function runQuery() {
  // Stub — Task 13 fills this in
  window.showToast("runQuery stub — Task 12 wires this up", "info");
}

// ---- Bootstrap --------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initEditor();
  loadSchema();
  document.getElementById("btn-run").addEventListener("click", runQuery);
});
