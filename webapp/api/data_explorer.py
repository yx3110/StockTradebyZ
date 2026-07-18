"""Flask blueprint for the data explorer page.

Endpoints (all under url_prefix='/api/explorer'):
  GET    /schema              -> { success, schema: {cat: [TableInfo...]} }
  POST   /query               -> { success, columns, rows, row_count, truncated,
                                   took_ms, warnings, chart_hint }
  GET    /saved               -> { success, queries: [SavedQuery...] }
  POST   /saved               -> 201 { success, query }
  GET    /saved/<id>          -> { success, query }
  PUT    /saved/<id>          -> { success, query }
  DELETE /saved/<id>          -> { success }
  POST   /saved/<id>/run      -> run a saved query, increment run_count
"""
from __future__ import annotations

import dataclasses
import logging

from flask import Blueprint, current_app, jsonify, request

from core.data_explorer.query_runner import (
    InvalidQueryError,
    QueryTimeoutError,
    run_query,
)
from core.data_explorer.query_store import (
    apply_migration,
    create_query,
    delete_query,
    get_query,
    list_queries,
    seed_default_queries,
    touch_query,
    update_query,
)
from core.data_explorer.schema_discovery import discover


logger = logging.getLogger(__name__)

data_explorer_bp = Blueprint("data_explorer", __name__)


def _err(code: str, message: str, http: int = 400):
    return jsonify({"success": False, "error": message, "code": code}), http


def _ensure_store_ready() -> None:
    """Idempotent migration + seed on first API hit."""
    db = current_app.config["WEBAPP_DB_PATH"]
    apply_migration(db)
    seed_default_queries(db)


@data_explorer_bp.route("/schema", methods=["GET"])
def schema():
    try:
        refresh = request.args.get("refresh") == "1"
        schema_dict = discover(
            current_app.config["STOCK_DB_PATH"], refresh=refresh
        )
        return jsonify({"success": True, "schema": schema_dict})
    except Exception as e:
        logger.error("schema discovery failed", exc_info=True)
        return _err("server_error", str(e), 500)


@data_explorer_bp.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    sql = (payload.get("sql") or "").strip()
    if not sql:
        return _err("invalid_query", "empty sql")
    expand = bool(payload.get("expand_features", True))
    try:
        result = run_query(
            current_app.config["STOCK_DB_PATH"],
            sql,
            expand_features=expand,
        )
        body = dataclasses.asdict(result)
        body["success"] = True
        return jsonify(body)
    except InvalidQueryError as e:
        return _err("invalid_query", str(e), 400)
    except QueryTimeoutError as e:
        return _err("timeout", str(e), 504)
    except Exception as e:
        logger.error("run_query failed", exc_info=True)
        return _err("server_error", str(e), 500)


@data_explorer_bp.route("/saved", methods=["GET"])
def saved_list():
    _ensure_store_ready()
    tag = request.args.get("tag")
    return jsonify({
        "success": True,
        "queries": list_queries(current_app.config["WEBAPP_DB_PATH"], tag=tag),
    })


@data_explorer_bp.route("/saved", methods=["POST"])
def saved_create():
    _ensure_store_ready()
    payload = request.get_json(silent=True) or {}
    try:
        q = create_query(
            current_app.config["WEBAPP_DB_PATH"],
            name=payload["name"],
            sql=payload["sql"],
            tags=payload.get("tags"),
            description=payload.get("description"),
        )
        return jsonify({"success": True, "query": q}), 201
    except KeyError as e:
        return _err("invalid_payload", f"missing field: {e.args[0]}")
    except ValueError as e:
        return _err("conflict", str(e), 409)


@data_explorer_bp.route("/saved/<int:qid>", methods=["GET"])
def saved_get(qid: int):
    _ensure_store_ready()
    try:
        return jsonify({
            "success": True,
            "query": get_query(current_app.config["WEBAPP_DB_PATH"], qid),
        })
    except LookupError as e:
        return _err("not_found", str(e), 404)


@data_explorer_bp.route("/saved/<int:qid>", methods=["PUT"])
def saved_update(qid: int):
    _ensure_store_ready()
    payload = request.get_json(silent=True) or {}
    try:
        q = update_query(
            current_app.config["WEBAPP_DB_PATH"], qid, **payload
        )
        return jsonify({"success": True, "query": q})
    except LookupError as e:
        return _err("not_found", str(e), 404)
    except ValueError as e:
        # 改名撞 UNIQUE(name) → update_query 抛 ValueError, 与 saved_create 一致返回 409 而非 500
        return _err("conflict", str(e), 409)


@data_explorer_bp.route("/saved/<int:qid>", methods=["DELETE"])
def saved_delete(qid: int):
    _ensure_store_ready()
    delete_query(current_app.config["WEBAPP_DB_PATH"], qid)
    return jsonify({"success": True})


@data_explorer_bp.route("/saved/<int:qid>/run", methods=["POST"])
def saved_run(qid: int):
    """Run a saved query and increment its run_count / last_run_at."""
    _ensure_store_ready()
    try:
        q = get_query(current_app.config["WEBAPP_DB_PATH"], qid)
    except LookupError as e:
        return _err("not_found", str(e), 404)

    payload = request.get_json(silent=True) or {}
    expand = bool(payload.get("expand_features", True))
    try:
        result = run_query(
            current_app.config["STOCK_DB_PATH"], q["sql"], expand_features=expand,
        )
        touch_query(current_app.config["WEBAPP_DB_PATH"], qid)
        body = dataclasses.asdict(result)
        body["success"] = True
        return jsonify(body)
    except InvalidQueryError as e:
        return _err("invalid_query", str(e), 400)
    except QueryTimeoutError as e:
        return _err("timeout", str(e), 504)
