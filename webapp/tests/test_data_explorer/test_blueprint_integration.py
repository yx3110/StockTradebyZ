"""Integration tests: hit the blueprint through a live Flask test client."""
import json
import sys
from pathlib import Path

import pytest

webapp_root = Path(__file__).resolve().parents[2]
if str(webapp_root) not in sys.path:
    sys.path.insert(0, str(webapp_root))

from app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_stock_db: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config["STOCK_DB_PATH"] = tmp_stock_db
    app.config["WEBAPP_DB_PATH"] = tmp_path / "webapp.db"
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_schema_endpoint(client):
    resp = client.get("/api/explorer/schema")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    cats = body["schema"]
    assert "raw" in cats
    assert any(t["table"] == "daily_quotes" for t in cats["raw"])


def test_query_endpoint_happy_path(client):
    resp = client.post(
        "/api/explorer/query",
        data=json.dumps({"sql": "SELECT * FROM daily_quotes LIMIT 3"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["row_count"] == 3
    assert "columns" in body and "rows" in body


def test_query_endpoint_rejects_write(client):
    resp = client.post(
        "/api/explorer/query",
        data=json.dumps({"sql": "DELETE FROM daily_quotes"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert body["code"] == "invalid_query"


def test_saved_query_crud(client):
    # create
    resp = client.post(
        "/api/explorer/saved",
        data=json.dumps({
            "name": "t1", "sql": "SELECT 1",
            "tags": "test", "description": ""
        }),
        content_type="application/json",
    )
    assert resp.status_code == 201
    qid = resp.get_json()["query"]["id"]

    # list
    resp = client.get("/api/explorer/saved")
    names = [q["name"] for q in resp.get_json()["queries"]]
    assert "t1" in names

    # delete
    resp = client.delete(f"/api/explorer/saved/{qid}")
    assert resp.status_code == 200

    resp = client.get("/api/explorer/saved")
    assert not any(q["id"] == qid for q in resp.get_json()["queries"])
