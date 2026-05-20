"""Smoke tests for the Travel Planner Flask app.

These tests exercise the app without requiring any LLM API keys; the agent
falls back to the keyword router in that case.
"""

from __future__ import annotations

import os

import pytest

# Make sure no real LLM provider is selected for tests.
os.environ.setdefault("LLM_PROVIDER", "none")

from app.api import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_app_starts() -> None:
    app = create_app()
    assert app is not None
    assert app.name


def test_health_endpoint(client) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "travel-planner"
    assert "timestamp" in payload


def test_list_destinations(client) -> None:
    resp = client.get("/api/destinations")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "destinations" in payload
    assert isinstance(payload["destinations"], list)
    assert len(payload["destinations"]) >= 1
    first = payload["destinations"][0]
    for field in ("id", "name", "country", "avg_daily_cost"):
        assert field in first


def test_chat_keyword_fallback(client) -> None:
    resp = client.post(
        "/api/chat",
        json={
            "session_id": "smoke",
            "message": "Recommend a destination for a 5-day cultural trip",
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "response" in payload
    assert isinstance(payload["response"], str)
    assert payload["response"].strip()
    # With LLM_PROVIDER=none the agent must operate in fallback mode.
    assert payload.get("llm_enabled") is False


def test_destination_search(client) -> None:
    resp = client.get("/api/destinations/search?query=tokyo")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "results" in payload
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) >= 1


def test_budget_calculate(client) -> None:
    resp = client.post(
        "/api/budget/calculate",
        json={
            "destination": "Tokyo",
            "days": 5,
            "budget_level": "moderate",
            "travelers": 2,
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["destination"] == "Tokyo"
    assert payload["days"] == 5
    assert payload["travelers"] == 2
    assert "breakdown" in payload
    assert payload["total_estimate"] > 0
