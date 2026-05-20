"""
Flask API for the Travel Planner.

Run for development with:

    python -m app.api

For production, prefer running through gunicorn:

    gunicorn --bind 0.0.0.0:5000 --workers 2 app.api:app
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.wrappers import Response
from flask_cors import CORS

from app.agents.travel_agent import TravelPlanningAgent, create_travel_agent
from app.tools.travel_tools import (
    DESTINATIONS_DB,
    AccommodationSearchTool,
    ActivitySearchTool,
    BudgetCalculatorTool,
    DestinationSearchTool,
    WeatherForecastTool,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATIC = _REPO_ROOT / "static"
_DEFAULT_TEMPLATES = _REPO_ROOT / "templates"


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        static_folder=str(_DEFAULT_STATIC),
        template_folder=str(_DEFAULT_TEMPLATES),
    )
    CORS(app)

    agents: dict[str, TravelPlanningAgent] = {}

    def get_agent(session_id: str) -> TravelPlanningAgent:
        if session_id not in agents:
            agents[session_id] = create_travel_agent()
        return agents[session_id]

    @app.route("/")
    def index() -> Response | str:
        # Templates are optional in this repo (the demo also lives under docs/).
        if (_DEFAULT_TEMPLATES / "index.html").exists():
            return render_template("index.html")
        return jsonify(
            {
                "service": "travel-planner",
                "message": (
                    "Travel Planner API. See /api/health, /api/destinations, "
                    "/api/chat."
                ),
            }
        )

    @app.route("/static/<path:filename>")
    def serve_static(filename: str) -> Response:
        return send_from_directory(app.static_folder, filename)

    @app.route("/api/health", methods=["GET"])
    def health_check() -> Response:
        agent_count = len(agents)
        sample_agent = next(iter(agents.values()), None)
        llm_enabled = sample_agent.llm_enabled if sample_agent is not None else False
        return jsonify(
            {
                "status": "healthy",
                "service": "travel-planner",
                "timestamp": datetime.now(UTC).isoformat(),
                "active_sessions": agent_count,
                "llm_enabled": llm_enabled,
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)
            return jsonify(
                {
                    "response": response,
                    "context": agent.get_context_summary(),
                    "session_id": session_id,
                    "llm_enabled": agent.llm_enabled,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/destinations", methods=["GET"])
    def list_destinations() -> Response:
        destinations = [
            {
                "id": key,
                "name": dest.name,
                "country": dest.country,
                "description": dest.description,
                "best_season": dest.best_season,
                "avg_daily_cost": dest.avg_daily_cost,
                "safety_rating": dest.safety_rating,
            }
            for key, dest in DESTINATIONS_DB.items()
        ]
        return jsonify({"destinations": destinations, "count": len(destinations)})

    @app.route("/api/destinations/search", methods=["GET"])
    def search_destinations() -> Response:
        query = request.args.get("query", "")
        budget = request.args.get("budget")
        season = request.args.get("season")

        tool = DestinationSearchTool()
        result = tool._run(query=query, budget=budget, season=season)
        return jsonify({"results": json.loads(result), "query": query})

    @app.route("/api/accommodations/search", methods=["GET"])
    def search_accommodations() -> tuple[Response, int] | Response:
        destination = request.args.get("destination", "")
        acc_type = request.args.get("type")
        max_price = request.args.get("max_price", type=float)

        if not destination:
            return jsonify({"error": "Destination is required"}), 400

        tool = AccommodationSearchTool()
        result = tool._run(
            destination=destination,
            accommodation_type=acc_type,
            max_price=max_price,
        )
        return jsonify(
            {"accommodations": json.loads(result), "destination": destination}
        )

    @app.route("/api/activities/search", methods=["GET"])
    def search_activities() -> tuple[Response, int] | Response:
        destination = request.args.get("destination", "")
        activity_type = request.args.get("type")
        max_duration = request.args.get("max_duration", type=float)

        if not destination:
            return jsonify({"error": "Destination is required"}), 400

        tool = ActivitySearchTool()
        result = tool._run(
            destination=destination,
            activity_type=activity_type,
            max_duration=max_duration,
        )
        return jsonify(
            {"activities": json.loads(result), "destination": destination}
        )

    @app.route("/api/budget/calculate", methods=["POST"])
    def calculate_budget() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        destination = data.get("destination", "")
        days = data.get("days", 5)
        budget_level = data.get("budget_level", "moderate")
        travelers = data.get("travelers", 1)

        if not destination:
            return jsonify({"error": "Destination is required"}), 400

        tool = BudgetCalculatorTool()
        result = tool._run(
            destination=destination,
            days=days,
            accommodation_budget=budget_level,
            travelers=travelers,
        )
        return jsonify(json.loads(result))

    @app.route("/api/weather", methods=["GET"])
    def get_weather() -> tuple[Response, int] | Response:
        destination = request.args.get("destination", "")
        date = request.args.get(
            "date", datetime.now(UTC).strftime("%Y-%m-%d")
        )

        if not destination:
            return jsonify({"error": "Destination is required"}), 400

        tool = WeatherForecastTool()
        result = tool._run(destination=destination, date=date)
        return jsonify(json.loads(result))

    @app.route("/api/itinerary", methods=["POST"])
    def generate_itinerary() -> tuple[Response, int] | Response:
        """Generate a full itinerary using the current session context."""
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        agent = get_agent(session_id)
        # Trigger the agent's itinerary path with a canonical phrase.
        response = agent.chat("create itinerary")
        return jsonify(
            {
                "itinerary": response,
                "context": agent.get_context_summary(),
                "session_id": session_id,
                "llm_enabled": agent.llm_enabled,
            }
        )

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session() -> Response:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        if session_id in agents:
            agents[session_id].reset()
            del agents[session_id]
        return jsonify({"status": "reset", "session_id": session_id})

    @app.route("/api/session/context", methods=["GET"])
    def get_session_context() -> Response:
        session_id = request.args.get("session_id", "default")
        if session_id in agents:
            return jsonify(
                {
                    "context": agents[session_id].get_context_summary(),
                    "session_id": session_id,
                }
            )
        return jsonify({"context": {}, "session_id": session_id})

    return app


app = create_app()


if __name__ == "__main__":
    # Development entry point only. Use gunicorn (or another WSGI server) in
    # production: ``gunicorn --bind 0.0.0.0:5000 --workers 2 app.api:app``.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
