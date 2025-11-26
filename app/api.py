"""
Flask API for Travel Planner
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime

from app.agents.travel_agent import TravelPlanningAgent, create_travel_agent
from app.tools.travel_tools import (
    DestinationSearchTool,
    AccommodationSearchTool,
    ActivitySearchTool,
    BudgetCalculatorTool,
    WeatherForecastTool,
    DESTINATIONS_DB
)


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__,
                static_folder='../static',
                template_folder='../templates')
    CORS(app)

    # Store agents per session (in production, use Redis or similar)
    agents = {}

    def get_agent(session_id: str) -> TravelPlanningAgent:
        """Get or create agent for session."""
        if session_id not in agents:
            agents[session_id] = create_travel_agent()
        return agents[session_id]

    @app.route('/')
    def index():
        """Serve the main demo page."""
        return render_template('index.html')

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'service': 'travel-planner',
            'timestamp': datetime.utcnow().isoformat()
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Main chat endpoint for travel planning."""
        data = request.json
        session_id = data.get('session_id', 'default')
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)

            return jsonify({
                'response': response,
                'context': agent.get_context_summary(),
                'session_id': session_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/destinations', methods=['GET'])
    def get_destinations():
        """Get list of available destinations."""
        destinations = []
        for key, dest in DESTINATIONS_DB.items():
            destinations.append({
                'id': key,
                'name': dest.name,
                'country': dest.country,
                'description': dest.description,
                'best_season': dest.best_season,
                'avg_daily_cost': dest.avg_daily_cost,
                'safety_rating': dest.safety_rating
            })

        return jsonify({
            'destinations': destinations,
            'count': len(destinations)
        })

    @app.route('/api/destinations/search', methods=['GET'])
    def search_destinations():
        """Search destinations based on criteria."""
        query = request.args.get('query', '')
        budget = request.args.get('budget')
        season = request.args.get('season')

        tool = DestinationSearchTool()
        result = tool._run(query=query, budget=budget, season=season)

        return jsonify({
            'results': json.loads(result),
            'query': query
        })

    @app.route('/api/accommodations/search', methods=['GET'])
    def search_accommodations():
        """Search accommodations for a destination."""
        destination = request.args.get('destination', '')
        acc_type = request.args.get('type')
        max_price = request.args.get('max_price', type=float)

        if not destination:
            return jsonify({'error': 'Destination is required'}), 400

        tool = AccommodationSearchTool()
        result = tool._run(
            destination=destination,
            accommodation_type=acc_type,
            max_price=max_price
        )

        return jsonify({
            'accommodations': json.loads(result),
            'destination': destination
        })

    @app.route('/api/activities/search', methods=['GET'])
    def search_activities():
        """Search activities for a destination."""
        destination = request.args.get('destination', '')
        activity_type = request.args.get('type')
        max_duration = request.args.get('max_duration', type=float)

        if not destination:
            return jsonify({'error': 'Destination is required'}), 400

        tool = ActivitySearchTool()
        result = tool._run(
            destination=destination,
            activity_type=activity_type,
            max_duration=max_duration
        )

        return jsonify({
            'activities': json.loads(result),
            'destination': destination
        })

    @app.route('/api/budget/calculate', methods=['POST'])
    def calculate_budget():
        """Calculate trip budget."""
        data = request.json
        destination = data.get('destination', '')
        days = data.get('days', 5)
        budget_level = data.get('budget_level', 'moderate')
        travelers = data.get('travelers', 1)

        if not destination:
            return jsonify({'error': 'Destination is required'}), 400

        tool = BudgetCalculatorTool()
        result = tool._run(
            destination=destination,
            days=days,
            accommodation_budget=budget_level,
            travelers=travelers
        )

        return jsonify(json.loads(result))

    @app.route('/api/weather', methods=['GET'])
    def get_weather():
        """Get weather forecast for destination."""
        destination = request.args.get('destination', '')
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

        if not destination:
            return jsonify({'error': 'Destination is required'}), 400

        tool = WeatherForecastTool()
        result = tool._run(destination=destination, date=date)

        return jsonify(json.loads(result))

    @app.route('/api/session/reset', methods=['POST'])
    def reset_session():
        """Reset a session's context."""
        data = request.json
        session_id = data.get('session_id', 'default')

        if session_id in agents:
            agents[session_id].reset_context()
            del agents[session_id]

        return jsonify({
            'status': 'reset',
            'session_id': session_id
        })

    @app.route('/api/session/context', methods=['GET'])
    def get_session_context():
        """Get current session context."""
        session_id = request.args.get('session_id', 'default')

        if session_id in agents:
            return jsonify({
                'context': agents[session_id].get_context_summary(),
                'session_id': session_id
            })

        return jsonify({
            'context': {},
            'session_id': session_id
        })

    return app


# Create default app instance
app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
