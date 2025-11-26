# AI Travel Planner

LangChain-powered intelligent travel planning assistant that helps you plan your perfect trip.

## Live Demo

[**View Demo**](https://yoon-gu.github.io/langchain-travel-planner/)

## Features

- **Destination Discovery**: Search and compare travel destinations based on budget, season, and interests
- **Smart Recommendations**: AI-powered suggestions for accommodations, activities, and restaurants
- **Itinerary Generation**: Automatically create day-by-day travel plans
- **Budget Calculator**: Estimate trip costs with detailed breakdowns
- **Weather Integration**: Get weather forecasts and packing recommendations
- **Multi-turn Conversations**: Natural conversation flow with context awareness

## Architecture

```
langchain-travel-planner/
├── app/
│   ├── agents/
│   │   └── travel_agent.py      # Main LangChain agent
│   ├── chains/
│   │   └── planning_chains.py   # Itinerary generation chains
│   ├── tools/
│   │   └── travel_tools.py      # Custom LangChain tools
│   └── api.py                   # Flask API endpoints
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
└── requirements.txt
```

## LangChain Components

### Custom Tools
- `DestinationSearchTool`: Search destinations by query, budget, season
- `AccommodationSearchTool`: Find hotels, hostels, apartments
- `ActivitySearchTool`: Discover activities and attractions
- `TransportationSearchTool`: Search flights, trains, buses
- `WeatherForecastTool`: Get weather forecasts
- `BudgetCalculatorTool`: Calculate trip budgets

### Agent Architecture
```python
from app.agents.travel_agent import create_travel_agent

# Create agent
agent = create_travel_agent()

# Chat with context awareness
response = agent.chat("I want to visit Tokyo for 5 days")
response = agent.chat("Find me a mid-range hotel")
response = agent.chat("Create my itinerary")
```

### Chain Pipeline
```python
from app.chains.planning_chains import ItineraryGenerator, TravelPreferences

generator = ItineraryGenerator()
preferences = TravelPreferences(
    destination="Tokyo",
    start_date="2024-04-01",
    duration_days=5,
    budget_level="moderate",
    travelers=2,
    interests=["cultural", "food"],
    pace="moderate",
    accommodation_type="hotel"
)

itinerary = generator.generate_itinerary(
    preferences=preferences,
    destination_info=destination_data,
    activities=activities_list
)
```

## Installation

```bash
# Clone repository
git clone https://github.com/yoon-gu/langchain-travel-planner.git
cd langchain-travel-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional, for API features)
export OPENAI_API_KEY=your_key_here

# Run the application
python -m app.api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat endpoint |
| `/api/destinations` | GET | List all destinations |
| `/api/destinations/search` | GET | Search destinations |
| `/api/accommodations/search` | GET | Search accommodations |
| `/api/activities/search` | GET | Search activities |
| `/api/budget/calculate` | POST | Calculate trip budget |
| `/api/weather` | GET | Get weather forecast |

## Tech Stack

- **LangChain**: Agent framework and tool orchestration
- **OpenAI GPT-4**: Language model (optional, works offline with mock data)
- **Flask**: Backend API server
- **Python 3.9+**: Core language
- **Pydantic**: Data validation
- **FAISS**: Vector similarity (for RAG features)

## Supported Destinations

Currently includes detailed data for:
- Tokyo, Japan
- Paris, France
- Seoul, South Korea
- New York, USA
- Barcelona, Spain
- Bangkok, Thailand
- Rome, Italy
- Sydney, Australia
- Singapore
- London, UK
- Dubai, UAE
- Bali, Indonesia

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for learning and development.
