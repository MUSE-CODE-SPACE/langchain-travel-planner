# AI Travel Planner

LangChain-powered intelligent travel planning assistant that helps you discover
destinations, build itineraries, and estimate budgets.

## Live Demo

[**View Demo**](https://yoon-k.github.io/langchain-travel-planner/)

## Features

- **Destination Discovery**: Search and compare destinations by budget, season, and interests
- **Accommodation Search**: Hotels, hostels, apartments, and resorts with filters
- **Activity Discovery**: Tours, attractions, food experiences, and more
- **Transportation**: Compare flights, trains, and buses between cities
- **Weather & Packing**: Forecasts and packing recommendations per season
- **Budget Estimation**: Detailed cost breakdowns by category
- **Itinerary Generation**: Day-by-day travel plans tailored to your preferences
- **Multi-turn Conversations**: Natural conversation flow with context awareness

## Quick Start

### Using `pip`

```bash
git clone https://github.com/yoon-k/langchain-travel-planner.git
cd langchain-travel-planner

python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Editable install with dev extras (pytest, ruff, mypy).
pip install -e ".[dev]"

# (optional) configure LLM keys
cp .env.example .env

# Start the dev server
python -m app.api
```

### Using `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m app.api
```

### Using `make`

```bash
make install
make dev
make test
make lint
```

## LLM Modes

The agent supports two execution modes and selects one automatically at startup:

| Mode | When it activates | Behavior |
|------|-------------------|----------|
| **LLM-backed** | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set | LangChain LCEL `create_tool_calling_agent` + `AgentExecutor`, with travel tools bound to the model |
| **Keyword router (fallback)** | No API keys present, or `LLM_PROVIDER=none` | Pure-Python rule-based responses, no external calls. Useful for demos, tests, and CI |

You can force a provider with `LLM_PROVIDER=anthropic|openai|none`. Default
models are `claude-haiku-4-5-20251001` (Anthropic) and `gpt-4o-mini` (OpenAI),
overridable via `ANTHROPIC_MODEL` / `OPENAI_MODEL`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | auto-detect | `anthropic`, `openai`, or `none` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key for the Claude provider |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Override the Claude model id |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Override the OpenAI model id |
| `PORT` | `5000` | Port for the Flask dev server / gunicorn |
| `FLASK_DEBUG` | `1` | Set to `0` to disable Werkzeug debug mode |

A ready-to-edit template lives at `.env.example`.

## Docker

```bash
# Build the image (multi-stage, runs as non-root, gunicorn 2 workers)
make docker-build

# Run with your local .env mounted
make docker-run

# ...or directly
docker build -t travel-planner:dev .
docker run --rm -p 5000:5000 --env-file .env travel-planner:dev
```

The image is based on `python:3.12-slim`, installs runtime deps from
`requirements.txt`, drops to a non-root `app` user, and serves the Flask app
via gunicorn.

## Architecture

```
langchain-travel-planner/
├── app/
│   ├── agents/
│   │   └── travel_agent.py        # Main travel-planning agent (LLM or fallback)
│   ├── chains/
│   │   └── planning_chains.py     # Itinerary generation helpers
│   ├── tools/
│   │   └── travel_tools.py        # LangChain BaseTool implementations
│   └── api.py                     # Flask API endpoints
├── tests/
│   └── test_smoke.py              # Smoke tests (no API keys required)
├── docs/                          # Static GitHub Pages demo
├── Dockerfile                     # Production-ready image (gunicorn)
├── Makefile                       # install / dev / test / lint / docker
├── pyproject.toml                 # PEP 621 metadata + ruff/pytest/mypy config
└── requirements.txt               # Pinned runtime dependencies
```

## LangChain Components

### Custom Tools
- `DestinationSearchTool`: Search destinations by query, budget, season
- `AccommodationSearchTool`: Find hotels, hostels, apartments, resorts
- `ActivitySearchTool`: Discover activities and attractions
- `TransportationSearchTool`: Search flights, trains, buses
- `WeatherForecastTool`: Get weather forecasts
- `BudgetCalculatorTool`: Calculate trip budgets

### Agent Architecture

```python
from app.agents.travel_agent import create_travel_agent

agent = create_travel_agent()       # picks LLM if a key is present
print(agent.llm_enabled)            # True / False
print(agent.chat("I want to visit Tokyo for 5 days"))
print(agent.chat("Find me a mid-range hotel"))
print(agent.chat("Create my itinerary"))
```

### Itinerary Generator

```python
from app.chains.planning_chains import ItineraryGenerator, TravelPreferences

generator = ItineraryGenerator()
preferences = TravelPreferences(
    destination="Tokyo",
    start_date="2026-04-01",
    duration_days=5,
    budget_level="moderate",
    travelers=2,
    interests=["cultural", "food"],
    pace="moderate",
    accommodation_type="hotel",
)
itinerary = generator.generate_itinerary(
    preferences=preferences,
    destination_info={},
    activities=[],
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health probe (includes `llm_enabled`) |
| `/api/chat` | POST | Main chat endpoint |
| `/api/destinations` | GET | List all destinations |
| `/api/destinations/search` | GET | Search destinations with filters |
| `/api/accommodations/search` | GET | Search accommodations |
| `/api/activities/search` | GET | Search activities |
| `/api/budget/calculate` | POST | Calculate trip budget |
| `/api/weather` | GET | Get weather forecast |
| `/api/itinerary` | POST | Generate a full itinerary from session context |
| `/api/session/reset` | POST | Reset an agent session |
| `/api/session/context` | GET | Inspect the current session context |

## Supported Destinations

Tokyo, Paris, Seoul, New York, Barcelona, Bangkok, Rome, Sydney, Singapore,
London, Dubai, Bali (plus generic fallbacks for any other destination).

## Tech Stack

- **LangChain 0.3** (`langchain`, `langchain-core`, `langchain-community`)
- **Provider SDKs**: `langchain-anthropic`, `langchain-openai`
- **Flask 3.1** + `flask-cors`
- **Pydantic v2** for tool input schemas
- **Python 3.12** (PEP 585 / PEP 604 typing throughout)
- **Tooling**: `ruff`, `pytest`, `mypy`, `gunicorn`, Docker

## Contributing

Contributions are welcome! Feel free to add new destinations, tools, or
features. Please run `make lint` and `make test` before opening a PR.

## License

MIT License - feel free to use this project for learning and development.

---

## 한국어

LangChain 0.3 기반의 여행 플래너입니다. 목적지 추천, 숙소/활동 검색, 예산
계산, 일정 자동 생성, 날씨 안내 등을 지원하며, API 키가 있으면 Anthropic
Claude / OpenAI GPT 중 자동 선택해 LLM + 툴콜링으로 응답합니다. 키가 없으면
내장 키워드 라우터로 폴백되어 외부 호출 없이도 동작합니다.

### 빠른 시작

```bash
git clone https://github.com/yoon-k/langchain-travel-planner.git
cd langchain-travel-planner

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # 필요 시 API 키 입력
python -m app.api
```

`uv` 사용자는 `uv venv --python 3.12 && uv pip install -e ".[dev]"`도 OK.
간단히 쓰려면 `make install && make dev`.

### LLM 모드

| 조건 | 동작 |
|------|------|
| `ANTHROPIC_API_KEY`만 있음 | Claude (`claude-haiku-4-5-20251001`) + 툴콜링 |
| `OPENAI_API_KEY`만 있음 | OpenAI (`gpt-4o-mini`) + 툴콜링 |
| 둘 다 있음 | Anthropic 우선 |
| 둘 다 없음 | 키워드 라우터로 폴백 (외부 호출 없음) |

`LLM_PROVIDER=anthropic|openai|none`으로 강제 지정할 수 있고,
`ANTHROPIC_MODEL` / `OPENAI_MODEL`로 모델을 바꿀 수 있습니다.

### Docker로 실행

```bash
make docker-build
make docker-run
```

이미지는 `python:3.12-slim` 기반, gunicorn 2 워커, 비루트 사용자로 실행됩니다.

### 테스트 & 린트

```bash
make test    # pytest 스모크 테스트
make lint    # ruff check
make format  # ruff format + auto-fix
```

테스트는 `LLM_PROVIDER=none`으로 실행되어 API 키 없이도 통과합니다.
