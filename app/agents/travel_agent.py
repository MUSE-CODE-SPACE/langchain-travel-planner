"""
Travel Planning Agent - LangChain agent for trip planning.

Supports two execution modes:
1. LLM-backed mode (preferred): uses an Anthropic or OpenAI chat model with
   tool-calling via LangChain's LCEL ``create_tool_calling_agent`` +
   ``AgentExecutor``. Selected automatically when an API key is present.
2. Keyword router fallback: zero-dependency rule-based responder, used when no
   LLM credentials are available. Keeps the demo functional without API keys.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.chains.planning_chains import (
    ItineraryGenerator,
    TravelItinerary,
    TravelPreferences,
)
from app.tools.travel_tools import (
    ACTIVITIES_DB,
    DESTINATIONS_DB,
    AccommodationSearchTool,
    ActivitySearchTool,
    BudgetCalculatorTool,
    DestinationSearchTool,
    WeatherForecastTool,
    get_travel_tools,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class TravelContext:
    """Maintains conversation context for multi-turn travel planning."""
    user_id: str = "default"
    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_days: int | None = None
    budget_level: str | None = None
    travelers: int | None = None
    interests: list[str] = field(default_factory=list)
    pace: str | None = None
    accommodation_type: str | None = None
    special_requirements: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if we have enough info to generate an itinerary."""
        return all([self.destination, self.duration_days, self.budget_level])

    def get_missing_info(self) -> list[str]:
        """Return a list of missing required fields."""
        missing: list[str] = []
        if not self.destination:
            missing.append("destination")
        if not self.duration_days:
            missing.append("trip duration")
        if not self.budget_level:
            missing.append("budget level")
        return missing

    def to_preferences(self) -> TravelPreferences:
        """Convert to a ``TravelPreferences`` for itinerary generation."""
        return TravelPreferences(
            destination=self.destination or "",
            start_date=self.start_date
            or datetime.now(UTC).strftime("%Y-%m-%d"),
            duration_days=self.duration_days or 5,
            budget_level=self.budget_level or "moderate",
            travelers=self.travelers or 1,
            interests=self.interests or ["cultural", "food"],
            pace=self.pace or "moderate",
            accommodation_type=self.accommodation_type or "hotel",
        )


def _resolve_llm() -> BaseChatModel | None:
    """Return a chat model based on env config, or ``None`` if unavailable.

    Selection logic:
    - ``LLM_PROVIDER`` env var, one of ``anthropic`` / ``openai`` / ``none``.
    - Otherwise auto-detect from ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``.
    - Returns ``None`` if no provider is usable; caller falls back to the
      keyword router.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if provider == "none":
        return None

    if not provider:
        if has_anthropic:
            provider = "anthropic"
        elif has_openai:
            provider = "openai"
        else:
            return None

    try:
        if provider == "anthropic":
            if not has_anthropic:
                logger.warning(
                    "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing."
                )
                return None
            from langchain_anthropic import ChatAnthropic

            model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
            return ChatAnthropic(model=model, temperature=0.3)

        if provider == "openai":
            if not has_openai:
                logger.warning(
                    "LLM_PROVIDER=openai but OPENAI_API_KEY is missing."
                )
                return None
            from langchain_openai import ChatOpenAI

            model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            return ChatOpenAI(model=model, temperature=0.3)
    except Exception as exc:
        logger.warning("Failed to construct LLM for provider %s: %s", provider, exc)
        return None

    logger.warning(
        "Unknown LLM_PROVIDER=%s; falling back to keyword router.", provider
    )
    return None


class TravelPlanningAgent:
    """Travel-planning assistant with optional LLM-backed tool calling."""

    SYSTEM_PROMPT = """You are an expert travel planning assistant with deep knowledge of destinations worldwide.

Your capabilities:
1. **Destination Discovery**: Recommend cities based on interests, budget, and season
2. **Accommodation Search**: Suggest hotels, hostels, apartments, or resorts
3. **Activity & Attraction Search**: Find tours, sights, and experiences
4. **Transportation**: Compare flights, trains, and buses between locations
5. **Weather & Packing**: Provide forecasts and packing suggestions
6. **Budget Estimation**: Break down costs by category for any trip
7. **Itinerary Generation**: Build complete day-by-day travel plans
8. **Multi-turn Context**: Remember user preferences across the conversation

Guidelines:
- Ask clarifying questions when needed (destination, dates, duration, budget)
- Consider budget constraints carefully and offer alternatives
- Balance must-see attractions with authentic local experiences
- Account for travel time between locations and realistic scheduling
- Consider weather, season, and visa requirements
- Be enthusiastic but practical and safety-conscious
- Prefer using the provided tools to ground answers in real destination data

You have access to detailed data for Tokyo, Paris, Seoul, New York, Barcelona, Bangkok,
Rome, Sydney, Singapore, London, Dubai, and Bali, plus generic fallbacks for any other
destination."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
        history_window: int = 10,
    ) -> None:
        """Initialise the travel planning agent.

        Args:
            llm: Optional pre-built chat model. If ``None``, one is auto-resolved
                from env (or the keyword router fallback is used).
            verbose: Enable agent executor verbose logging.
            history_window: Number of recent conversation turns to retain.
        """
        self.verbose = verbose
        self.context = TravelContext()
        self.conversation_history: list[dict[str, str]] = []
        self._recent_turns: deque[tuple[str, str]] = deque(maxlen=history_window * 2)

        self.tools: list[BaseTool] = get_travel_tools()
        self.llm: BaseChatModel | None = llm if llm is not None else _resolve_llm()
        self.itinerary_generator = ItineraryGenerator(llm=self.llm)

        self._agent_executor: Any | None = None
        if self.llm is not None:
            self._agent_executor = self._build_agent_executor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """``True`` if requests are answered by an LLM-backed agent."""
        return self._agent_executor is not None

    def chat(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        self._update_context_from_message(user_message)

        timestamp = datetime.now(UTC).isoformat()
        self.conversation_history.append(
            {"role": "user", "content": user_message, "timestamp": timestamp}
        )
        self._recent_turns.append(("user", user_message))

        try:
            response = self._generate_response(user_message)
        except Exception as exc:
            logger.exception("Agent failed; falling back to keyword router: %s", exc)
            response = self._keyword_response(user_message)

        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self._recent_turns.append(("assistant", response))
        return response

    def reset(self) -> None:
        """Reset session state."""
        self.context = TravelContext()
        self.conversation_history = []
        self._recent_turns.clear()

    def get_context_summary(self) -> dict[str, Any]:
        """Return the current planning context as a dict."""
        return asdict(self.context)

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------
    def _update_context_from_message(self, message: str) -> None:
        """Extract preferences from a user message into ``self.context``."""
        message_lower = message.lower()

        for dest_key, dest in DESTINATIONS_DB.items():
            if dest.name.lower() in message_lower or dest_key in message_lower:
                self.context.destination = dest.name
                break

        duration_patterns = [
            r"(\d+)\s*(?:days?|nights?)",
            r"(\d+)-day",
            r"for\s+(\d+)\s+(?:days?|nights?)",
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, message_lower)
            if match:
                self.context.duration_days = int(match.group(1))
                break

        if any(
            word in message_lower
            for word in ("cheap", "budget", "affordable", "backpack")
        ):
            self.context.budget_level = "budget"
        elif any(
            word in message_lower
            for word in ("luxury", "premium", "high-end", "5-star")
        ):
            self.context.budget_level = "luxury"
        elif any(
            word in message_lower for word in ("moderate", "mid-range", "reasonable")
        ):
            self.context.budget_level = "moderate"

        if re.search(r"couple|two of us", message_lower):
            self.context.travelers = 2
        elif re.search(r"solo|alone|myself", message_lower):
            self.context.travelers = 1
        else:
            for pattern in (
                r"(\d+)\s*(?:people|persons?|travelers?|of us)",
                r"(?:for|with)\s+(\d+)",
            ):
                match = re.search(pattern, message_lower)
                if match:
                    self.context.travelers = int(match.group(1))
                    break

        interest_keywords = {
            "cultural": ["culture", "museum", "history", "temple", "heritage"],
            "food": ["food", "culinary", "restaurant", "cuisine", "eating", "foodie"],
            "adventure": ["adventure", "hiking", "outdoor", "extreme", "active"],
            "relaxation": ["relax", "spa", "beach", "peaceful", "quiet"],
            "shopping": ["shopping", "market", "mall", "souvenir"],
            "nature": ["nature", "park", "wildlife", "scenic"],
            "nightlife": ["nightlife", "bar", "club", "party"],
        }
        for interest, keywords in interest_keywords.items():
            if (
                any(kw in message_lower for kw in keywords)
                and interest not in self.context.interests
            ):
                self.context.interests.append(interest)

        if any(
            word in message_lower for word in ("relaxed", "slow", "easy", "leisure")
        ):
            self.context.pace = "relaxed"
        elif any(
            word in message_lower
            for word in ("packed", "busy", "intensive", "maximum")
        ):
            self.context.pace = "packed"

        if any(word in message_lower for word in ("hostel", "backpacker")):
            self.context.accommodation_type = "hostel"
        elif any(word in message_lower for word in ("resort", "beach resort")):
            self.context.accommodation_type = "resort"
        elif any(word in message_lower for word in ("apartment", "airbnb", "flat")):
            self.context.accommodation_type = "apartment"
        elif "hotel" in message_lower:
            self.context.accommodation_type = "hotel"

    # ------------------------------------------------------------------
    # LLM-backed agent
    # ------------------------------------------------------------------
    def _build_agent_executor(self) -> Any:
        """Construct a LangChain tool-calling agent executor."""
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=6,
            handle_parsing_errors=True,
        )

    def _build_chat_history_messages(self) -> list[tuple[str, str]]:
        """Return prior turns formatted for LangChain message placeholders."""
        history = list(self._recent_turns)[:-1]
        return [
            ("human" if role == "user" else "ai", content) for role, content in history
        ]

    def _generate_response(self, message: str) -> str:
        """Dispatch to LLM-backed agent or keyword router."""
        if self._agent_executor is None:
            return self._keyword_response(message)

        result = self._agent_executor.invoke(
            {
                "input": message,
                "chat_history": self._build_chat_history_messages(),
            }
        )
        output = result.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for block in output:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "\n".join(parts).strip() or self._keyword_response(message)
        if isinstance(output, str) and output.strip():
            return output
        return self._keyword_response(message)

    # ------------------------------------------------------------------
    # Keyword-router fallback (used when no LLM is available)
    # ------------------------------------------------------------------
    def _keyword_response(self, message: str) -> str:
        """Generate a response using simple keyword routing."""
        message_lower = message.lower()

        itinerary_triggers = (
            "create itinerary",
            "plan my trip",
            "make itinerary",
            "generate plan",
            "full plan",
            "complete plan",
            "build itinerary",
        )
        if any(trigger in message_lower for trigger in itinerary_triggers):
            return self._handle_itinerary_request()

        if any(
            word in message_lower
            for word in ("recommend", "suggest", "where should", "best place", "destination")
        ):
            return self._handle_destination_search(message)

        if any(
            word in message_lower
            for word in ("hotel", "stay", "accommodation", "hostel", "sleep", "lodging")
        ):
            return self._handle_accommodation_search()

        if any(
            word in message_lower
            for word in (
                "activity",
                "activities",
                "things to do",
                "what to do",
                "attractions",
                "visit",
                "see",
                "tour",
            )
        ):
            return self._handle_activity_search()

        if any(
            word in message_lower
            for word in ("cost", "budget", "price", "expensive", "afford")
        ):
            return self._handle_budget_inquiry()

        if any(
            word in message_lower
            for word in ("weather", "climate", "temperature", "rain", "forecast")
        ):
            return self._handle_weather_inquiry()

        return self._handle_general_inquiry()

    # ------------------------------------------------------------------
    # Keyword-router handlers
    # ------------------------------------------------------------------
    def _handle_destination_search(self, query: str) -> str:
        tool = DestinationSearchTool()

        season: str | None = None
        seasons = {
            "spring": ("spring", "march", "april", "may"),
            "summer": ("summer", "june", "july", "august"),
            "autumn": ("autumn", "fall", "september", "october", "november"),
            "winter": ("winter", "december", "january", "february"),
        }
        query_lower = query.lower()
        for season_name, keywords in seasons.items():
            if any(kw in query_lower for kw in keywords):
                season = season_name
                break

        result = tool._run(query=query, budget=self.context.budget_level, season=season)
        destinations = json.loads(result)

        response = "Based on your preferences, here are some great destination options:\n\n"
        for i, dest in enumerate(destinations[:3], 1):
            response += f"**{i}. {dest['name']}, {dest['country']}**\n"
            response += f"   {dest['description']}\n"
            response += f"   - Best seasons: {', '.join(dest['best_season'])}\n"
            response += f"   - Average daily cost: ${dest['avg_daily_cost_usd']}\n"
            response += (
                f"   - Top attractions: {', '.join(dest['top_attractions'])}\n\n"
            )
        response += (
            "Would you like more details about any of these destinations? Or let me "
            "know your preferred destination to continue planning!"
        )
        return response

    def _handle_accommodation_search(self) -> str:
        if not self.context.destination:
            return (
                "I'd love to help you find accommodation! Which city are you "
                "planning to visit?"
            )

        tool = AccommodationSearchTool()
        result = tool._run(
            destination=self.context.destination,
            accommodation_type=self.context.accommodation_type,
        )
        accommodations = json.loads(result)

        response = f"Here are accommodation options in {self.context.destination}:\n\n"
        for acc in accommodations:
            response += f"**{acc['name']}** ({acc['type'].title()})\n"
            response += (
                f"   - ${acc['price_per_night']}/night | Rating: {acc['rating']}/5\n"
            )
            response += (
                f"   - Location: {acc['location']} "
                f"({acc['distance_to_center']}km from center)\n"
            )
            response += (
                f"   - Amenities: {', '.join(acc['amenities'][:4])}\n\n"
            )
        response += "Would you like different options or to continue planning your trip?"
        return response

    def _handle_activity_search(self) -> str:
        if not self.context.destination:
            return (
                "I'd be happy to suggest activities! Which destination are you "
                "interested in?"
            )

        tool = ActivitySearchTool()
        activity_type: str | None = None
        if self.context.interests:
            interest_to_type = {
                "cultural": "cultural",
                "food": "food",
                "adventure": "adventure",
                "relaxation": "relaxation",
            }
            for interest in self.context.interests:
                if interest in interest_to_type:
                    activity_type = interest_to_type[interest]
                    break

        result = tool._run(
            destination=self.context.destination, activity_type=activity_type
        )
        activities = json.loads(result)

        response = (
            f"Here are recommended activities in {self.context.destination}:\n\n"
        )
        for act in activities[:6]:
            response += f"**{act['name']}** ({act['type'].title()})\n"
            response += f"   {act['description']}\n"
            response += (
                f"   - Duration: {act['duration_hours']} hours | "
                f"Cost: ${act['price']}\n"
            )
            response += f"   - Best time: {act['best_time']}\n\n"
        response += "I can include any of these in your itinerary. What catches your interest?"
        return response

    def _handle_budget_inquiry(self) -> str:
        if not self.context.destination or not self.context.duration_days:
            missing: list[str] = []
            if not self.context.destination:
                missing.append("destination")
            if not self.context.duration_days:
                missing.append("trip duration")
            return (
                "To calculate a budget estimate, I need to know your "
                f"{' and '.join(missing)}. Could you provide that?"
            )

        tool = BudgetCalculatorTool()
        result = tool._run(
            destination=self.context.destination,
            days=self.context.duration_days,
            accommodation_budget=self.context.budget_level or "moderate",
            travelers=self.context.travelers or 1,
        )
        budget = json.loads(result)

        response = (
            f"Here's a budget estimate for your {budget['days']}-day trip to "
            f"{budget['destination']}:\n\n"
        )
        response += f"**Total Estimated Budget: ${budget['total_estimate']:,.2f}**\n"
        response += f"(For {budget['travelers']} traveler(s))\n\n"
        response += "**Breakdown:**\n"
        for category, amount in budget["breakdown"].items():
            category_name = category.replace("_", " ").title()
            response += f"- {category_name}: ${amount:,.2f}\n"
        response += (
            f"\n**Daily Average:** ${budget['daily_average']:,.2f} per person\n\n"
        )
        response += "**Money-saving tips:**\n"
        for tip in budget["tips"][:3]:
            response += f"- {tip}\n"
        return response

    def _handle_weather_inquiry(self) -> str:
        if not self.context.destination:
            return "Which destination would you like weather information for?"

        tool = WeatherForecastTool()
        date = self.context.start_date or datetime.now(UTC).strftime("%Y-%m-%d")
        result = tool._run(destination=self.context.destination, date=date)
        weather = json.loads(result)

        response = f"Weather forecast for {weather['destination']}:\n\n"
        response += f"**Date:** {weather['date']}\n"
        response += f"**Condition:** {weather['condition']}\n"
        response += (
            f"**Temperature:** {weather['low_temp_c']}C - "
            f"{weather['high_temp_c']}C\n"
        )
        response += f"**Humidity:** {weather['humidity']}%\n"
        response += f"**Chance of rain:** {weather['rain_chance']}%\n\n"
        response += f"**Packing tip:** {weather['recommendation']}"
        return response

    def _handle_itinerary_request(self) -> str:
        if not self.context.is_complete():
            missing = self.context.get_missing_info()
            return (
                "I need a bit more information to create your itinerary. Could you "
                "tell me:\n" + "\n".join(f"- Your {item}" for item in missing)
            )

        dest_key = (self.context.destination or "").lower().replace(" ", "_")
        dest_info = DESTINATIONS_DB.get(dest_key)
        dest_info_dict = asdict(dest_info) if dest_info is not None else {}

        activities = ACTIVITIES_DB.get(dest_key)
        if not activities:
            tool = ActivitySearchTool()
            result = tool._run(destination=self.context.destination or "")
            activities_data = json.loads(result)
        else:
            activities_data = [asdict(a) for a in activities]

        preferences = self.context.to_preferences()
        itinerary = self.itinerary_generator.generate_itinerary(
            preferences=preferences,
            destination_info=dest_info_dict,
            activities=activities_data,
        )
        return self._format_itinerary(itinerary)

    def _format_itinerary(self, itinerary: TravelItinerary) -> str:
        response = f"# {itinerary.trip_name}\n\n"
        response += (
            f"**Dates:** {itinerary.start_date} to {itinerary.end_date}\n"
        )
        response += f"**Travelers:** {itinerary.travelers}\n"
        response += (
            f"**Estimated Total Budget:** ${itinerary.total_budget:,.2f}\n\n"
        )
        response += "---\n\n"

        for day in itinerary.itinerary:
            response += f"## Day {day.day_number}: {day.theme}\n"
            response += f"*{day.date}*\n\n"
            for activity in day.activities:
                response += f"**{activity.time}** - {activity.activity}\n"
                response += f"   Location: {activity.location}"
                if activity.cost_estimate > 0:
                    response += f" | Cost: ${activity.cost_estimate}"
                response += f" | Duration: {activity.duration_hours}h\n"
                if activity.notes:
                    response += f"   Note: {activity.notes}\n"
            response += "\n**Meals:**\n"
            for meal_type, suggestion in day.meals.items():
                response += f"- {meal_type.title()}: {suggestion}\n"
            response += (
                f"\n**Estimated day cost:** ${day.estimated_daily_cost:.2f}\n\n"
            )
            response += "---\n\n"

        response += "## Accommodation\n"
        acc = itinerary.accommodation
        response += f"**Type:** {acc['type'].title()}\n"
        response += f"**Area:** {acc['recommended_area']}\n"
        response += (
            f"**Check-in/out:** {acc['check_in']} / {acc['check_out']}\n\n"
        )

        response += "## Packing List\n"
        for item in itinerary.packing_list:
            response += f"- {item}\n"

        response += "\n## Important Tips\n"
        for tip in itinerary.important_tips:
            response += f"- {tip}\n"
        return response

    def _handle_general_inquiry(self) -> str:
        if not self.context.destination:
            return (
                "# Travel Planner\n\n"
                "Hello! I'm your AI travel planning assistant. I can help you:\n\n"
                "- **Find perfect destinations** based on your interests and budget\n"
                "- **Book accommodations** that fit your style (hotels, hostels, apartments, resorts)\n"
                "- **Discover activities** and must-see attractions\n"
                "- **Create detailed itineraries** day by day\n"
                "- **Calculate budgets** and share money-saving tips\n"
                "- **Check weather** and suggest what to pack\n\n"
                "---\n"
                "**Try asking:**\n"
                "- *\"Recommend a destination for a 7-day cultural trip\"*\n"
                "- *\"Show me hotels in Tokyo\"*\n"
                "- *\"What's the budget for 5 days in Paris for 2 people?\"*\n"
                "- *\"Create my itinerary\"*\n"
            )

        status = (
            f"I'm helping you plan a trip to **{self.context.destination}**.\n\n"
        )
        if self.context.duration_days:
            status += f"- Duration: {self.context.duration_days} days\n"
        if self.context.budget_level:
            status += f"- Budget: {self.context.budget_level.title()}\n"
        if self.context.travelers:
            status += f"- Travelers: {self.context.travelers}\n"
        if self.context.interests:
            status += f"- Interests: {', '.join(self.context.interests)}\n"

        status += "\nWhat would you like to do next?\n"
        status += "- Search for accommodations\n"
        status += "- Find activities and attractions\n"
        status += "- Check weather forecast\n"
        status += "- Calculate budget\n"
        status += "- Generate complete itinerary"
        return status


def create_travel_agent(
    llm: BaseChatModel | None = None, verbose: bool = False
) -> TravelPlanningAgent:
    """Factory function to create a travel planning agent."""
    return TravelPlanningAgent(llm=llm, verbose=verbose)
