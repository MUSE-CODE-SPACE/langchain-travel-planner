"""
Travel Planning Agent - LangChain Agent for comprehensive travel planning
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from app.tools.travel_tools import (
    get_travel_tools,
    DestinationSearchTool,
    AccommodationSearchTool,
    ActivitySearchTool,
    TransportationSearchTool,
    WeatherForecastTool,
    BudgetCalculatorTool,
    DESTINATIONS_DB,
    ACTIVITIES_DB
)
from app.chains.planning_chains import (
    ItineraryGenerator,
    TravelPreferences,
    TravelItinerary
)


@dataclass
class ConversationContext:
    """Maintains conversation context for multi-turn planning."""
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    budget_level: Optional[str] = None
    travelers: Optional[int] = None
    interests: List[str] = None
    pace: Optional[str] = None
    accommodation_type: Optional[str] = None
    special_requirements: List[str] = None

    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.special_requirements is None:
            self.special_requirements = []

    def is_complete(self) -> bool:
        """Check if we have enough info to generate itinerary."""
        return all([
            self.destination,
            self.duration_days,
            self.budget_level
        ])

    def get_missing_info(self) -> List[str]:
        """Get list of missing required information."""
        missing = []
        if not self.destination:
            missing.append("destination")
        if not self.duration_days:
            missing.append("trip duration")
        if not self.budget_level:
            missing.append("budget level")
        return missing

    def to_preferences(self) -> TravelPreferences:
        """Convert to TravelPreferences object."""
        return TravelPreferences(
            destination=self.destination or "",
            start_date=self.start_date or datetime.now().strftime("%Y-%m-%d"),
            duration_days=self.duration_days or 5,
            budget_level=self.budget_level or "moderate",
            travelers=self.travelers or 1,
            interests=self.interests or ["cultural", "food"],
            pace=self.pace or "moderate",
            accommodation_type=self.accommodation_type or "hotel"
        )


class TravelPlanningAgent:
    """
    Intelligent travel planning agent that can:
    - Search and recommend destinations
    - Find accommodations
    - Suggest activities
    - Create complete itineraries
    - Answer travel questions
    """

    SYSTEM_PROMPT = """You are an expert travel planning assistant with deep knowledge of destinations worldwide.
You help users plan their perfect trips by:

1. Understanding their preferences, budget, and interests
2. Recommending suitable destinations
3. Finding ideal accommodations
4. Suggesting activities and experiences
5. Creating detailed day-by-day itineraries
6. Providing practical travel tips

Guidelines:
- Ask clarifying questions when needed
- Consider budget constraints carefully
- Balance popular attractions with local experiences
- Account for travel time between locations
- Consider weather and seasons
- Suggest alternatives when appropriate
- Be enthusiastic but practical

You have access to tools for searching destinations, accommodations, activities, and calculating budgets.
Use these tools to provide accurate, helpful information.

Current conversation context will be maintained to provide personalized recommendations."""

    def __init__(self, llm=None, verbose: bool = False):
        """Initialize the travel planning agent."""
        self.llm = llm
        self.verbose = verbose
        self.context = ConversationContext()
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = get_travel_tools()
        self.itinerary_generator = ItineraryGenerator(llm=llm)

        # Setup prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Setup memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )

    def _update_context_from_message(self, message: str):
        """Extract and update context from user message."""
        message_lower = message.lower()

        # Detect destination mentions
        for dest_key, dest in DESTINATIONS_DB.items():
            if dest.name.lower() in message_lower or dest_key in message_lower:
                self.context.destination = dest.name
                break

        # Detect duration
        import re
        duration_patterns = [
            r'(\d+)\s*(?:days?|nights?)',
            r'(\d+)-day',
            r'for\s+(\d+)\s+(?:days?|nights?)'
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, message_lower)
            if match:
                self.context.duration_days = int(match.group(1))
                break

        # Detect budget level
        if any(word in message_lower for word in ['cheap', 'budget', 'affordable', 'backpack']):
            self.context.budget_level = "budget"
        elif any(word in message_lower for word in ['luxury', 'premium', 'high-end', '5-star']):
            self.context.budget_level = "luxury"
        elif any(word in message_lower for word in ['moderate', 'mid-range', 'reasonable']):
            self.context.budget_level = "moderate"

        # Detect travelers count
        traveler_patterns = [
            r'(\d+)\s*(?:people|persons?|travelers?|of us)',
            r'(?:for|with)\s+(\d+)',
            r'couple|two of us',
            r'solo|alone|myself'
        ]
        for pattern in traveler_patterns:
            if pattern in ['couple|two of us']:
                if re.search(pattern, message_lower):
                    self.context.travelers = 2
                    break
            elif pattern in ['solo|alone|myself']:
                if re.search(pattern, message_lower):
                    self.context.travelers = 1
                    break
            else:
                match = re.search(pattern, message_lower)
                if match:
                    self.context.travelers = int(match.group(1))
                    break

        # Detect interests
        interest_keywords = {
            'cultural': ['culture', 'museum', 'history', 'temple', 'heritage'],
            'food': ['food', 'culinary', 'restaurant', 'cuisine', 'eating', 'foodie'],
            'adventure': ['adventure', 'hiking', 'outdoor', 'extreme', 'active'],
            'relaxation': ['relax', 'spa', 'beach', 'peaceful', 'quiet'],
            'shopping': ['shopping', 'market', 'mall', 'souvenir'],
            'nature': ['nature', 'park', 'wildlife', 'scenic'],
            'nightlife': ['nightlife', 'bar', 'club', 'party']
        }

        for interest, keywords in interest_keywords.items():
            if any(kw in message_lower for kw in keywords):
                if interest not in self.context.interests:
                    self.context.interests.append(interest)

        # Detect pace
        if any(word in message_lower for word in ['relaxed', 'slow', 'easy', 'leisure']):
            self.context.pace = "relaxed"
        elif any(word in message_lower for word in ['packed', 'busy', 'intensive', 'maximum']):
            self.context.pace = "packed"

        # Detect accommodation type
        if any(word in message_lower for word in ['hostel', 'backpacker']):
            self.context.accommodation_type = "hostel"
        elif any(word in message_lower for word in ['resort', 'beach resort']):
            self.context.accommodation_type = "resort"
        elif any(word in message_lower for word in ['apartment', 'airbnb', 'flat']):
            self.context.accommodation_type = "apartment"
        elif any(word in message_lower for word in ['hotel']):
            self.context.accommodation_type = "hotel"

    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User's input message

        Returns:
            Agent's response string
        """
        # Update context from message
        self._update_context_from_message(user_message)

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Determine response type
        response = self._generate_response(user_message)

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _generate_response(self, user_message: str) -> str:
        """Generate appropriate response based on context and message."""
        message_lower = user_message.lower()

        # Check if user wants to generate itinerary
        itinerary_triggers = ['create itinerary', 'plan my trip', 'make itinerary',
                             'generate plan', 'full plan', 'complete plan']

        if any(trigger in message_lower for trigger in itinerary_triggers):
            return self._handle_itinerary_request()

        # Check for destination search
        if any(word in message_lower for word in ['recommend', 'suggest', 'where should', 'best place']):
            return self._handle_destination_search(user_message)

        # Check for accommodation search
        if any(word in message_lower for word in ['hotel', 'stay', 'accommodation', 'hostel', 'sleep']):
            return self._handle_accommodation_search()

        # Check for activity search
        if any(word in message_lower for word in ['activity', 'things to do', 'what to do', 'attractions', 'visit']):
            return self._handle_activity_search()

        # Check for budget inquiry
        if any(word in message_lower for word in ['cost', 'budget', 'price', 'expensive', 'afford']):
            return self._handle_budget_inquiry()

        # Check for weather inquiry
        if any(word in message_lower for word in ['weather', 'climate', 'temperature', 'rain']):
            return self._handle_weather_inquiry()

        # Default: conversational response
        return self._handle_general_inquiry(user_message)

    def _handle_destination_search(self, query: str) -> str:
        """Handle destination recommendation request."""
        tool = DestinationSearchTool()

        # Extract potential preferences from query
        budget = self.context.budget_level
        season = None

        # Check for season mentions
        seasons = {
            'spring': ['spring', 'march', 'april', 'may'],
            'summer': ['summer', 'june', 'july', 'august'],
            'autumn': ['autumn', 'fall', 'september', 'october', 'november'],
            'winter': ['winter', 'december', 'january', 'february']
        }

        query_lower = query.lower()
        for s, keywords in seasons.items():
            if any(kw in query_lower for kw in keywords):
                season = s
                break

        result = tool._run(query=query, budget=budget, season=season)
        destinations = json.loads(result)

        response = "Based on your preferences, here are some great destination options:\n\n"

        for i, dest in enumerate(destinations[:3], 1):
            response += f"**{i}. {dest['name']}, {dest['country']}**\n"
            response += f"   {dest['description']}\n"
            response += f"   - Best seasons: {', '.join(dest['best_season'])}\n"
            response += f"   - Average daily cost: ${dest['avg_daily_cost_usd']}\n"
            response += f"   - Top attractions: {', '.join(dest['top_attractions'])}\n\n"

        response += "Would you like more details about any of these destinations? "
        response += "Or let me know your preferred destination to continue planning!"

        return response

    def _handle_accommodation_search(self) -> str:
        """Handle accommodation search request."""
        if not self.context.destination:
            return "I'd love to help you find accommodation! Which city are you planning to visit?"

        tool = AccommodationSearchTool()
        result = tool._run(
            destination=self.context.destination,
            accommodation_type=self.context.accommodation_type
        )
        accommodations = json.loads(result)

        response = f"Here are accommodation options in {self.context.destination}:\n\n"

        for acc in accommodations:
            response += f"**{acc['name']}** ({acc['type'].title()})\n"
            response += f"   - ${acc['price_per_night']}/night | Rating: {acc['rating']}/5\n"
            response += f"   - Location: {acc['location']} ({acc['distance_to_center']}km from center)\n"
            response += f"   - Amenities: {', '.join(acc['amenities'][:4])}\n\n"

        response += "Would you like me to find different options or continue planning your trip?"
        return response

    def _handle_activity_search(self) -> str:
        """Handle activity search request."""
        if not self.context.destination:
            return "I'd be happy to suggest activities! Which destination are you interested in?"

        tool = ActivitySearchTool()

        # Determine activity type from interests
        activity_type = None
        if self.context.interests:
            interest_to_type = {
                'cultural': 'cultural',
                'food': 'food',
                'adventure': 'adventure',
                'relaxation': 'relaxation'
            }
            for interest in self.context.interests:
                if interest in interest_to_type:
                    activity_type = interest_to_type[interest]
                    break

        result = tool._run(
            destination=self.context.destination,
            activity_type=activity_type
        )
        activities = json.loads(result)

        response = f"Here are recommended activities in {self.context.destination}:\n\n"

        for act in activities[:6]:
            response += f"**{act['name']}** ({act['type'].title()})\n"
            response += f"   {act['description']}\n"
            response += f"   - Duration: {act['duration_hours']} hours | "
            response += f"Cost: ${act['price']}\n"
            response += f"   - Best time: {act['best_time']}\n\n"

        response += "I can include any of these in your itinerary. What catches your interest?"
        return response

    def _handle_budget_inquiry(self) -> str:
        """Handle budget-related questions."""
        if not self.context.destination or not self.context.duration_days:
            missing = []
            if not self.context.destination:
                missing.append("destination")
            if not self.context.duration_days:
                missing.append("trip duration")

            return f"To calculate a budget estimate, I need to know your {' and '.join(missing)}. Could you provide that?"

        tool = BudgetCalculatorTool()
        result = tool._run(
            destination=self.context.destination,
            days=self.context.duration_days,
            accommodation_budget=self.context.budget_level or "moderate",
            travelers=self.context.travelers or 1
        )
        budget = json.loads(result)

        response = f"Here's a budget estimate for your {budget['days']}-day trip to {budget['destination']}:\n\n"
        response += f"**Total Estimated Budget: ${budget['total_estimate']:,.2f}**\n"
        response += f"(For {budget['travelers']} traveler(s))\n\n"

        response += "**Breakdown:**\n"
        for category, amount in budget['breakdown'].items():
            category_name = category.replace('_', ' ').title()
            response += f"- {category_name}: ${amount:,.2f}\n"

        response += f"\n**Daily Average:** ${budget['daily_average']:,.2f} per person\n\n"

        response += "**Money-saving tips:**\n"
        for tip in budget['tips'][:3]:
            response += f"- {tip}\n"

        return response

    def _handle_weather_inquiry(self) -> str:
        """Handle weather-related questions."""
        if not self.context.destination:
            return "Which destination would you like weather information for?"

        tool = WeatherForecastTool()

        # Use context date or default to near future
        date = self.context.start_date or datetime.now().strftime("%Y-%m-%d")

        result = tool._run(
            destination=self.context.destination,
            date=date
        )
        weather = json.loads(result)

        response = f"Weather forecast for {weather['destination']}:\n\n"
        response += f"**Date:** {weather['date']}\n"
        response += f"**Condition:** {weather['condition']}\n"
        response += f"**Temperature:** {weather['low_temp_c']}°C - {weather['high_temp_c']}°C\n"
        response += f"**Humidity:** {weather['humidity']}%\n"
        response += f"**Chance of rain:** {weather['rain_chance']}%\n\n"
        response += f"**Packing tip:** {weather['recommendation']}"

        return response

    def _handle_itinerary_request(self) -> str:
        """Handle request to generate full itinerary."""
        # Check if we have enough info
        if not self.context.is_complete():
            missing = self.context.get_missing_info()
            return f"I need a bit more information to create your itinerary. Could you tell me:\n" + \
                   "\n".join(f"- Your {item}" for item in missing)

        # Get destination info and activities
        dest_key = self.context.destination.lower().replace(" ", "_")
        dest_info = DESTINATIONS_DB.get(dest_key, {})
        activities = ACTIVITIES_DB.get(dest_key, [])

        if not activities:
            # Use tool to get activities
            tool = ActivitySearchTool()
            result = tool._run(destination=self.context.destination)
            activities = json.loads(result)

        # Generate itinerary
        preferences = self.context.to_preferences()
        itinerary = self.itinerary_generator.generate_itinerary(
            preferences=preferences,
            destination_info=asdict(dest_info) if hasattr(dest_info, '__dataclass_fields__') else {},
            activities=activities
        )

        # Format response
        response = self._format_itinerary(itinerary)
        return response

    def _format_itinerary(self, itinerary: TravelItinerary) -> str:
        """Format itinerary as readable text."""
        response = f"# {itinerary.trip_name}\n\n"
        response += f"**Dates:** {itinerary.start_date} to {itinerary.end_date}\n"
        response += f"**Travelers:** {itinerary.travelers}\n"
        response += f"**Estimated Total Budget:** ${itinerary.total_budget:,.2f}\n\n"

        response += "---\n\n"

        for day in itinerary.itinerary:
            response += f"## Day {day.day_number}: {day.theme}\n"
            response += f"*{day.date}*\n\n"

            for activity in day.activities:
                response += f"**{activity.time}** - {activity.activity}\n"
                response += f"   📍 {activity.location}"
                if activity.cost_estimate > 0:
                    response += f" | 💰 ${activity.cost_estimate}"
                response += f" | ⏱️ {activity.duration_hours}h\n"
                if activity.notes:
                    response += f"   ℹ️ {activity.notes}\n"

            response += "\n**Meals:**\n"
            for meal_type, suggestion in day.meals.items():
                response += f"- {meal_type.title()}: {suggestion}\n"

            response += f"\n**Estimated day cost:** ${day.estimated_daily_cost:.2f}\n\n"
            response += "---\n\n"

        # Accommodation
        response += "## Accommodation\n"
        acc = itinerary.accommodation
        response += f"**Type:** {acc['type'].title()}\n"
        response += f"**Area:** {acc['recommended_area']}\n"
        response += f"**Check-in/out:** {acc['check_in']} / {acc['check_out']}\n\n"

        # Packing list
        response += "## Packing List\n"
        for item in itinerary.packing_list:
            response += f"- {item}\n"

        response += "\n## Important Tips\n"
        for tip in itinerary.important_tips:
            response += f"- {tip}\n"

        return response

    def _handle_general_inquiry(self, message: str) -> str:
        """Handle general conversational inquiries."""
        # Check current context status
        if not self.context.destination:
            return ("Hello! I'm your travel planning assistant. I can help you:\n\n"
                   "🌍 **Find perfect destinations** based on your interests\n"
                   "🏨 **Book accommodations** that fit your budget\n"
                   "🎯 **Discover activities** and must-see attractions\n"
                   "📋 **Create detailed itineraries** for your trip\n"
                   "💰 **Calculate budgets** and find money-saving tips\n\n"
                   "Where would you like to travel? Or would you like me to recommend some destinations?")

        # We have some context, provide status update
        status = f"I'm helping you plan a trip to **{self.context.destination}**.\n\n"

        if self.context.duration_days:
            status += f"📅 Duration: {self.context.duration_days} days\n"
        if self.context.budget_level:
            status += f"💰 Budget: {self.context.budget_level.title()}\n"
        if self.context.travelers:
            status += f"👥 Travelers: {self.context.travelers}\n"
        if self.context.interests:
            status += f"❤️ Interests: {', '.join(self.context.interests)}\n"

        status += "\nWhat would you like to do next?\n"
        status += "- Search for accommodations\n"
        status += "- Find activities and attractions\n"
        status += "- Check weather forecast\n"
        status += "- Calculate budget\n"
        status += "- Generate complete itinerary"

        return status

    def reset_context(self):
        """Reset conversation context."""
        self.context = ConversationContext()
        self.conversation_history = []
        self.memory.clear()

    def get_context_summary(self) -> Dict[str, Any]:
        """Get current planning context summary."""
        return asdict(self.context)


def create_travel_agent(llm=None, verbose: bool = False) -> TravelPlanningAgent:
    """Factory function to create a travel planning agent."""
    return TravelPlanningAgent(llm=llm, verbose=verbose)
