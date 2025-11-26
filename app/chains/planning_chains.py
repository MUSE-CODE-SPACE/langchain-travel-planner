"""
Travel Planning Chains - LangChain chains for itinerary generation
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class DayActivity(BaseModel):
    """Single activity in an itinerary."""
    time: str = Field(description="Start time (e.g., '09:00')")
    activity: str = Field(description="Activity name")
    duration_hours: float = Field(description="Duration in hours")
    location: str = Field(description="Location/venue name")
    cost_estimate: float = Field(description="Estimated cost in USD")
    notes: Optional[str] = Field(default=None, description="Additional notes or tips")


class DayItinerary(BaseModel):
    """Single day itinerary."""
    day_number: int = Field(description="Day number in the trip")
    date: str = Field(description="Date (YYYY-MM-DD)")
    theme: str = Field(description="Day theme (e.g., 'Cultural Exploration')")
    activities: List[DayActivity] = Field(description="List of activities")
    meals: Dict[str, str] = Field(description="Recommended meals")
    estimated_daily_cost: float = Field(description="Total estimated cost for the day")


class TravelItinerary(BaseModel):
    """Complete travel itinerary."""
    destination: str = Field(description="Main destination")
    trip_name: str = Field(description="Custom trip name")
    start_date: str = Field(description="Trip start date")
    end_date: str = Field(description="Trip end date")
    travelers: int = Field(description="Number of travelers")
    total_days: int = Field(description="Total trip duration")
    itinerary: List[DayItinerary] = Field(description="Day by day itinerary")
    accommodation: Dict[str, Any] = Field(description="Accommodation details")
    total_budget: float = Field(description="Total estimated budget")
    packing_list: List[str] = Field(description="Recommended packing items")
    important_tips: List[str] = Field(description="Important travel tips")


@dataclass
class TravelPreferences:
    """User travel preferences."""
    destination: str
    start_date: str
    duration_days: int
    budget_level: str  # budget, moderate, luxury
    travelers: int
    interests: List[str]  # cultural, food, adventure, relaxation, shopping
    pace: str  # relaxed, moderate, packed
    accommodation_type: str  # hostel, hotel, resort, apartment


class ItineraryGenerator:
    """Generates detailed travel itineraries using LangChain."""

    def __init__(self, llm=None):
        """Initialize the generator with optional LLM."""
        self.llm = llm
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates."""
        self.itinerary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel planner with deep knowledge of destinations worldwide.
Create detailed, practical itineraries that balance must-see attractions with local experiences.
Consider travel time between locations and realistic scheduling.
Always include meal recommendations and budget estimates."""),
            ("human", """Create a detailed {duration_days}-day itinerary for {destination}.

Travel Preferences:
- Budget Level: {budget_level}
- Interests: {interests}
- Travel Pace: {pace}
- Number of Travelers: {travelers}
- Start Date: {start_date}

Destination Information:
{destination_info}

Available Activities:
{activities_info}

Please create a day-by-day itinerary with:
1. Specific times for each activity
2. Location details
3. Cost estimates
4. Meal recommendations
5. Transportation tips between locations

Format as a structured itinerary.""")
        ])

        self.optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel logistics expert. Optimize itineraries for:
- Minimal travel time between locations
- Logical geographical flow
- Avoiding crowds at popular attractions
- Best times for specific activities"""),
            ("human", """Optimize this itinerary for {destination}:

{draft_itinerary}

Consider:
1. Opening hours and best visit times
2. Geographic proximity of activities
3. Crowd patterns
4. Weather considerations for {season}

Provide optimized schedule.""")
        ])

    def generate_itinerary(
        self,
        preferences: TravelPreferences,
        destination_info: Dict[str, Any],
        activities: List[Dict[str, Any]]
    ) -> TravelItinerary:
        """Generate a complete travel itinerary."""

        # Parse start date
        try:
            start = datetime.strptime(preferences.start_date, "%Y-%m-%d")
        except ValueError:
            start = datetime.now() + timedelta(days=30)

        end = start + timedelta(days=preferences.duration_days - 1)

        # Generate day-by-day itinerary
        daily_itineraries = []
        daily_cost_estimate = self._calculate_daily_budget(preferences)

        # Group activities by type for balanced distribution
        activity_groups = self._group_activities(activities, preferences.interests)

        for day in range(1, preferences.duration_days + 1):
            current_date = start + timedelta(days=day - 1)
            day_itinerary = self._generate_day(
                day,
                current_date,
                preferences,
                activity_groups,
                destination_info
            )
            daily_itineraries.append(day_itinerary)

        # Calculate total budget
        total_budget = sum(d.estimated_daily_cost for d in daily_itineraries)
        accommodation_cost = self._estimate_accommodation(preferences)
        total_budget += accommodation_cost

        # Create complete itinerary
        itinerary = TravelItinerary(
            destination=preferences.destination,
            trip_name=f"{preferences.destination} {preferences.duration_days}-Day Adventure",
            start_date=preferences.start_date,
            end_date=end.strftime("%Y-%m-%d"),
            travelers=preferences.travelers,
            total_days=preferences.duration_days,
            itinerary=daily_itineraries,
            accommodation=self._get_accommodation_details(preferences),
            total_budget=round(total_budget * preferences.travelers, 2),
            packing_list=self._generate_packing_list(destination_info, preferences),
            important_tips=self._generate_tips(destination_info, preferences)
        )

        return itinerary

    def _generate_day(
        self,
        day_number: int,
        date: datetime,
        preferences: TravelPreferences,
        activity_groups: Dict[str, List[Dict]],
        destination_info: Dict[str, Any]
    ) -> DayItinerary:
        """Generate single day itinerary."""

        # Determine day theme based on day number and preferences
        themes = self._get_day_themes(preferences)
        theme = themes[(day_number - 1) % len(themes)]

        # Select activities for the day
        activities = self._select_day_activities(
            day_number,
            theme,
            activity_groups,
            preferences
        )

        # Calculate daily cost
        daily_cost = sum(a.cost_estimate for a in activities)

        # Add food costs
        food_cost = self._estimate_food_cost(preferences)
        daily_cost += food_cost

        return DayItinerary(
            day_number=day_number,
            date=date.strftime("%Y-%m-%d"),
            theme=theme,
            activities=activities,
            meals=self._recommend_meals(theme, destination_info, preferences),
            estimated_daily_cost=round(daily_cost, 2)
        )

    def _select_day_activities(
        self,
        day_number: int,
        theme: str,
        activity_groups: Dict[str, List[Dict]],
        preferences: TravelPreferences
    ) -> List[DayActivity]:
        """Select activities for a single day."""

        activities = []
        current_time = 9  # Start at 9 AM

        # Determine number of activities based on pace
        max_activities = {
            "relaxed": 3,
            "moderate": 4,
            "packed": 6
        }.get(preferences.pace, 4)

        # Theme to activity type mapping
        theme_types = {
            "Cultural Exploration": ["cultural", "sightseeing"],
            "Food & Local Experience": ["food", "cultural"],
            "Adventure Day": ["adventure", "sightseeing"],
            "Relaxation & Leisure": ["relaxation", "food"],
            "Shopping & Entertainment": ["shopping", "entertainment"],
            "Nature & Outdoors": ["nature", "adventure"],
            "Historical Discovery": ["cultural", "sightseeing"],
            "Art & Museums": ["cultural", "art"]
        }

        preferred_types = theme_types.get(theme, ["sightseeing", "cultural"])

        # Collect candidate activities
        candidates = []
        for pref_type in preferred_types:
            candidates.extend(activity_groups.get(pref_type, []))

        # Add some variety from other categories
        for other_type, acts in activity_groups.items():
            if other_type not in preferred_types:
                candidates.extend(acts[:2])

        # Remove duplicates and shuffle
        seen = set()
        unique_candidates = []
        for act in candidates:
            if act.get("name") not in seen:
                seen.add(act.get("name"))
                unique_candidates.append(act)

        # Select activities fitting the schedule
        selected_count = 0
        for act in unique_candidates:
            if selected_count >= max_activities:
                break

            if current_time >= 21:  # Don't schedule past 9 PM
                break

            duration = act.get("duration_hours", 2.0)

            # Add break time between activities
            if activities:
                current_time += 0.5

            time_str = f"{int(current_time):02d}:{int((current_time % 1) * 60):02d}"

            activities.append(DayActivity(
                time=time_str,
                activity=act.get("name", "Activity"),
                duration_hours=duration,
                location=act.get("location", preferences.destination),
                cost_estimate=act.get("price", 0.0),
                notes=act.get("best_time", None)
            ))

            current_time += duration
            selected_count += 1

        return activities

    def _group_activities(
        self,
        activities: List[Dict],
        interests: List[str]
    ) -> Dict[str, List[Dict]]:
        """Group activities by type."""
        groups: Dict[str, List[Dict]] = {}

        for activity in activities:
            act_type = activity.get("type", "sightseeing")
            if act_type not in groups:
                groups[act_type] = []
            groups[act_type].append(activity)

        return groups

    def _get_day_themes(self, preferences: TravelPreferences) -> List[str]:
        """Get day themes based on interests."""
        all_themes = [
            "Cultural Exploration",
            "Food & Local Experience",
            "Historical Discovery",
            "Art & Museums",
            "Nature & Outdoors",
            "Shopping & Entertainment",
            "Relaxation & Leisure",
            "Adventure Day"
        ]

        # Prioritize themes based on interests
        interest_theme_map = {
            "cultural": ["Cultural Exploration", "Historical Discovery", "Art & Museums"],
            "food": ["Food & Local Experience"],
            "adventure": ["Adventure Day", "Nature & Outdoors"],
            "relaxation": ["Relaxation & Leisure"],
            "shopping": ["Shopping & Entertainment"],
            "nature": ["Nature & Outdoors"],
            "art": ["Art & Museums"]
        }

        themes = []
        for interest in preferences.interests:
            themes.extend(interest_theme_map.get(interest.lower(), []))

        # Add variety
        for theme in all_themes:
            if theme not in themes:
                themes.append(theme)

        return themes[:preferences.duration_days] if themes else all_themes

    def _calculate_daily_budget(self, preferences: TravelPreferences) -> float:
        """Calculate estimated daily budget."""
        base_costs = {
            "budget": 80.0,
            "moderate": 150.0,
            "luxury": 350.0
        }
        return base_costs.get(preferences.budget_level, 150.0)

    def _estimate_food_cost(self, preferences: TravelPreferences) -> float:
        """Estimate daily food cost."""
        food_costs = {
            "budget": 25.0,
            "moderate": 50.0,
            "luxury": 120.0
        }
        return food_costs.get(preferences.budget_level, 50.0)

    def _estimate_accommodation(self, preferences: TravelPreferences) -> float:
        """Estimate total accommodation cost."""
        nightly_rates = {
            "hostel": 30.0,
            "hotel": 120.0,
            "resort": 250.0,
            "apartment": 100.0
        }
        rate = nightly_rates.get(preferences.accommodation_type, 120.0)

        # Adjust by budget level
        budget_mult = {"budget": 0.7, "moderate": 1.0, "luxury": 1.8}
        rate *= budget_mult.get(preferences.budget_level, 1.0)

        return rate * (preferences.duration_days - 1)  # -1 for check-out day

    def _get_accommodation_details(self, preferences: TravelPreferences) -> Dict[str, Any]:
        """Get accommodation recommendation details."""
        type_descriptions = {
            "hostel": "Budget-friendly hostel with social atmosphere",
            "hotel": "Comfortable hotel with good amenities",
            "resort": "Luxury resort with full services",
            "apartment": "Private apartment for more space and flexibility"
        }

        return {
            "type": preferences.accommodation_type,
            "description": type_descriptions.get(preferences.accommodation_type, "Comfortable accommodation"),
            "recommended_area": f"Central {preferences.destination}",
            "check_in": "15:00",
            "check_out": "11:00",
            "estimated_nightly_rate": self._estimate_accommodation(preferences) / max(1, preferences.duration_days - 1)
        }

    def _recommend_meals(
        self,
        theme: str,
        destination_info: Dict[str, Any],
        preferences: TravelPreferences
    ) -> Dict[str, str]:
        """Recommend meals based on theme and destination."""
        destination = preferences.destination.lower()

        # Destination-specific recommendations
        meal_recommendations = {
            "tokyo": {
                "breakfast": "Visit a local kissaten (coffee shop) for morning set",
                "lunch": "Try ramen or a bento box",
                "dinner": "Experience izakaya dining or sushi"
            },
            "paris": {
                "breakfast": "Croissant and café crème at a local boulangerie",
                "lunch": "Bistro lunch with prix fixe menu",
                "dinner": "Fine dining or classic French brasserie"
            },
            "seoul": {
                "breakfast": "Korean breakfast at guesthouse or local restaurant",
                "lunch": "Bibimbap or Korean BBQ",
                "dinner": "Korean fried chicken with beer (chimaek)"
            },
            "bangkok": {
                "breakfast": "Street food breakfast or hotel buffet",
                "lunch": "Pad Thai or curry at local restaurant",
                "dinner": "Riverside dining or night market food tour"
            }
        }

        default_meals = {
            "breakfast": "Local café or hotel breakfast",
            "lunch": "Local restaurant near activities",
            "dinner": "Recommended restaurant in the area"
        }

        return meal_recommendations.get(destination, default_meals)

    def _generate_packing_list(
        self,
        destination_info: Dict[str, Any],
        preferences: TravelPreferences
    ) -> List[str]:
        """Generate packing list based on destination and activities."""
        essentials = [
            "Passport and travel documents",
            "Phone charger and adapter",
            "Comfortable walking shoes",
            "Weather-appropriate clothing",
            "Toiletries and medications",
            "Sunglasses and sunscreen",
            "Small daypack/bag"
        ]

        interest_items = {
            "adventure": ["Athletic wear", "Water bottle", "First aid kit"],
            "cultural": ["Modest clothing for temples", "Light scarf"],
            "food": ["Antacids/digestive aids", "Wet wipes"],
            "relaxation": ["Swimwear", "Book or e-reader"],
            "shopping": ["Extra bag for purchases", "Calculator app"]
        }

        packing_list = essentials.copy()

        for interest in preferences.interests:
            packing_list.extend(interest_items.get(interest.lower(), []))

        return list(set(packing_list))

    def _generate_tips(
        self,
        destination_info: Dict[str, Any],
        preferences: TravelPreferences
    ) -> List[str]:
        """Generate important travel tips."""
        destination = preferences.destination.lower()

        general_tips = [
            "Keep copies of important documents",
            "Notify your bank of travel dates",
            "Download offline maps",
            "Learn basic local phrases",
            "Check visa requirements early"
        ]

        destination_tips = {
            "tokyo": [
                "Get a Suica/Pasmo card for easy transportation",
                "Many places are cash-only - carry yen",
                "Tipping is not customary in Japan",
                "Trains stop around midnight",
                "Remove shoes when entering homes/some restaurants"
            ],
            "paris": [
                "Get a Navigo pass for unlimited metro rides",
                "Most shops close on Sundays",
                "Book popular museums in advance",
                "Be aware of pickpockets in tourist areas",
                "Try to learn basic French phrases"
            ],
            "seoul": [
                "Get a T-money card for transportation",
                "Download Naver Maps (better than Google)",
                "Many restaurants have picture menus",
                "K-beauty shopping is best in Myeongdong",
                "Tipping is not expected"
            ]
        }

        tips = general_tips + destination_tips.get(destination, [])
        return tips[:10]  # Limit to 10 tips


class ItineraryOptimizer:
    """Optimizes travel itineraries for better logistics."""

    def __init__(self, llm=None):
        self.llm = llm

    def optimize_route(self, itinerary: TravelItinerary) -> TravelItinerary:
        """Optimize the travel route for minimal transit time."""
        # Group nearby activities
        for day in itinerary.itinerary:
            day.activities = self._sort_by_proximity(day.activities)

        return itinerary

    def _sort_by_proximity(self, activities: List[DayActivity]) -> List[DayActivity]:
        """Sort activities by proximity (simplified)."""
        # In production, use actual geocoding and routing
        # For now, keep chronological order
        return sorted(activities, key=lambda x: x.time)

    def balance_days(self, itinerary: TravelItinerary) -> TravelItinerary:
        """Balance activity load across days."""
        # Calculate activity hours per day
        day_loads = []
        for day in itinerary.itinerary:
            total_hours = sum(a.duration_hours for a in day.activities)
            day_loads.append((day, total_hours))

        # Flag overloaded days
        avg_load = sum(h for _, h in day_loads) / len(day_loads)

        for day, hours in day_loads:
            if hours > avg_load * 1.5:
                # Remove longest activity
                if len(day.activities) > 2:
                    day.activities.sort(key=lambda x: x.duration_hours, reverse=True)
                    day.activities = day.activities[1:]  # Remove longest

        return itinerary


def create_itinerary_chain(llm=None) -> ItineraryGenerator:
    """Factory function to create itinerary generation chain."""
    return ItineraryGenerator(llm=llm)
