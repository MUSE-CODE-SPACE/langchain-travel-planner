"""
Itinerary generation helpers.

This module used to contain LangChain 0.1 ``LLMChain`` constructs. It has been
ported to a plain-Python helper that produces structured itineraries from
``TravelPreferences``. The Travel Planning Agent calls into ``ItineraryGenerator``
either from its keyword-router branch or as part of an LLM-guided flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class DayActivity(BaseModel):
    """Single activity in an itinerary."""
    time: str = Field(description="Start time (e.g., '09:00')")
    activity: str = Field(description="Activity name")
    duration_hours: float = Field(description="Duration in hours")
    location: str = Field(description="Location/venue name")
    cost_estimate: float = Field(description="Estimated cost in USD")
    notes: str | None = Field(default=None, description="Additional notes or tips")


class DayItinerary(BaseModel):
    """Single day itinerary."""
    day_number: int = Field(description="Day number in the trip")
    date: str = Field(description="Date (YYYY-MM-DD)")
    theme: str = Field(description="Day theme (e.g., 'Cultural Exploration')")
    activities: list[DayActivity] = Field(description="List of activities")
    meals: dict[str, str] = Field(description="Recommended meals")
    estimated_daily_cost: float = Field(
        description="Total estimated cost for the day"
    )


class TravelItinerary(BaseModel):
    """Complete travel itinerary."""
    destination: str = Field(description="Main destination")
    trip_name: str = Field(description="Custom trip name")
    start_date: str = Field(description="Trip start date")
    end_date: str = Field(description="Trip end date")
    travelers: int = Field(description="Number of travelers")
    total_days: int = Field(description="Total trip duration")
    itinerary: list[DayItinerary] = Field(description="Day by day itinerary")
    accommodation: dict[str, Any] = Field(description="Accommodation details")
    total_budget: float = Field(description="Total estimated budget")
    packing_list: list[str] = Field(description="Recommended packing items")
    important_tips: list[str] = Field(description="Important travel tips")


@dataclass
class TravelPreferences:
    """User travel preferences."""
    destination: str
    start_date: str
    duration_days: int
    budget_level: str  # budget, moderate, luxury
    travelers: int
    interests: list[str] = field(default_factory=list)
    pace: str = "moderate"
    accommodation_type: str = "hotel"


class ItineraryGenerator:
    """Generates detailed travel itineraries from preferences and activity data."""

    def __init__(self, llm: Any | None = None) -> None:
        """Initialise the generator.

        The ``llm`` argument is accepted for forward compatibility; the current
        implementation is deterministic and rule-based to keep itineraries
        reproducible and runnable without API keys.
        """
        self.llm = llm

    def generate_itinerary(
        self,
        preferences: TravelPreferences,
        destination_info: dict[str, Any],
        activities: list[dict[str, Any]],
    ) -> TravelItinerary:
        """Generate a complete travel itinerary."""
        try:
            start = datetime.strptime(preferences.start_date, "%Y-%m-%d").replace(
                tzinfo=UTC
            )
        except ValueError:
            start = datetime.now(UTC) + timedelta(days=30)

        end = start + timedelta(days=preferences.duration_days - 1)

        activity_groups = self._group_activities(activities)

        daily_itineraries: list[DayItinerary] = []
        for day in range(1, preferences.duration_days + 1):
            current_date = start + timedelta(days=day - 1)
            day_itinerary = self._generate_day(
                day, current_date, preferences, activity_groups, destination_info
            )
            daily_itineraries.append(day_itinerary)

        total_budget = sum(d.estimated_daily_cost for d in daily_itineraries)
        accommodation_cost = self._estimate_accommodation(preferences)
        total_budget += accommodation_cost

        return TravelItinerary(
            destination=preferences.destination,
            trip_name=f"{preferences.destination} {preferences.duration_days}-Day Adventure",
            start_date=preferences.start_date,
            end_date=end.strftime("%Y-%m-%d"),
            travelers=preferences.travelers,
            total_days=preferences.duration_days,
            itinerary=daily_itineraries,
            accommodation=self._get_accommodation_details(preferences),
            total_budget=round(total_budget * preferences.travelers, 2),
            packing_list=self._generate_packing_list(preferences),
            important_tips=self._generate_tips(preferences),
        )

    # ------------------------------------------------------------------
    # Day-level helpers
    # ------------------------------------------------------------------
    def _generate_day(
        self,
        day_number: int,
        date: datetime,
        preferences: TravelPreferences,
        activity_groups: dict[str, list[dict]],
        destination_info: dict[str, Any],
    ) -> DayItinerary:
        themes = self._get_day_themes(preferences)
        theme = themes[(day_number - 1) % len(themes)]

        activities = self._select_day_activities(
            theme, activity_groups, preferences
        )

        daily_cost = sum(a.cost_estimate for a in activities)
        daily_cost += self._estimate_food_cost(preferences)

        return DayItinerary(
            day_number=day_number,
            date=date.strftime("%Y-%m-%d"),
            theme=theme,
            activities=activities,
            meals=self._recommend_meals(destination_info, preferences),
            estimated_daily_cost=round(daily_cost, 2),
        )

    def _select_day_activities(
        self,
        theme: str,
        activity_groups: dict[str, list[dict]],
        preferences: TravelPreferences,
    ) -> list[DayActivity]:
        activities: list[DayActivity] = []
        current_time = 9.0

        max_activities = {"relaxed": 3, "moderate": 4, "packed": 6}.get(
            preferences.pace, 4
        )

        theme_types = {
            "Cultural Exploration": ["cultural", "sightseeing"],
            "Food & Local Experience": ["food", "cultural"],
            "Adventure Day": ["adventure", "sightseeing"],
            "Relaxation & Leisure": ["relaxation", "food"],
            "Shopping & Entertainment": ["shopping", "entertainment"],
            "Nature & Outdoors": ["nature", "adventure"],
            "Historical Discovery": ["cultural", "sightseeing"],
            "Art & Museums": ["cultural", "art"],
        }
        preferred_types = theme_types.get(theme, ["sightseeing", "cultural"])

        candidates: list[dict] = []
        for pref_type in preferred_types:
            candidates.extend(activity_groups.get(pref_type, []))
        for other_type, acts in activity_groups.items():
            if other_type not in preferred_types:
                candidates.extend(acts[:2])

        seen: set[str] = set()
        unique_candidates: list[dict] = []
        for act in candidates:
            name = act.get("name", "")
            if name and name not in seen:
                seen.add(name)
                unique_candidates.append(act)

        for act in unique_candidates:
            if len(activities) >= max_activities or current_time >= 21:
                break

            duration = float(act.get("duration_hours", 2.0))
            if activities:
                current_time += 0.5

            time_str = f"{int(current_time):02d}:{int((current_time % 1) * 60):02d}"
            activities.append(
                DayActivity(
                    time=time_str,
                    activity=act.get("name", "Activity"),
                    duration_hours=duration,
                    location=act.get("location", preferences.destination),
                    cost_estimate=float(act.get("price", 0.0)),
                    notes=act.get("best_time"),
                )
            )
            current_time += duration

        return activities

    def _group_activities(
        self, activities: list[dict]
    ) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for activity in activities:
            act_type = activity.get("type", "sightseeing")
            groups.setdefault(act_type, []).append(activity)
        return groups

    def _get_day_themes(self, preferences: TravelPreferences) -> list[str]:
        all_themes = [
            "Cultural Exploration",
            "Food & Local Experience",
            "Historical Discovery",
            "Art & Museums",
            "Nature & Outdoors",
            "Shopping & Entertainment",
            "Relaxation & Leisure",
            "Adventure Day",
        ]
        interest_theme_map = {
            "cultural": ["Cultural Exploration", "Historical Discovery", "Art & Museums"],
            "food": ["Food & Local Experience"],
            "adventure": ["Adventure Day", "Nature & Outdoors"],
            "relaxation": ["Relaxation & Leisure"],
            "shopping": ["Shopping & Entertainment"],
            "nature": ["Nature & Outdoors"],
            "art": ["Art & Museums"],
        }
        themes: list[str] = []
        for interest in preferences.interests:
            themes.extend(interest_theme_map.get(interest.lower(), []))
        for theme in all_themes:
            if theme not in themes:
                themes.append(theme)
        return themes[: preferences.duration_days] if themes else all_themes

    def _estimate_food_cost(self, preferences: TravelPreferences) -> float:
        return {"budget": 25.0, "moderate": 50.0, "luxury": 120.0}.get(
            preferences.budget_level, 50.0
        )

    def _estimate_accommodation(self, preferences: TravelPreferences) -> float:
        nightly_rates = {
            "hostel": 30.0,
            "hotel": 120.0,
            "resort": 250.0,
            "apartment": 100.0,
        }
        rate = nightly_rates.get(preferences.accommodation_type, 120.0)
        budget_mult = {"budget": 0.7, "moderate": 1.0, "luxury": 1.8}
        rate *= budget_mult.get(preferences.budget_level, 1.0)
        return rate * max(1, preferences.duration_days - 1)

    def _get_accommodation_details(
        self, preferences: TravelPreferences
    ) -> dict[str, Any]:
        type_descriptions = {
            "hostel": "Budget-friendly hostel with social atmosphere",
            "hotel": "Comfortable hotel with good amenities",
            "resort": "Luxury resort with full services",
            "apartment": "Private apartment for more space and flexibility",
        }
        nightly_total = self._estimate_accommodation(preferences)
        nights = max(1, preferences.duration_days - 1)
        return {
            "type": preferences.accommodation_type,
            "description": type_descriptions.get(
                preferences.accommodation_type, "Comfortable accommodation"
            ),
            "recommended_area": f"Central {preferences.destination}",
            "check_in": "15:00",
            "check_out": "11:00",
            "estimated_nightly_rate": round(nightly_total / nights, 2),
        }

    def _recommend_meals(
        self,
        destination_info: dict[str, Any],
        preferences: TravelPreferences,
    ) -> dict[str, str]:
        destination = preferences.destination.lower()
        meal_recommendations = {
            "tokyo": {
                "breakfast": "Visit a local kissaten (coffee shop) for morning set",
                "lunch": "Try ramen or a bento box",
                "dinner": "Experience izakaya dining or sushi",
            },
            "paris": {
                "breakfast": "Croissant and café crème at a local boulangerie",
                "lunch": "Bistro lunch with prix fixe menu",
                "dinner": "Fine dining or classic French brasserie",
            },
            "seoul": {
                "breakfast": "Korean breakfast at guesthouse or local restaurant",
                "lunch": "Bibimbap or Korean BBQ",
                "dinner": "Korean fried chicken with beer (chimaek)",
            },
            "bangkok": {
                "breakfast": "Street food breakfast or hotel buffet",
                "lunch": "Pad Thai or curry at local restaurant",
                "dinner": "Riverside dining or night market food tour",
            },
        }
        default_meals = {
            "breakfast": "Local café or hotel breakfast",
            "lunch": "Local restaurant near activities",
            "dinner": "Recommended restaurant in the area",
        }
        # destination_info is preserved for future use (e.g., LLM-aware recommendations).
        _ = destination_info
        return meal_recommendations.get(destination, default_meals)

    def _generate_packing_list(
        self, preferences: TravelPreferences
    ) -> list[str]:
        essentials = [
            "Passport and travel documents",
            "Phone charger and adapter",
            "Comfortable walking shoes",
            "Weather-appropriate clothing",
            "Toiletries and medications",
            "Sunglasses and sunscreen",
            "Small daypack/bag",
        ]
        interest_items = {
            "adventure": ["Athletic wear", "Water bottle", "First aid kit"],
            "cultural": ["Modest clothing for temples", "Light scarf"],
            "food": ["Antacids/digestive aids", "Wet wipes"],
            "relaxation": ["Swimwear", "Book or e-reader"],
            "shopping": ["Extra bag for purchases", "Calculator app"],
        }
        packing_list = essentials.copy()
        for interest in preferences.interests:
            packing_list.extend(interest_items.get(interest.lower(), []))
        # Preserve order while de-duplicating.
        seen: set[str] = set()
        unique: list[str] = []
        for item in packing_list:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _generate_tips(self, preferences: TravelPreferences) -> list[str]:
        destination = preferences.destination.lower()
        general_tips = [
            "Keep copies of important documents",
            "Notify your bank of travel dates",
            "Download offline maps",
            "Learn basic local phrases",
            "Check visa requirements early",
        ]
        destination_tips = {
            "tokyo": [
                "Get a Suica/Pasmo card for easy transportation",
                "Many places are cash-only - carry yen",
                "Tipping is not customary in Japan",
                "Trains stop around midnight",
                "Remove shoes when entering homes/some restaurants",
            ],
            "paris": [
                "Get a Navigo pass for unlimited metro rides",
                "Most shops close on Sundays",
                "Book popular museums in advance",
                "Be aware of pickpockets in tourist areas",
                "Try to learn basic French phrases",
            ],
            "seoul": [
                "Get a T-money card for transportation",
                "Download Naver Maps (better than Google)",
                "Many restaurants have picture menus",
                "K-beauty shopping is best in Myeongdong",
                "Tipping is not expected",
            ],
        }
        return (general_tips + destination_tips.get(destination, []))[:10]


def create_itinerary_chain(llm: Any | None = None) -> ItineraryGenerator:
    """Factory for the itinerary generator."""
    return ItineraryGenerator(llm=llm)
