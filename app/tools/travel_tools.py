"""
Travel Planning Tools - LangChain tools for the travel planning assistant.

Each tool is a ``BaseTool`` subclass with a Pydantic ``args_schema`` so it can be
bound to a tool-calling LLM. The underlying data is in-memory and deterministic
(no external API calls), which keeps the demo runnable without credentials.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class Destination:
    """Represents a travel destination."""
    name: str
    country: str
    description: str
    best_season: list[str]
    attractions: list[str]
    avg_daily_cost: float
    currency: str
    language: str
    timezone: str
    visa_required: bool
    safety_rating: float  # 1-5


@dataclass
class Accommodation:
    """Represents an accommodation option."""
    name: str
    type: str  # hotel, hostel, apartment, resort
    price_per_night: float
    rating: float
    amenities: list[str]
    location: str
    distance_to_center: float


@dataclass
class Activity:
    """Represents a tourist activity."""
    name: str
    type: str  # sightseeing, adventure, cultural, food, relaxation
    duration_hours: float
    price: float
    description: str
    best_time: str
    booking_required: bool


@dataclass
class Transportation:
    """Represents a transportation option."""
    type: str  # flight, train, bus, car_rental
    from_location: str
    to_location: str
    duration_hours: float
    price: float
    provider: str
    departure_time: str


# Comprehensive destination database.
DESTINATIONS_DB: dict[str, Destination] = {
    "tokyo": Destination(
        name="Tokyo",
        country="Japan",
        description=(
            "A fascinating blend of ultra-modern and traditional, Tokyo offers ancient "
            "temples alongside neon-lit skyscrapers."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Senso-ji Temple",
            "Shibuya Crossing",
            "Meiji Shrine",
            "Tokyo Skytree",
            "Tsukiji Fish Market",
            "Akihabara",
        ],
        avg_daily_cost=150.0,
        currency="JPY",
        language="Japanese",
        timezone="Asia/Tokyo",
        visa_required=False,
        safety_rating=4.8,
    ),
    "paris": Destination(
        name="Paris",
        country="France",
        description=(
            "The City of Light captivates with its iconic landmarks, world-class "
            "museums, and exquisite cuisine."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Eiffel Tower",
            "Louvre Museum",
            "Notre-Dame",
            "Champs-Élysées",
            "Montmartre",
            "Palace of Versailles",
        ],
        avg_daily_cost=180.0,
        currency="EUR",
        language="French",
        timezone="Europe/Paris",
        visa_required=False,
        safety_rating=4.2,
    ),
    "seoul": Destination(
        name="Seoul",
        country="South Korea",
        description=(
            "Dynamic capital blending ancient palaces with K-pop culture, cutting-edge "
            "technology, and incredible food."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Gyeongbokgung Palace",
            "Bukchon Hanok Village",
            "Myeongdong",
            "N Seoul Tower",
            "Hongdae",
            "Gangnam",
        ],
        avg_daily_cost=100.0,
        currency="KRW",
        language="Korean",
        timezone="Asia/Seoul",
        visa_required=False,
        safety_rating=4.7,
    ),
    "new_york": Destination(
        name="New York City",
        country="USA",
        description=(
            "The city that never sleeps offers world-famous landmarks, Broadway shows, "
            "and diverse neighborhoods."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Statue of Liberty",
            "Central Park",
            "Times Square",
            "Empire State Building",
            "Brooklyn Bridge",
            "Metropolitan Museum",
        ],
        avg_daily_cost=250.0,
        currency="USD",
        language="English",
        timezone="America/New_York",
        visa_required=True,
        safety_rating=4.0,
    ),
    "barcelona": Destination(
        name="Barcelona",
        country="Spain",
        description=(
            "Vibrant coastal city famous for Gaudí architecture, Mediterranean beaches, "
            "and lively nightlife."
        ),
        best_season=["spring", "early_summer", "autumn"],
        attractions=[
            "Sagrada Familia",
            "Park Güell",
            "La Rambla",
            "Gothic Quarter",
            "Casa Batlló",
            "Barceloneta Beach",
        ],
        avg_daily_cost=130.0,
        currency="EUR",
        language="Spanish, Catalan",
        timezone="Europe/Madrid",
        visa_required=False,
        safety_rating=4.1,
    ),
    "bangkok": Destination(
        name="Bangkok",
        country="Thailand",
        description=(
            "Bustling capital with ornate temples, floating markets, and a legendary "
            "street food scene."
        ),
        best_season=["winter", "early_spring"],
        attractions=[
            "Grand Palace",
            "Wat Pho",
            "Chatuchak Market",
            "Khao San Road",
            "Wat Arun",
            "Jim Thompson House",
        ],
        avg_daily_cost=60.0,
        currency="THB",
        language="Thai",
        timezone="Asia/Bangkok",
        visa_required=False,
        safety_rating=4.3,
    ),
    "rome": Destination(
        name="Rome",
        country="Italy",
        description=(
            "The Eternal City offers unparalleled ancient history, Renaissance art, "
            "and authentic Italian cuisine."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Colosseum",
            "Vatican City",
            "Trevi Fountain",
            "Pantheon",
            "Roman Forum",
            "Spanish Steps",
        ],
        avg_daily_cost=140.0,
        currency="EUR",
        language="Italian",
        timezone="Europe/Rome",
        visa_required=False,
        safety_rating=4.2,
    ),
    "sydney": Destination(
        name="Sydney",
        country="Australia",
        description=(
            "Stunning harbor city with iconic architecture, beautiful beaches, and a "
            "laid-back lifestyle."
        ),
        best_season=["spring", "autumn"],
        attractions=[
            "Sydney Opera House",
            "Harbour Bridge",
            "Bondi Beach",
            "Darling Harbour",
            "Taronga Zoo",
            "The Rocks",
        ],
        avg_daily_cost=170.0,
        currency="AUD",
        language="English",
        timezone="Australia/Sydney",
        visa_required=True,
        safety_rating=4.6,
    ),
    "singapore": Destination(
        name="Singapore",
        country="Singapore",
        description=(
            "Futuristic city-state with stunning gardens, diverse food culture, and "
            "world-class shopping."
        ),
        best_season=["winter", "spring"],
        attractions=[
            "Marina Bay Sands",
            "Gardens by the Bay",
            "Sentosa Island",
            "Orchard Road",
            "Chinatown",
            "Little India",
        ],
        avg_daily_cost=160.0,
        currency="SGD",
        language="English, Mandarin, Malay, Tamil",
        timezone="Asia/Singapore",
        visa_required=False,
        safety_rating=4.9,
    ),
    "london": Destination(
        name="London",
        country="UK",
        description=(
            "Historic capital blending royal heritage with cutting-edge culture, "
            "theater, and diverse cuisine."
        ),
        best_season=["spring", "summer"],
        attractions=[
            "Tower of London",
            "British Museum",
            "Buckingham Palace",
            "Big Ben",
            "London Eye",
            "Westminster Abbey",
        ],
        avg_daily_cost=200.0,
        currency="GBP",
        language="English",
        timezone="Europe/London",
        visa_required=False,
        safety_rating=4.3,
    ),
    "dubai": Destination(
        name="Dubai",
        country="UAE",
        description=(
            "Ultra-modern metropolis with record-breaking architecture, luxury "
            "shopping, and desert adventures."
        ),
        best_season=["winter", "early_spring"],
        attractions=[
            "Burj Khalifa",
            "Dubai Mall",
            "Palm Jumeirah",
            "Dubai Marina",
            "Gold Souk",
            "Desert Safari",
        ],
        avg_daily_cost=200.0,
        currency="AED",
        language="Arabic, English",
        timezone="Asia/Dubai",
        visa_required=False,
        safety_rating=4.8,
    ),
    "bali": Destination(
        name="Bali",
        country="Indonesia",
        description=(
            "Tropical paradise with ancient temples, rice terraces, pristine beaches, "
            "and spiritual retreats."
        ),
        best_season=["spring", "summer"],
        attractions=[
            "Uluwatu Temple",
            "Ubud Rice Terraces",
            "Tanah Lot",
            "Seminyak Beach",
            "Sacred Monkey Forest",
            "Mount Batur",
        ],
        avg_daily_cost=70.0,
        currency="IDR",
        language="Indonesian, Balinese",
        timezone="Asia/Makassar",
        visa_required=False,
        safety_rating=4.4,
    ),
}

# Accommodation templates per destination.
ACCOMMODATIONS_DB: dict[str, list[Accommodation]] = {
    "tokyo": [
        Accommodation(
            "Park Hyatt Tokyo", "hotel", 450.0, 4.9,
            ["spa", "pool", "gym", "restaurant", "bar"], "Shinjuku", 0.5,
        ),
        Accommodation(
            "Shibuya Stream Excel Hotel", "hotel", 200.0, 4.5,
            ["gym", "restaurant", "wifi"], "Shibuya", 0.2,
        ),
        Accommodation(
            "Khaosan Tokyo Samurai", "hostel", 35.0, 4.2,
            ["wifi", "lounge", "kitchen"], "Asakusa", 1.0,
        ),
        Accommodation(
            "Shinjuku Granbell Hotel", "hotel", 150.0, 4.4,
            ["wifi", "restaurant"], "Shinjuku", 0.3,
        ),
        Accommodation(
            "Tokyo Bay Ariake Washington Hotel", "hotel", 120.0, 4.1,
            ["wifi", "restaurant", "shuttle"], "Odaiba", 5.0,
        ),
    ],
    "paris": [
        Accommodation(
            "The Ritz Paris", "hotel", 1200.0, 4.9,
            ["spa", "pool", "restaurant", "bar", "concierge"], "1st Arr.", 0.1,
        ),
        Accommodation(
            "Hotel Le Marais", "hotel", 180.0, 4.4,
            ["wifi", "breakfast", "bar"], "Le Marais", 0.5,
        ),
        Accommodation(
            "Generator Paris", "hostel", 40.0, 4.1,
            ["wifi", "bar", "lounge", "kitchen"], "10th Arr.", 2.0,
        ),
        Accommodation(
            "Citadines Saint-Germain", "apartment", 220.0, 4.3,
            ["kitchen", "wifi", "laundry"], "6th Arr.", 0.8,
        ),
        Accommodation(
            "Novotel Paris Centre Tour Eiffel", "hotel", 250.0, 4.2,
            ["pool", "gym", "restaurant"], "15th Arr.", 1.5,
        ),
    ],
    "seoul": [
        Accommodation(
            "The Shilla Seoul", "hotel", 350.0, 4.8,
            ["spa", "pool", "gym", "restaurant", "bar"], "Jung-gu", 1.0,
        ),
        Accommodation(
            "L7 Hongdae", "hotel", 130.0, 4.5,
            ["gym", "rooftop bar", "wifi"], "Hongdae", 0.2,
        ),
        Accommodation(
            "Zzzip Guesthouse", "hostel", 25.0, 4.3,
            ["wifi", "kitchen", "lounge"], "Jongno", 0.5,
        ),
        Accommodation(
            "Glad Hotel Mapo", "hotel", 100.0, 4.2,
            ["gym", "wifi", "restaurant"], "Mapo", 1.5,
        ),
        Accommodation(
            "Four Seasons Seoul", "hotel", 400.0, 4.9,
            ["spa", "pool", "restaurant", "concierge"], "Jongno", 0.3,
        ),
    ],
}

# Activities per destination.
ACTIVITIES_DB: dict[str, list[Activity]] = {
    "tokyo": [
        Activity(
            "Senso-ji Temple Visit", "cultural", 2.0, 0.0,
            "Ancient Buddhist temple in Asakusa with iconic Kaminarimon gate",
            "morning", False,
        ),
        Activity(
            "Tsukiji Outer Market Food Tour", "food", 3.0, 80.0,
            "Guided tour sampling fresh sushi, tamagoyaki, and street food",
            "morning", True,
        ),
        Activity(
            "TeamLab Borderless", "cultural", 3.0, 30.0,
            "Immersive digital art museum experience", "afternoon", True,
        ),
        Activity(
            "Shibuya Crossing Experience", "sightseeing", 1.0, 0.0,
            "World's busiest pedestrian crossing and shopping district",
            "evening", False,
        ),
        Activity(
            "Meiji Shrine & Harajuku Walk", "cultural", 3.0, 0.0,
            "Peaceful shrine visit followed by colorful Takeshita Street",
            "morning", False,
        ),
        Activity(
            "Robot Restaurant Show", "entertainment", 2.0, 80.0,
            "Wild cabaret show with robots and dancers in Shinjuku",
            "evening", True,
        ),
        Activity(
            "Tokyo Skytree Observation", "sightseeing", 2.0, 25.0,
            "Panoramic city views from world's tallest tower",
            "afternoon", False,
        ),
        Activity(
            "Ramen Making Class", "food", 3.0, 75.0,
            "Learn to make authentic Japanese ramen from scratch",
            "afternoon", True,
        ),
    ],
    "paris": [
        Activity(
            "Eiffel Tower Summit", "sightseeing", 2.0, 28.0,
            "Iconic tower with breathtaking city views", "afternoon", True,
        ),
        Activity(
            "Louvre Museum Tour", "cultural", 4.0, 17.0,
            "World's largest art museum with Mona Lisa", "morning", True,
        ),
        Activity(
            "Seine River Cruise", "sightseeing", 1.5, 15.0,
            "Romantic boat tour passing major landmarks", "evening", False,
        ),
        Activity(
            "Montmartre Walking Tour", "cultural", 3.0, 25.0,
            "Artistic neighborhood with Sacré-Cœur basilica", "morning", False,
        ),
        Activity(
            "French Cooking Class", "food", 4.0, 120.0,
            "Learn classic French cuisine from a Parisian chef",
            "afternoon", True,
        ),
        Activity(
            "Versailles Day Trip", "cultural", 6.0, 50.0,
            "Magnificent palace and gardens of French royalty",
            "full_day", True,
        ),
        Activity(
            "Wine Tasting in Le Marais", "food", 2.0, 60.0,
            "Sample French wines with sommelier guidance", "evening", True,
        ),
        Activity(
            "Notre-Dame & Latin Quarter", "cultural", 3.0, 0.0,
            "Gothic cathedral and historic university district",
            "afternoon", False,
        ),
    ],
    "seoul": [
        Activity(
            "Gyeongbokgung Palace Tour", "cultural", 2.5, 3.0,
            "Grand palace with changing of the guard ceremony",
            "morning", False,
        ),
        Activity(
            "Korean BBQ Dinner Experience", "food", 2.0, 35.0,
            "Traditional grilled meat with banchan sides", "evening", True,
        ),
        Activity(
            "Bukchon Hanok Village Walk", "cultural", 2.0, 0.0,
            "Traditional Korean houses in historic neighborhood",
            "afternoon", False,
        ),
        Activity(
            "K-pop Dance Class", "entertainment", 2.0, 40.0,
            "Learn choreography from popular K-pop songs",
            "afternoon", True,
        ),
        Activity(
            "DMZ Tour", "sightseeing", 6.0, 80.0,
            "Visit the border between North and South Korea",
            "full_day", True,
        ),
        Activity(
            "Myeongdong Shopping & Street Food", "food", 3.0, 0.0,
            "Cosmetics shopping and Korean street food", "evening", False,
        ),
        Activity(
            "Jjimjilbang Experience", "relaxation", 4.0, 15.0,
            "Traditional Korean bathhouse and sauna", "evening", False,
        ),
        Activity(
            "Namsan Tower & Love Locks", "sightseeing", 2.0, 12.0,
            "Iconic tower with panoramic Seoul views", "evening", False,
        ),
    ],
}


def get_all_destinations() -> list[str]:
    """Return a list of known destination keys."""
    return list(DESTINATIONS_DB.keys())


def get_destination(name_or_key: str) -> Destination | None:
    """Look up a destination by key or by display name (case-insensitive)."""
    key = name_or_key.lower().replace(" ", "_")
    if key in DESTINATIONS_DB:
        return DESTINATIONS_DB[key]
    lowered = name_or_key.lower()
    for dest in DESTINATIONS_DB.values():
        if dest.name.lower() == lowered:
            return dest
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class DestinationSearchInput(BaseModel):
    """Input for destination search."""
    query: str = Field(default="", description="Search query for destination")
    budget: str | None = Field(
        default=None, description="Budget level: budget, moderate, luxury"
    )
    season: str | None = Field(default=None, description="Preferred travel season")


class DestinationSearchTool(BaseTool):
    """Tool for searching travel destinations."""
    name: str = "destination_search"
    description: str = (
        "Search for travel destinations based on a free-text query, optional budget "
        "(budget/moderate/luxury), and optional season."
    )
    args_schema: type[BaseModel] = DestinationSearchInput

    def _run(
        self,
        query: str = "",
        budget: str | None = None,
        season: str | None = None,
    ) -> str:
        query_lower = (query or "").lower()
        results: list[tuple[int, Destination]] = []

        budget_ranges = {
            "budget": (0, 80),
            "moderate": (80, 160),
            "luxury": (160, 1000),
        }

        for dest in DESTINATIONS_DB.values():
            score = 0

            if query_lower and (
                query_lower in dest.name.lower()
                or query_lower in dest.country.lower()
            ):
                score += 10

            if query_lower and query_lower in dest.description.lower():
                score += 5

            for attraction in dest.attractions:
                if query_lower and query_lower in attraction.lower():
                    score += 3

            if budget and budget in budget_ranges:
                min_cost, max_cost = budget_ranges[budget]
                if min_cost <= dest.avg_daily_cost <= max_cost:
                    score += 5

            if season and season.lower() in [s.lower() for s in dest.best_season]:
                score += 5

            if score > 0:
                results.append((score, dest))

        results.sort(key=lambda x: x[0], reverse=True)

        if not results:
            results = [(5, dest) for dest in list(DESTINATIONS_DB.values())[:5]]

        output = [
            {
                "name": dest.name,
                "country": dest.country,
                "description": dest.description,
                "best_season": dest.best_season,
                "avg_daily_cost_usd": dest.avg_daily_cost,
                "safety_rating": dest.safety_rating,
                "top_attractions": dest.attractions[:3],
            }
            for _score, dest in results[:5]
        ]

        return json.dumps(output, indent=2)


class AccommodationSearchInput(BaseModel):
    """Input for accommodation search."""
    destination: str = Field(description="Destination city name")
    accommodation_type: str | None = Field(
        default=None, description="Type: hotel, hostel, apartment, resort"
    )
    max_price: float | None = Field(
        default=None, description="Maximum price per night in USD"
    )


class AccommodationSearchTool(BaseTool):
    """Tool for searching accommodations."""
    name: str = "accommodation_search"
    description: str = (
        "Search for hotels, hostels, apartments, or resorts at a destination."
    )
    args_schema: type[BaseModel] = AccommodationSearchInput

    def _run(
        self,
        destination: str,
        accommodation_type: str | None = None,
        max_price: float | None = None,
    ) -> str:
        dest_key = destination.lower().replace(" ", "_")
        accommodations = ACCOMMODATIONS_DB.get(dest_key)

        if not accommodations:
            accommodations = self._generate_accommodations(destination)

        results: list[dict] = []
        for acc in accommodations:
            if accommodation_type and acc.type != accommodation_type:
                continue
            if max_price is not None and acc.price_per_night > max_price:
                continue
            results.append(asdict(acc))

        if not results:
            results = [asdict(acc) for acc in accommodations[:3]]

        return json.dumps(results[:5], indent=2)

    def _generate_accommodations(self, destination: str) -> list[Accommodation]:
        """Generate generic accommodations for unknown destinations."""
        return [
            Accommodation(
                f"{destination} Grand Hotel", "hotel", 180.0, 4.5,
                ["wifi", "pool", "restaurant"], "City Center", 0.5,
            ),
            Accommodation(
                f"{destination} Budget Inn", "hotel", 80.0, 4.0,
                ["wifi", "breakfast"], "Downtown", 1.0,
            ),
            Accommodation(
                f"{destination} Backpackers", "hostel", 30.0, 4.2,
                ["wifi", "kitchen", "lounge"], "Tourist Area", 0.8,
            ),
            Accommodation(
                f"{destination} Luxury Resort", "resort", 350.0, 4.8,
                ["spa", "pool", "gym", "restaurant", "bar"], "Beachfront", 2.0,
            ),
            Accommodation(
                f"{destination} City Apartment", "apartment", 120.0, 4.3,
                ["kitchen", "wifi", "laundry"], "Residential", 1.5,
            ),
        ]


class ActivitySearchInput(BaseModel):
    """Input for activity search."""
    destination: str = Field(description="Destination city name")
    activity_type: str | None = Field(
        default=None,
        description="Type: sightseeing, cultural, food, adventure, relaxation",
    )
    max_duration: float | None = Field(
        default=None, description="Maximum duration in hours"
    )


class ActivitySearchTool(BaseTool):
    """Tool for searching activities and attractions."""
    name: str = "activity_search"
    description: str = (
        "Search for activities, tours, and attractions at a destination."
    )
    args_schema: type[BaseModel] = ActivitySearchInput

    def _run(
        self,
        destination: str,
        activity_type: str | None = None,
        max_duration: float | None = None,
    ) -> str:
        dest_key = destination.lower().replace(" ", "_")
        activities = ACTIVITIES_DB.get(dest_key)

        if not activities:
            activities = self._generate_activities(destination)

        results: list[dict] = []
        for act in activities:
            if activity_type and act.type != activity_type:
                continue
            if max_duration is not None and act.duration_hours > max_duration:
                continue
            results.append(asdict(act))

        if not results:
            results = [asdict(act) for act in activities[:5]]

        return json.dumps(results[:8], indent=2)

    def _generate_activities(self, destination: str) -> list[Activity]:
        """Generate generic activities for unknown destinations."""
        return [
            Activity(
                f"{destination} City Tour", "sightseeing", 4.0, 40.0,
                f"Comprehensive tour of {destination}'s highlights",
                "morning", True,
            ),
            Activity(
                f"{destination} Food Tour", "food", 3.0, 60.0,
                "Sample local cuisine and street food", "afternoon", True,
            ),
            Activity(
                f"Historical {destination} Walk", "cultural", 2.5, 25.0,
                "Explore historic sites and monuments", "morning", False,
            ),
            Activity(
                f"{destination} Museum Visit", "cultural", 3.0, 15.0,
                "Visit the main museum and art galleries",
                "afternoon", False,
            ),
            Activity(
                "Local Market Experience", "food", 2.0, 0.0,
                "Browse local markets and taste authentic food",
                "morning", False,
            ),
        ]


class TransportationSearchInput(BaseModel):
    """Input for transportation search."""
    from_location: str = Field(description="Origin city")
    to_location: str = Field(description="Destination city")
    date: str = Field(description="Travel date (YYYY-MM-DD)")
    transport_type: str | None = Field(
        default=None, description="Type: flight, train, bus"
    )


class TransportationSearchTool(BaseTool):
    """Tool for searching transportation options."""
    name: str = "transportation_search"
    description: str = (
        "Search for flight, train, or bus options between two locations on a date."
    )
    args_schema: type[BaseModel] = TransportationSearchInput

    def _run(
        self,
        from_location: str,
        to_location: str,
        date: str,
        transport_type: str | None = None,
    ) -> str:
        options = self._generate_transport_options(
            from_location, to_location, date, transport_type
        )
        return json.dumps([asdict(opt) for opt in options], indent=2)

    def _generate_transport_options(
        self,
        from_loc: str,
        to_loc: str,
        date: str,
        transport_type: str | None,
    ) -> list[Transportation]:
        options: list[Transportation] = []

        if not transport_type or transport_type == "flight":
            flight_times = ["06:30", "10:15", "14:45", "18:30", "21:00"]
            airlines = [
                "Korean Air",
                "Asiana",
                "Japan Airlines",
                "ANA",
                "Singapore Airlines",
                "Emirates",
                "Lufthansa",
            ]
            for _ in range(3):
                duration = random.uniform(1.5, 14.0)
                price = random.uniform(200, 1200)
                options.append(
                    Transportation(
                        type="flight",
                        from_location=from_loc,
                        to_location=to_loc,
                        duration_hours=round(duration, 1),
                        price=round(price, 2),
                        provider=random.choice(airlines),
                        departure_time=random.choice(flight_times),
                    )
                )

        if not transport_type or transport_type == "train":
            train_times = ["07:00", "09:30", "12:00", "15:00", "18:00"]
            train_companies = ["KTX", "Shinkansen", "TGV", "Eurostar", "Amtrak", "ICE"]
            for _ in range(2):
                duration = random.uniform(2.0, 8.0)
                price = random.uniform(50, 300)
                options.append(
                    Transportation(
                        type="train",
                        from_location=from_loc,
                        to_location=to_loc,
                        duration_hours=round(duration, 1),
                        price=round(price, 2),
                        provider=random.choice(train_companies),
                        departure_time=random.choice(train_times),
                    )
                )

        if not transport_type or transport_type == "bus":
            bus_times = ["06:00", "10:00", "14:00", "22:00"]
            options.append(
                Transportation(
                    type="bus",
                    from_location=from_loc,
                    to_location=to_loc,
                    duration_hours=round(random.uniform(4.0, 16.0), 1),
                    price=round(random.uniform(20, 80), 2),
                    provider="Express Bus",
                    departure_time=random.choice(bus_times),
                )
            )

        options.sort(key=lambda x: x.price)
        # Suppress the unused 'date' argument warning while keeping the signature.
        _ = date
        return options


class WeatherInput(BaseModel):
    """Input for weather forecast."""
    destination: str = Field(description="Destination city name")
    date: str = Field(description="Date for weather forecast (YYYY-MM-DD)")


class WeatherForecastTool(BaseTool):
    """Tool for getting weather forecasts."""
    name: str = "weather_forecast"
    description: str = "Get weather forecast for a destination on a specific date."
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, destination: str, date: str) -> str:
        try:
            forecast_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            forecast_date = datetime.now(UTC) + timedelta(days=7)

        month = forecast_date.month

        climate_data = {
            "tokyo": {"winter": (5, 12), "spring": (12, 22), "summer": (25, 32), "autumn": (15, 25)},
            "paris": {"winter": (3, 8), "spring": (10, 18), "summer": (18, 28), "autumn": (10, 18)},
            "seoul": {"winter": (-5, 5), "spring": (10, 20), "summer": (23, 32), "autumn": (10, 22)},
            "bangkok": {"winter": (25, 32), "spring": (28, 35), "summer": (27, 33), "autumn": (26, 32)},
            "new_york": {"winter": (-2, 6), "spring": (10, 20), "summer": (22, 30), "autumn": (12, 22)},
        }

        if month in (12, 1, 2):
            season = "winter"
        elif month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        else:
            season = "autumn"

        dest_key = destination.lower().replace(" ", "_")
        temps = climate_data.get(
            dest_key,
            {"winter": (5, 15), "spring": (15, 25), "summer": (25, 35), "autumn": (15, 25)},
        )
        low, high = temps[season]

        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]

        forecast = {
            "destination": destination,
            "date": date,
            "high_temp_c": high + random.randint(-3, 3),
            "low_temp_c": low + random.randint(-2, 2),
            "condition": random.choice(conditions),
            "humidity": random.randint(40, 80),
            "rain_chance": random.randint(0, 50),
            "recommendation": self._get_recommendation(season),
        }
        return json.dumps(forecast, indent=2)

    def _get_recommendation(self, season: str) -> str:
        recommendations = {
            "winter": (
                "Pack warm layers, a good coat, and waterproof boots. Don't forget "
                "gloves and a scarf."
            ),
            "spring": (
                "Bring layers as temperatures vary. A light jacket and umbrella are "
                "essential."
            ),
            "summer": (
                "Pack light, breathable clothing. Sunscreen and sunglasses are "
                "must-haves."
            ),
            "autumn": (
                "Layer clothing for changing temperatures. A rain jacket is "
                "recommended."
            ),
        }
        return recommendations.get(season, "Pack versatile clothing for variable weather.")


class BudgetCalculatorInput(BaseModel):
    """Input for budget calculation."""
    destination: str = Field(description="Destination city")
    days: int = Field(description="Number of days")
    accommodation_budget: str = Field(
        description="Budget level: budget, moderate, luxury"
    )
    travelers: int = Field(default=1, description="Number of travelers")


class BudgetCalculatorTool(BaseTool):
    """Tool for calculating trip budget."""
    name: str = "budget_calculator"
    description: str = (
        "Calculate an estimated trip budget based on destination, duration, "
        "accommodation tier, and number of travelers."
    )
    args_schema: type[BaseModel] = BudgetCalculatorInput

    def _run(
        self,
        destination: str,
        days: int,
        accommodation_budget: str,
        travelers: int = 1,
    ) -> str:
        dest_key = destination.lower().replace(" ", "_")
        dest = DESTINATIONS_DB.get(dest_key)

        base_daily = dest.avg_daily_cost if dest else 100.0

        acc_multipliers = {"budget": 0.5, "moderate": 1.0, "luxury": 2.5}
        acc_mult = acc_multipliers.get(accommodation_budget, 1.0)

        accommodation = base_daily * 0.4 * acc_mult * days
        food = base_daily * 0.25 * days
        activities = base_daily * 0.2 * days
        transport_local = base_daily * 0.1 * days
        misc = base_daily * 0.05 * days

        subtotal = (
            (accommodation + food + activities + transport_local + misc) * travelers
        )

        flight_estimate = random.uniform(400, 1200) * travelers

        budget = {
            "destination": destination,
            "days": days,
            "travelers": travelers,
            "breakdown": {
                "accommodation": round(accommodation * travelers, 2),
                "food_and_dining": round(food * travelers, 2),
                "activities_and_tours": round(activities * travelers, 2),
                "local_transportation": round(transport_local * travelers, 2),
                "miscellaneous": round(misc * travelers, 2),
                "estimated_flights": round(flight_estimate, 2),
            },
            "subtotal_without_flights": round(subtotal, 2),
            "total_estimate": round(subtotal + flight_estimate, 2),
            "daily_average": round((subtotal / days) / max(1, travelers), 2),
            "currency": "USD",
            "tips": self._get_budget_tips(accommodation_budget),
        }
        return json.dumps(budget, indent=2)

    def _get_budget_tips(self, budget_level: str) -> list[str]:
        tips = {
            "budget": [
                "Stay in hostels or budget hotels",
                "Eat at local restaurants and street food stalls",
                "Use public transportation",
                "Book activities in advance for discounts",
                "Travel during shoulder season",
            ],
            "moderate": [
                "Mix mid-range hotels with boutique options",
                "Balance fine dining with local eateries",
                "Consider day passes for transportation",
                "Book popular attractions ahead",
            ],
            "luxury": [
                "Consider package deals at luxury hotels",
                "Book private tours for personalized experiences",
                "Reserve top restaurants in advance",
                "Consider travel insurance for peace of mind",
            ],
        }
        return tips.get(budget_level, tips["moderate"])


def get_travel_tools() -> list[BaseTool]:
    """Return the full list of travel-planning tools."""
    return [
        DestinationSearchTool(),
        AccommodationSearchTool(),
        ActivitySearchTool(),
        TransportationSearchTool(),
        WeatherForecastTool(),
        BudgetCalculatorTool(),
    ]
