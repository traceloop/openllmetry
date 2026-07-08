import asyncio
import random
import argparse
import time
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from traceloop.sdk import Traceloop

from agents import Agent, function_tool, RunContextWrapper, Runner, ToolCallOutputItem
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseOutputRefusal,
)

load_dotenv()

Traceloop.init(
    app_name="travel-agent-demo",
    disable_batch=False,
)


class CountryInfo(BaseModel):
    name: str
    capital: str
    region: str
    subregion: str
    population: int
    currencies: List[str]
    languages: List[str]
    timezones: List[str]


class DestinationSearchResponse(BaseModel):
    status: str
    message: str
    countries: List[CountryInfo] | None = None
    count: int = 0


class DailyForecast(BaseModel):
    date: str
    temp_max: float
    temp_min: float
    precipitation: float


class WeatherForecast(BaseModel):
    location: str
    latitude: float
    longitude: float
    current_temperature: float
    current_conditions: str
    forecast_days: List[DailyForecast]


class WeatherResponse(BaseModel):
    status: str
    message: str
    forecast: WeatherForecast | None = None


class LocationCoordinates(BaseModel):
    location_name: str
    latitude: float
    longitude: float
    country: str
    display_name: str


class CoordinatesResponse(BaseModel):
    status: str
    message: str
    coordinates: LocationCoordinates | None = None


class DestinationInfo(BaseModel):
    title: str
    summary: str
    extract: str


class DestinationInfoResponse(BaseModel):
    status: str
    message: str
    info: DestinationInfo | None = None


class TravelDistance(BaseModel):
    from_location: str
    to_location: str
    distance_km: float
    flight_time_hours: float


class DistanceResponse(BaseModel):
    status: str
    message: str
    distance_info: TravelDistance | None = None


class DayActivity(BaseModel):
    time: str
    activity: str
    location: str
    notes: str


class DayPlan(BaseModel):
    day_number: int
    date: str
    title: str
    activities: List[DayActivity]
    meals: List[str]
    accommodation: str


class TravelItinerary(BaseModel):
    trip_title: str
    destination: str
    duration_days: int
    total_budget_estimate: str
    daily_plans: List[DayPlan]
    travel_tips: List[str]
    packing_suggestions: List[str]


class ItineraryResponse(BaseModel):
    status: str
    message: str
    itinerary: TravelItinerary | None = None


@dataclass
class TravelContext:
    """Context for the travel agent application."""
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


@function_tool
async def search_destinations(
    cw: RunContextWrapper[TravelContext],
    region: str = "",
    subregion: str = ""
) -> DestinationSearchResponse:
    """
    Search for travel destinations by region or subregion using REST Countries API.

    Args:
        region: Region to search (e.g., "Europe", "Asia", "Americas")
        subregion: Subregion to search (e.g., "Southern Europe", "Southeast Asia")

    Returns:
        List of countries matching the search criteria
    """
    print(f"Searching destinations for region: '{region}', subregion: '{subregion}'")

    try:
        # Add small delay to respect rate limits
        await asyncio.sleep(0.5)

        if region:
            url = f"https://restcountries.com/v3.1/region/{region}"
        else:
            url = "https://restcountries.com/v3.1/all"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        countries_data = response.json()

        # Filter by subregion if provided
        if subregion:
            countries_data = [c for c in countries_data
                              if c.get("subregion", "").lower() == subregion.lower()]

        # Limit to 10 countries to avoid too much data
        countries_data = countries_data[:10]

        countries = []
        for country in countries_data:
            country_info = CountryInfo(
                name=country.get("name", {}).get("common", "Unknown"),
                capital=", ".join(country.get("capital", ["Unknown"])),
                region=country.get("region", "Unknown"),
                subregion=country.get("subregion", "Unknown"),
                population=country.get("population", 0),
                currencies=list(country.get("currencies", {}).keys()),
                languages=list(country.get("languages", {}).values()),
                timezones=country.get("timezones", [])
            )
            countries.append(country_info)

        return DestinationSearchResponse(
            status="success",
            message=f"Found {len(countries)} destinations in {region or 'all regions'}",
            countries=countries,
            count=len(countries)
        )

    except requests.RequestException as e:
        print(f"Error searching destinations: {str(e)}")
        return DestinationSearchResponse(
            status="error",
            message=f"Failed to search destinations: {str(e)}"
        )


@function_tool
async def get_weather_forecast(
    cw: RunContextWrapper[TravelContext],
    location_name: str,
    latitude: float,
    longitude: float
) -> WeatherResponse:
    """
    Get current weather and 7-day forecast using Open-Meteo API (no API key required).

    Args:
        location_name: Name of the location
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Weather forecast information
    """
    print(f"Getting weather forecast for {location_name} ({latitude}, {longitude})")

    try:
        # Add small delay to respect rate limits
        await asyncio.sleep(0.5)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 7
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        # Map weather codes to conditions (simplified)
        weather_code = current.get("weather_code", 0)
        conditions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            61: "Light rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Light snow",
            95: "Thunderstorm"
        }
        current_conditions = conditions.get(weather_code, "Unknown")

        forecast_days = []
        times = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        for i in range(min(len(times), len(temp_max), len(temp_min), len(precip))):
            daily_forecast = DailyForecast(
                date=times[i] if i < len(times) else "",
                temp_max=float(temp_max[i]) if i < len(temp_max) and temp_max[i] is not None else 0.0,
                temp_min=float(temp_min[i]) if i < len(temp_min) and temp_min[i] is not None else 0.0,
                precipitation=float(precip[i]) if i < len(precip) and precip[i] is not None else 0.0
            )
            forecast_days.append(daily_forecast)

        forecast = WeatherForecast(
            location=location_name,
            latitude=float(latitude),
            longitude=float(longitude),
            current_temperature=float(
                current.get("temperature_2m", 0.0)
            ) if current.get("temperature_2m") is not None else 0.0,
            current_conditions=current_conditions,
            forecast_days=forecast_days
        )

        return WeatherResponse(
            status="success",
            message=f"Weather forecast retrieved for {location_name}",
            forecast=forecast
        )

    except requests.RequestException as e:
        print(f"Error getting weather forecast (RequestException): {str(e)}")
        return WeatherResponse(
            status="error",
            message=f"Failed to get weather forecast: {str(e)}"
        )
    except Exception as e:
        print(f"Error getting weather forecast (Exception): {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return WeatherResponse(
            status="error",
            message=f"Failed to get weather forecast: {type(e).__name__}: {str(e)}"
        )


@function_tool
async def get_location_coordinates(
    cw: RunContextWrapper[TravelContext],
    location_name: str
) -> CoordinatesResponse:
    """
    Get coordinates for a location using Nominatim (OpenStreetMap) API.

    Args:
        location_name: Name of the city or location

    Returns:
        Latitude and longitude coordinates
    """
    print(f"Getting coordinates for location: {location_name}")

    try:
        # Add small delay to respect rate limits (Nominatim requires max 1 req/sec)
        await asyncio.sleep(1.1)

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location_name,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "TravelAgentDemo/1.0 (OpenTelemetry Sample App)"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data:
            return CoordinatesResponse(
                status="error",
                message=f"Location not found: {location_name}"
            )

        location_data = data[0]
        coordinates = LocationCoordinates(
            location_name=location_name,
            latitude=float(location_data.get("lat", 0.0)),
            longitude=float(location_data.get("lon", 0.0)),
            country=location_data.get("display_name", "").split(",")[-1].strip(),
            display_name=location_data.get("display_name", "")
        )

        return CoordinatesResponse(
            status="success",
            message=f"Coordinates found for {location_name}",
            coordinates=coordinates
        )

    except requests.RequestException as e:
        print(f"Error getting coordinates: {str(e)}")
        return CoordinatesResponse(
            status="error",
            message=f"Failed to get coordinates: {str(e)}"
        )


@function_tool
async def get_destination_info(
    cw: RunContextWrapper[TravelContext],
    destination_name: str
) -> DestinationInfoResponse:
    """
    Get information about a destination from Wikipedia API.

    Args:
        destination_name: Name of the destination

    Returns:
        Summary and information about the destination
    """
    print(f"Getting destination info for: {destination_name}")

    try:
        # Add small delay to respect rate limits
        await asyncio.sleep(0.5)

        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + destination_name
        headers = {
            "User-Agent": "TravelAgentDemo/1.0 (OpenTelemetry Sample App)"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        info = DestinationInfo(
            title=data.get("title", destination_name),
            summary=data.get("description", "No description available"),
            extract=data.get("extract", "No information available")
        )

        return DestinationInfoResponse(
            status="success",
            message=f"Retrieved information for {destination_name}",
            info=info
        )

    except requests.RequestException as e:
        print(f"Error getting destination info: {str(e)}")
        return DestinationInfoResponse(
            status="error",
            message=f"Failed to get destination info: {str(e)}"
        )


@function_tool
async def calculate_travel_distance(
    cw: RunContextWrapper[TravelContext],
    from_location: str,
    to_location: str,
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float
) -> DistanceResponse:
    """
    Calculate distance and estimated flight time between two locations.
    Uses Haversine formula for distance calculation.

    Args:
        from_location: Starting location name
        to_location: Destination location name
        from_lat: Starting latitude
        from_lon: Starting longitude
        to_lat: Destination latitude
        to_lon: Destination longitude

    Returns:
        Distance and flight time information
    """
    print(f"Calculating distance from {from_location} to {to_location}")

    try:
        import math

        # Haversine formula
        R = 6371  # Earth's radius in kilometers

        lat1, lon1 = math.radians(from_lat), math.radians(from_lon)
        lat2, lon2 = math.radians(to_lat), math.radians(to_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = R * c

        # Estimate flight time (average speed ~800 km/h including takeoff/landing)
        flight_time_hours = distance_km / 800.0

        distance_info = TravelDistance(
            from_location=from_location,
            to_location=to_location,
            distance_km=round(distance_km, 2),
            flight_time_hours=round(flight_time_hours, 2)
        )

        return DistanceResponse(
            status="success",
            message=f"Distance calculated: {distance_km:.2f} km, ~{flight_time_hours:.2f} hours flight",
            distance_info=distance_info
        )

    except Exception as e:
        print(f"Error calculating distance: {str(e)}")
        return DistanceResponse(
            status="error",
            message=f"Failed to calculate distance: {str(e)}"
        )


@function_tool
async def create_itinerary(
    cw: RunContextWrapper[TravelContext],
    destination: str,
    duration_days: int,
    budget: str,
    interests: str,
    weather_info: str = "",
    destination_details: str = ""
) -> ItineraryResponse:
    """
    Create a detailed day-by-day travel itinerary using AI.

    Args:
        destination: Main destination for the trip
        duration_days: Number of days for the trip
        budget: Budget level (budget, moderate, luxury)
        interests: Traveler interests (e.g., food, history, nature)
        weather_info: Weather forecast information (optional)
        destination_details: Additional destination details (optional)

    Returns:
        Detailed travel itinerary
    """
    print(f"Creating {duration_days}-day itinerary for {destination} ({budget} budget)")

    try:
        import openai

        # Add small delay
        await asyncio.sleep(0.5)

        client = openai.OpenAI()

        itinerary_prompt = f"""
You are an expert travel planner. Create a detailed {duration_days}-day itinerary for {destination}.

Trip Details:
- Destination: {destination}
- Duration: {duration_days} days
- Budget: {budget}
- Interests: {interests}
{f"- Weather: {weather_info}" if weather_info else ""}
{f"- Destination Info: {destination_details}" if destination_details else ""}

Create a comprehensive itinerary that includes:
1. A catchy trip title
2. Day-by-day plans with specific activities and timings
3. Meal recommendations for each day
4. Accommodation suggestions
5. Travel tips specific to this destination
6. Packing suggestions based on the weather and activities

Make the itinerary practical, engaging, and tailored to the {budget} budget level and {interests} interests.
Each day should have 3-5 activities with specific times, locations, and helpful notes.
"""

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert travel planner who creates detailed, practical itineraries."
                },
                {"role": "user", "content": itinerary_prompt}
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format=TravelItinerary
        )

        itinerary = response.choices[0].message.parsed

        return ItineraryResponse(
            status="success",
            message=f"Created {duration_days}-day itinerary for {destination}",
            itinerary=itinerary
        )

    except Exception as e:
        print(f"Error creating itinerary: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ItineraryResponse(
            status="error",
            message=f"Failed to create itinerary: {str(e)}"
        )


class TravelPlannerAgent(Agent[TravelContext]):
    """Specialized agent for travel planning with 6 tools, always creating itineraries."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="Travel Planner Agent",
            instructions="""
            You are an expert travel planning assistant. Your PRIMARY GOAL is to ALWAYS create a detailed
            travel itinerary for the user, no matter how broad or specific their request is.

            Your workflow:
            1. Gather information based on the user's request (use research tools as needed)
            2. Make reasonable assumptions for missing details (budget, duration, interests)
            3. ALWAYS end by creating a complete itinerary using the create_itinerary tool

            Your 6 tools:
            1. search_destinations - Find destinations by region
            2. get_location_coordinates - Get lat/long for locations
            3. get_weather_forecast - Check weather forecasts
            4. get_destination_info - Get details about places
            5. calculate_travel_distance - Calculate distances between locations
            6. create_itinerary - CREATE THE FINAL ITINERARY (REQUIRED!)

            Response patterns based on request specificity:

            SPECIFIC REQUESTS (destination, duration, budget mentioned):
            - Gather targeted information (weather, details)
            - Immediately create itinerary

            BROAD REQUESTS (just region or vague preferences):
            - Search for destinations in the region
            - Pick 1-2 promising destinations
            - Get coordinates and weather
            - Make reasonable assumptions for duration (default 5-7 days) and budget (default moderate)
            - Create itinerary with your recommendations

            VERY VAGUE REQUESTS (no clear destination):
            - Search popular destinations
            - Recommend based on weather/season
            - Assume moderate budget, 5-7 days
            - Create itinerary

            CRITICAL: Every response must end with a complete itinerary. Never skip the create_itinerary step.
            If information is missing, make sensible assumptions and explain them in the itinerary.

            When creating itineraries, use information from your research tools to make them relevant and practical.
            """,
            model=model,
            tools=[
                search_destinations,
                get_location_coordinates,
                get_weather_forecast,
                get_destination_info,
                calculate_travel_distance,
                create_itinerary
            ],
        )


async def handle_runner_stream(runner: "Runner"):
    """Process runner events and display output."""

    tool_calls_made = []
    response_text_parts = []

    async for event in runner.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                response_text_parts.append(event.data.delta)
            elif isinstance(event.data, ResponseOutputItemAddedEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    tool_name = event.data.item.name
                    tool_calls_made.append(tool_name)
                    print(f"\n[Calling tool: {tool_name}]\n")
        elif event.type == "run_item_stream_event":
            if event.name == "tool_output" and isinstance(
                event.item, ToolCallOutputItem
            ):
                raw_item = event.item.raw_item
                content = (
                    raw_item.get("content")
                    if isinstance(raw_item, dict)
                    else getattr(raw_item, "content", "")
                )
                if content:
                    print(f"\n[Tool output: {content[:200]}...]\n", end="", flush=True)

            elif event.name == "message_output_created":
                raw_item = event.item.raw_item
                role = getattr(raw_item, "role", None)
                if role is None and isinstance(raw_item, dict):
                    role = raw_item.get("role")

                if role == "assistant":
                    content_parts = []
                    for part in getattr(raw_item, "content", []):
                        if isinstance(part, ResponseOutputText):
                            content_parts.append(part.text)
                            response_text_parts.append(part.text)
                        elif isinstance(part, ResponseOutputRefusal):
                            content_parts.append(part.refusal)
                            response_text_parts.append(part.refusal)
                    if content_parts:
                        print("".join(content_parts), end="", flush=True)

    print()
    return tool_calls_made, "".join(response_text_parts)


async def run_travel_query(query: str, return_response_text: bool = False):
    """
    Run a single travel planning query.

    Args:
        query: The travel planning query
        return_response_text: If True, returns the response text.
            If False, returns tool_calls (for backward compatibility)

    Returns:
        Either response_text (str) or tool_calls (list) depending on parameter
    """

    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    travel_agent = TravelPlannerAgent()

    print("\nAgent Response: ", end="", flush=True)

    messages = [{"role": "user", "content": query}]
    runner = Runner().run_streamed(starting_agent=travel_agent, input=messages)
    tool_calls, response_text = await handle_runner_stream(runner)

    print(f"\n{'='*80}")
    print(f"✅ Query completed! Tools used: {', '.join(tool_calls) if tool_calls else 'None'}")
    print(f"{'='*80}\n")

    if return_response_text:
        return response_text
    else:
        return tool_calls  # Backward compatibility for existing code


def generate_travel_queries(n: int = 10) -> List[str]:
    """Generate diverse travel planning queries with varying specificity, all leading to itinerary creation."""

    # Mix of SPECIFIC, BROAD, and VAGUE templates that all result in itinerary creation
    templates = [
        # === VERY SPECIFIC REQUESTS (most details provided) ===
        # Agent should: get weather + destination info + create itinerary (3-4 tools)
        ("Plan a {duration}-day {budget} trip to {city} for {travelers} interested in "
         "{interest}. Create a complete itinerary."),

        ("I want to visit {city} for {duration} days with a {budget} budget. I love "
         "{interest}. Create me an itinerary."),

        ("Create a {duration}-day itinerary for {city}. Budget: {budget}, interests: "
         "{interest} and {interest2}."),

        # === MODERATELY SPECIFIC REQUESTS (some details, need destination selection) ===
        # Agent should: search destinations + coordinates + weather + info + create itinerary (5-6 tools)
        ("I want a {duration}-day {budget} {season} vacation in {region}. I'm "
         "interested in {interest}. Plan my trip."),

        ("Plan a {adjective} trip to {region} for {travelers}. Budget is {budget}, "
         "duration {duration} days. Interested in {interest}."),

        ("I need a {duration}-day itinerary for {region} focusing on {interest} and "
         "{interest2}. Budget: {budget}."),

        # === BROAD REQUESTS (region only, need destination research) ===
        # Agent should: search destinations + coordinates + weather + info + create itinerary (5-7 tools)
        ("I want to explore {region} in {season}. Find good destinations and create an "
         "itinerary for me."),

        ("Plan a {budget} trip to {region}. I love {interest}. Find the best place and "
         "create an itinerary."),

        ("Help me plan a vacation in {region} for {travelers}. I'm interested in "
         "{interest}."),

        ("I want to visit {region}. Create a travel plan for me focusing on {interest} "
         "and {interest2}."),

        # === COMPARISON REQUESTS (compare then decide and create itinerary) ===
        # Agent should: coordinates + weather + info for both + create itinerary for winner (6-8 tools)
        ("Should I visit {city1} or {city2} for a {duration}-day {season} trip? "
         "Compare them and create an itinerary for the better option."),

        ("I'm deciding between {city1} and {city2}. Check weather, compare them, and "
         "create a {duration}-day itinerary for your recommendation."),

        # === VAGUE/OPEN-ENDED REQUESTS (minimal details) ===
        # Agent should: search region + pick destination + weather + info + create itinerary (5-7 tools)
        "I need a vacation. I like {interest}. Plan something for me.",

        "Plan a {season} getaway for {travelers}. Surprise me with a good destination.",

        "I want to go somewhere {adjective} for {interest}. Create a trip for me.",

        "Find me a great {budget} destination and plan my trip.",

        # === RESEARCH-HEAVY REQUESTS (lots of comparison before itinerary) ===
        # Agent should: search + coordinates (multiple) + weather (multiple) + info + create itinerary (7-10 tools)
        ("Find the best {season} destinations in {region}. Check weather for top 3, "
         "then create an itinerary for the best one."),

        ("I want a {budget} {interest} trip. Search {region}, compare weather in "
         "several places, and create an itinerary for the top pick."),

        ("Show me good {adjective} destinations in {region}. Compare a few, then plan "
         "a {duration}-day trip to your favorite."),

        # === MULTI-CITY REQUESTS ===
        # Agent should: coordinates (multiple) + distances + weather (multiple)
        # + create multi-city itinerary (7-9 tools)
        ("Plan a {duration}-day multi-city trip visiting {city1}, {city2}, and "
         "{city3}. Create a complete itinerary."),

        ("I want to visit multiple cities in {region} over {duration} days. Find the "
         "best route and create an itinerary."),
    ]

    regions = ["Europe", "Asia", "Americas", "Africa", "Oceania"]
    budgets = ["budget", "moderate", "luxury"]
    durations = ["3", "5", "7", "10", "14"]
    travelers = ["solo travelers", "couples", "families", "groups"]
    seasons = ["spring", "summer", "fall", "winter"]
    interests = ["food", "history", "nature", "beaches", "museums", "adventure", "culture", "nightlife"]
    adjectives = ["quick", "relaxing", "adventurous", "cultural", "romantic", "family-friendly", "exciting", "peaceful"]
    cities = [
        "Paris", "Tokyo", "New York", "London", "Barcelona", "Rome", "Bangkok",
        "Dubai", "Singapore", "Amsterdam", "Berlin", "Sydney", "Istanbul", "Prague",
        "Vienna", "Lisbon", "Cairo", "Mumbai", "Toronto", "Buenos Aires"
    ]

    queries = []
    for _ in range(n):
        template = random.choice(templates)

        # Pick two different interests
        interest1 = random.choice(interests)
        interest2 = random.choice([i for i in interests if i != interest1])

        # Pick three different cities
        city_choices = random.sample(cities, 3)

        query = template.format(
            region=random.choice(regions),
            budget=random.choice(budgets),
            duration=random.choice(durations),
            travelers=random.choice(travelers),
            season=random.choice(seasons),
            interest=interest1,
            interest2=interest2,
            adjective=random.choice(adjectives),
            city=random.choice(cities),
            city1=city_choices[0],
            city2=city_choices[1],
            city3=city_choices[2]
        )
        queries.append(query)

    return queries


async def main():
    """Main entry point for the travel agent application."""

    parser = argparse.ArgumentParser(description="Travel Planning Agent Demo")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of queries to run (default: 3)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between queries in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Travel Planning Agent with OpenAI Agents SDK")
    print("=" * 80)
    print(f"Running {args.count} travel planning queries...")
    print("Goal: Create complete itineraries with varying research depth")
    print("Using real APIs: REST Countries, Open-Meteo, Nominatim, Wikipedia, OpenAI")
    print("6 Tools: search, coordinates, weather, info, distance, itinerary")
    print("=" * 80)
    print()

    queries = generate_travel_queries(args.count)

    all_tool_calls = []
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Query {i} of {args.count}")
        print(f"{'#'*80}\n")

        tool_calls = await run_travel_query(query)
        all_tool_calls.append({
            "query": query,
            "tools_used": tool_calls,
            "tool_count": len(tool_calls)
        })

        if i < args.count:
            print(f"\nWaiting {args.delay} seconds before next query...")
            time.sleep(args.delay)

    # Summary
    print("\n\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total queries executed: {len(all_tool_calls)}")

    tool_usage = {}
    for result in all_tool_calls:
        for tool in result["tools_used"]:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    print("\nTool usage statistics:")
    for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tool}: {count} times")

    print("\nTrajectory variation:")
    unique_trajectories = len(set(tuple(r["tools_used"]) for r in all_tool_calls))
    print(f"  - Unique tool call sequences: {unique_trajectories}/{len(all_tool_calls)}")

    avg_tools = sum(r["tool_count"] for r in all_tool_calls) / len(all_tool_calls)
    print(f"  - Average tools per query: {avg_tools:.2f}")

    print("\n" + "=" * 80)
    print("✅ Travel Agent demo completed successfully!")
    print("🔍 All spans captured by OpenTelemetry instrumentation")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
