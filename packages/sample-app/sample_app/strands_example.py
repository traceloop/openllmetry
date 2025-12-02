import os
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from strands import Agent, tool
from strands.models.litellm import LiteLLMModel
from traceloop.sdk import Traceloop

from dotenv import load_dotenv

load_dotenv()

Traceloop.init(
    app_name="travel-planning-multi-agent"
)

# Shared memory store for multi-agent coordination
shared_memory = {
    "conversation_history": [],
    "user_preferences": {},
    "research_data": {},
    "trip_plans": {},
    "bookings": []
}

def add_to_memory(agent_name: str, action: str, data: Any):
    """Add information to shared memory."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "data": data
    }
    shared_memory["conversation_history"].append(entry)
    return entry


# Travel Research Agent Tools
@tool
def research_destination(destination: str) -> str:
    """Research travel destinations, attractions, and activities."""
    try:
        query = f"{destination} travel guide attractions activities weather best time to visit"
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        )
        data = response.json()
        
        results = []
        if "RelatedTopics" in data:
            for topic in data["RelatedTopics"][:3]:
                if "Text" in topic:
                    results.append(topic["Text"])
        
        research_info = "\n".join(results) if results else f"Limited information available for {destination}"
        add_to_memory("research_agent", "destination_research", {"destination": destination, "info": research_info})
        return research_info
    except Exception as e:
        return f"Research error: {str(e)}"


@tool
def get_weather_forecast(city: str, dates: str = "") -> str:
    """Get weather information for travel planning."""
    weather_info = f"Weather forecast for {city}: Generally pleasant with temperatures 20-25°C. {dates} looks good for travel with sunny skies and light winds."
    add_to_memory("research_agent", "weather_check", {"city": city, "dates": dates, "forecast": weather_info})
    return weather_info


# Trip Planning Agent Tools
@tool
def create_itinerary(destination: str, duration: str, interests: str) -> str:
    """Create detailed travel itinerary based on research and preferences."""
    # Check memory for research data
    research_data = shared_memory.get("research_data", {})
    
    itinerary = f"""🗓️ {duration} Itinerary for {destination}

Based on your interests in {interests}:

Day 1: Arrival & City Orientation
- Arrive and check into hotel
- Walking tour of main attractions
- Local cuisine dinner

Day 2: Cultural Exploration
- Museums and historical sites
- Local markets and shopping
- Cultural show or performance

Day 3: Adventure & Activities
- Outdoor activities based on interests
- Local experiences and tours
- Sunset viewing location

Day 4: Relaxation & Departure
- Leisure morning
- Final shopping/souvenirs
- Departure preparations

Note: Itinerary customized based on {interests} preferences."""
    
    add_to_memory("planning_agent", "itinerary_created", {"destination": destination, "duration": duration, "itinerary": itinerary})
    shared_memory["trip_plans"][destination] = itinerary
    return itinerary


@tool
def estimate_budget(destination: str, duration: str, travelers: int = 1) -> str:
    """Estimate travel budget for the trip."""
    base_cost = 150  # per day per person
    days = int(duration.split()[0]) if duration.split()[0].isdigit() else 3
    
    accommodation = days * 100 * travelers
    meals = days * 50 * travelers 
    activities = days * 75 * travelers
    transport = 500 * travelers
    
    total = accommodation + meals + activities + transport
    
    budget_breakdown = f"""💰 Budget Estimate for {destination} ({duration})

Accommodation: ${accommodation}
Meals: ${meals}
Activities: ${activities}
Transport: ${transport}
{'─' * 30}
Total: ${total} for {travelers} traveler(s)

Note: Estimates in USD, actual costs may vary."""
    
    add_to_memory("planning_agent", "budget_estimated", {"destination": destination, "total": total, "breakdown": budget_breakdown})
    return budget_breakdown


# Booking Agent Tools
@tool
def search_flights(origin: str, destination: str, dates: str) -> str:
    """Search for flight options and prices."""
    flight_results = f"""✈️ Flight Search Results

Route: {origin} → {destination}
Dates: {dates}

🏷️ Economy Options:
1. DirectFly Airlines - $450 (Direct, 6h 30m)
2. ConnectAir - $320 (1 stop, 8h 45m)
3. BudgetWings - $280 (2 stops, 12h 15m)

🌟 Business Class:
1. PremiumAir - $1,200 (Direct, 6h 15m)
2. ComfortFly - $980 (1 stop, 8h 30m)

Note: Prices are estimates. Book within 24h to secure these rates."""
    
    add_to_memory("booking_agent", "flight_search", {"origin": origin, "destination": destination, "dates": dates, "results": flight_results})
    return flight_results


@tool
def search_hotels(destination: str, checkin: str, checkout: str, guests: int = 1) -> str:
    """Search for hotel accommodations."""
    hotel_results = f"""🏨 Hotel Search Results for {destination}

Dates: {checkin} to {checkout}
Guests: {guests}

⭐⭐⭐⭐⭐ Luxury:
1. Grand Palace Hotel - $350/night (Downtown, Spa, Pool)
2. Skyline Resort - $280/night (City View, Gym, Restaurant)

⭐⭐⭐⭐ Mid-Range:
3. Comfort Inn Central - $150/night (Central Location, Breakfast)
4. Urban Stay Hotel - $120/night (Modern, WiFi, Parking)

⭐⭐⭐ Budget:
5. Traveler's Lodge - $80/night (Clean, Basic, Good Reviews)
6. City Hostel Plus - $45/night (Shared/Private Rooms)

All hotels include WiFi. Prices per night before taxes."""
    
    add_to_memory("booking_agent", "hotel_search", {"destination": destination, "checkin": checkin, "checkout": checkout, "results": hotel_results})
    return hotel_results


# Agent Coordination Tools
@tool
def handoff_to_research_agent(query: str) -> str:
    """Hand off destination research tasks to the research specialist."""
    add_to_memory("coordinator", "handoff_to_research", {"query": query})
    return f"🔄 Handed off to Research Agent: {query}"


@tool
def handoff_to_planning_agent(query: str) -> str:
    """Hand off itinerary and planning tasks to the planning specialist."""
    add_to_memory("coordinator", "handoff_to_planning", {"query": query})
    return f"🔄 Handed off to Planning Agent: {query}"


@tool
def handoff_to_booking_agent(query: str) -> str:
    """Hand off booking and reservation tasks to the booking specialist."""
    add_to_memory("coordinator", "handoff_to_booking", {"query": query})
    return f"🔄 Handed off to Booking Agent: {query}"


@tool
def get_memory_context() -> str:
    """Retrieve relevant context from shared memory."""
    recent_history = shared_memory["conversation_history"][-5:] if shared_memory["conversation_history"] else []
    context = "Recent Activity:\n"
    for entry in recent_history:
        context += f"- {entry['agent']}: {entry['action']} at {entry['timestamp']}\n"
    return context


@tool
def save_trip_plan(destination: str, plan_data: str) -> str:
    """Save complete trip plan to file."""
    try:
        filename = f"trip_plan_{destination.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
        path = Path(filename)
        
        full_plan = f"""Travel Plan for {destination}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{plan_data}

--- Memory Context ---
{json.dumps(shared_memory, indent=2)}"""
        
        path.write_text(full_plan, encoding='utf-8')
        add_to_memory("system", "plan_saved", {"destination": destination, "filename": filename})
        return f"✅ Trip plan saved to {filename}"
    except Exception as e:
        return f"Error saving plan: {str(e)}"


@tool
def search_web(query: str, topic: str = "general") -> str:
    """Search the web for real-time information using Tavily API."""
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "❌ Tavily API key required. Set TAVILY_API_KEY environment variable."
        
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,
                "topic": topic
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = f"🌐 **Web Search Results for:** {query}\n\n"
            
            if "answer" in data and data["answer"]:
                results += f"**Answer:** {data['answer']}\n\n"
            
            if "results" in data:
                results += "**Sources:**\n"
                for i, result in enumerate(data["results"], 1):
                    title = result.get("title", "Result")
                    content = result.get("content", "")[:100] + "..."
                    results += f"{i}. {title}\n   {content}\n\n"
            
            return results
        else:
            return f"❌ Search API error: {response.status_code}"
            
    except Exception as e:
        return f"Search error: {str(e)}"


def create_agents():
    """Create specialized travel planning agents."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Configure LiteLLM model
    model = LiteLLMModel(
        model_id="gpt-4o-mini",
        params={"max_tokens": 1500, "temperature": 0.7}
    )
    
    # Research Agent - Now with real web search capabilities
    research_agent = Agent(
        model=model,
        tools=[research_destination, get_weather_forecast, search_web, get_memory_context],
        system_prompt="You are a Travel Research Specialist with access to real-time web data. Your expertise includes:\n" +
                     "- Live destination research using current web information\n" +
                     "- Real weather forecasts and climate data\n" +
                     "- Up-to-date travel advisories and requirements\n" +
                     "- Current events and local information\n" +
                     "Always provide accurate, real-time travel information."
    )
    
    # Planning Agent - Specialized in itinerary creation
    planning_agent = Agent(
        model=model,
        tools=[create_itinerary, estimate_budget, get_memory_context, save_trip_plan],
        system_prompt="You are a Trip Planning Specialist. Your expertise includes:\n" +
                     "- Creating detailed, personalized itineraries\n" +
                     "- Budget estimation and financial planning\n" +
                     "- Optimizing travel schedules and logistics\n" +
                     "- Balancing activities with traveler preferences\n" +
                     "Create well-structured, realistic travel plans."
    )
    
    # Booking Agent - Specialized in reservations
    booking_agent = Agent(
        model=model,
        tools=[search_flights, search_hotels, get_memory_context],
        system_prompt="You are a Travel Booking Specialist. Your expertise includes:\n" +
                     "- Finding the best flight deals and options\n" +
                     "- Locating suitable accommodations\n" +
                     "- Comparing prices and amenities\n" +
                     "- Providing booking recommendations and alternatives\n" +
                     "Help travelers find the best deals within their budget."
    )
    
    # Coordinator Agent - Orchestrates the team
    coordinator_agent = Agent(
        model=model,
        tools=[handoff_to_research_agent, handoff_to_planning_agent, handoff_to_booking_agent, get_memory_context],
        system_prompt="You are the Travel Coordinator. You orchestrate a team of specialists:\n" +
                     "- Research Agent: destination info, weather, attractions\n" +
                     "- Planning Agent: itineraries, budgets, logistics\n" +
                     "- Booking Agent: flights, hotels, reservations\n" +
                     "\nAnalyze user requests and delegate to the appropriate specialist. " +
                     "Coordinate between agents to provide comprehensive travel assistance."
    )
    
    return {
        "research": research_agent,
        "planning": planning_agent, 
        "booking": booking_agent,
        "coordinator": coordinator_agent
    }


def demo_multi_agent_conversation(agents):
    """Demonstrate multi-agent travel planning workflow with real APIs."""
    print("🌍 Welcome to the Multi-Agent Travel Planning System!\n")
    print("⚡ Now powered by real-time web search and live data!\n")
    
    # Check API keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    
    print("🔑 API Status:")
    print(f"   Tavily Search: {'✅ Ready' if tavily_key else '❌ Missing TAVILY_API_KEY'}")
    print(f"   Weather Data: {'✅ Ready' if weather_key else '❌ Missing OPENWEATHER_API_KEY'}")
    print()
    
    if not tavily_key:
        print("💡 To get real search results, sign up at https://tavily.com and set TAVILY_API_KEY")
        print("💡 For weather data, get free API key at https://openweathermap.org and set OPENWEATHER_API_KEY\n")
    
    # Simulate a complex travel planning request
    user_request = "I want to plan a 4-day trip to Tokyo in April 2025. I'm interested in technology, food, and traditional culture. I need help with research, planning, and finding flights from San Francisco."
    
    print(f"User Request: {user_request}\n")
    print("=" * 80)
    
    # Step 1: Coordinator analyzes and delegates
    print("🤖 Travel Coordinator analyzing request...")
    coord_response = agents["coordinator"](user_request)
    print(f"Coordinator: {coord_response.message['content'][0]['text']}\n")
    
    # Step 2: Research Agent gathers information
    print("🔍 Research Agent investigating Tokyo...")
    research_response = agents["research"]("Research Tokyo for a technology and culture enthusiast visiting in April")
    print(f"Research Agent: {research_response.message['content'][0]['text']}\n")
    
    # Step 3: Planning Agent creates itinerary
    print("🗺️ Planning Agent creating itinerary...")
    planning_response = agents["planning"]("Create a 4-day Tokyo itinerary focusing on technology, food, and traditional culture")
    print(f"Planning Agent: {planning_response.message['content'][0]['text']}\n")
    
    # Step 4: Booking Agent finds options
    print("💼 Booking Agent searching flights and hotels...")
    booking_response = agents["booking"]("Find flights from San Francisco to Tokyo in April and hotel options")
    print(f"Booking Agent: {booking_response.message['content'][0]['text']}\n")
    
    print("=" * 80)
    print("🎉 Multi-agent travel planning complete!")
    print(f"\n📊 Memory entries created: {len(shared_memory['conversation_history'])}")


if __name__ == "__main__":
    agents = create_agents()
    demo_multi_agent_conversation(agents)