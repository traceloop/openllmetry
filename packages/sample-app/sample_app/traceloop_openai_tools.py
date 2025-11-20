import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import agent, tool, workflow
import openai

from dotenv import load_dotenv

load_dotenv()

Traceloop.init(
    app_name="traceloop-openai-tools"
)

# Memory for agent coordination
agent_memory = {
    "research_data": {},
    "trip_plans": {},
    "bookings": {},
    "conversation_log": []
}

def log_action(agent_name: str, action: str, data: Any = None):
    """Log agent actions."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "data": data
    }
    agent_memory["conversation_log"].append(entry)
    return entry

# Define tools for OpenAI function calling
@tool(name="tavily_search")
def search_tavily(query: str) -> str:
    """Search the web using Tavily API for real-time travel information."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "❌ Tavily API key required"
        
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": 3,
                "include_answer": True
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            result = f"🌐 **Search:** {query}\n\n"
            
            if "answer" in data and data["answer"]:
                result += f"**Summary:** {data['answer']}\n\n"
            
            if "results" in data:
                for i, res in enumerate(data["results"], 1):
                    title = res.get("title", "")[:50] + "..."
                    content = res.get("content", "")[:150] + "..."
                    url = res.get("url", "")
                    result += f"{i}. **{title}**\n   {content}\n   🔗 {url}\n\n"
            
            log_action("tavily_search", "search_completed", {"query": query})
            return result
        else:
            return f"❌ Search failed: {response.status_code}"
            
    except Exception as e:
        return f"Search error: {str(e)}"

@tool(name="weather_api")
def get_weather(city: str) -> str:
    """Get current weather data."""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return f"🌤️ Weather info for {city}: Generally pleasant in April, 20-25°C (simulated data - no API key)"
        
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            weather = f"🌤️ **Current Weather in {city}:**\n"
            weather += f"Temp: {data['main']['temp']}°C, {data['weather'][0]['description']}"
            log_action("weather_api", "weather_fetched", {"city": city, "temp": data['main']['temp']})
            return weather
        else:
            return f"🌤️ Weather for {city}: April typically 20-25°C, pleasant conditions (API unavailable)"
            
    except Exception as e:
        return f"🌤️ Weather for {city}: April typically 20-25°C (error: {str(e)[:50]})"

@tool(name="save_file")
def save_travel_plan(filename: str, content: str) -> str:
    """Save travel plan to a file."""
    try:
        path = Path(filename)
        path.write_text(content, encoding='utf-8')
        log_action("file_save", "plan_saved", {"filename": filename})
        return f"✅ Saved to {filename}"
    except Exception as e:
        return f"❌ Save error: {str(e)}"

# OpenAI tool definitions for function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_tavily",
            "description": "Search the web for real-time travel information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for travel information"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather data for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to get weather for"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_travel_plan",
            "description": "Save travel plan to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to save the plan"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to save"
                    }
                },
                "required": ["filename", "content"]
            }
        }
    }
]

# Function registry for tool calling
FUNCTION_REGISTRY = {
    "search_tavily": search_tavily,
    "get_weather": get_weather,
    "save_travel_plan": save_travel_plan
}

def execute_function_call(function_call) -> str:
    """Execute OpenAI function calls."""
    function_name = function_call.function.name
    arguments = json.loads(function_call.function.arguments)
    
    if function_name in FUNCTION_REGISTRY:
        return FUNCTION_REGISTRY[function_name](**arguments)
    else:
        return f"❌ Unknown function: {function_name}"

@agent(name="travel_agent_with_tools")
def travel_planning_agent(user_request: str) -> str:
    """AI travel agent with access to real tools via OpenAI function calling."""
    log_action("travel_agent", "started", {"request": user_request})
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "❌ OpenAI API key required"
    
    client = openai.OpenAI(api_key=api_key)
    
    system_prompt = """You are an expert travel planning agent with access to real-time tools:
    
    AVAILABLE TOOLS:
    - search_tavily: Search the web for current travel information
    - get_weather: Get real weather data for any city
    - save_travel_plan: Save complete travel plans to files
    
    PROCESS:
    1. Use search_tavily to research destinations, attractions, and travel info
    2. Use get_weather to get current conditions
    3. Create comprehensive travel plans
    4. Save the final plan using save_travel_plan
    
    Be thorough, use multiple searches, and provide detailed, actionable travel advice."""
    
    try:
        print(f"🔧 **Initializing OpenAI with {len(TOOL_DEFINITIONS)} tool definitions**")
        
        # Initial conversation with tools
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request}
            ],
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=1500
        )
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        # Process tool calls
        while response.choices[0].message.tool_calls:
            message = response.choices[0].message
            conversation.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                print(f"🔧 Calling tool: {tool_call.function.name}")
                tool_result = execute_function_call(tool_call)
                print(f"📊 Result: {tool_result[:100]}...\n")
                
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Continue conversation with tool results
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=1500
            )
        
        # Final response
        final_response = response.choices[0].message.content
        log_action("travel_agent", "completed", {"tools_used": len([msg for msg in conversation if msg.get("role") == "tool"])})
        
        return final_response
        
    except Exception as e:
        return f"❌ Agent error: {str(e)}"

@workflow(name="travel_planning_workflow")
def plan_travel(user_request: str) -> str:
    """Main workflow using OpenAI tool calling."""
    print("🚀 Starting Travel Planning with OpenAI Tool Calling\n")
    
    # Check API status
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    tavily_key = bool(os.getenv("TAVILY_API_KEY"))
    weather_key = bool(os.getenv("OPENWEATHER_API_KEY"))
    
    print("🔑 API Status:")
    print(f"   OpenAI: {'✅ Ready' if openai_key else '❌ Missing'}")
    print(f"   Tavily: {'✅ Ready' if tavily_key else '❌ Missing'}")
    print(f"   Weather: {'✅ Ready' if weather_key else '❌ Missing'}")
    print()
    
    if not openai_key:
        return "❌ OpenAI API key required for tool calling"
    
    print(f"📝 **Request:** {user_request}\n")
    print("=" * 60)
    
    # Run travel agent with OpenAI tool calling
    result = travel_planning_agent(user_request)
    
    print("\n" + "=" * 60)
    print(f"📊 **Stats:** {len(agent_memory['conversation_log'])} actions logged")
    
    return result

if __name__ == "__main__":
    travel_request = "Plan a 4-day trip to Tokyo in April 2025. I'm interested in technology, food, and traditional culture. Find flights from San Francisco and recommend hotels."
    
    final_result = plan_travel(travel_request)
    
    print("\n🎉 **Final Result:**")
    print(final_result)