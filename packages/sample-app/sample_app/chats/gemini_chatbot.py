import os
import uuid
from datetime import datetime
import google.genai as genai
from google.genai import types
from traceloop.sdk import Traceloop
from traceloop.sdk.associations import AssociationProperty
from traceloop.sdk.decorators import workflow

# Initialize Traceloop for observability
traceloop = Traceloop.init(app_name="gemini_chatbot")

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GENAI_API_KEY"))


# Define tools for the chatbot
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 68°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 52°F",
        "Tokyo": "Clear, 62°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone."""
    # Simplified - just return current UTC time
    return f"Current time ({timezone}): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information."""
    # Simulated knowledge base
    knowledge = {
        "company policy": "Our company policy includes 15 days of vacation per year.",
        "support hours": "Support is available Monday-Friday, 9 AM - 5 PM EST.",
        "pricing": "Our pricing starts at $29/month for the basic plan.",
    }

    for key, value in knowledge.items():
        if key in query.lower():
            return value

    return "I couldn't find specific information about that in our knowledge base."


# Define function declarations for Gemini
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a specific location",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "location": types.Schema(
                        type=types.Type.STRING,
                        description="The city name, e.g., 'San Francisco'"
                    )
                },
                required=["location"]
            )
        )
    ]
)

time_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_current_time",
            description="Get the current time in a specific timezone",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "timezone": types.Schema(
                        type=types.Type.STRING,
                        description="The timezone name, e.g., 'UTC', 'PST'"
                    )
                },
                required=[]
            )
        )
    ]
)

knowledge_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_knowledge_base",
            description="Search the company knowledge base for information about policies, support, or pricing",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query"
                    )
                },
                required=["query"]
            )
        )
    ]
)


# Function to execute tool calls
def execute_function(function_name: str, args: dict) -> str:
    """Execute the requested function with given arguments."""
    if function_name == "get_weather":
        return get_weather(args.get("location", ""))
    elif function_name == "get_current_time":
        return get_current_time(args.get("timezone", "UTC"))
    elif function_name == "search_knowledge_base":
        return search_knowledge_base(args.get("query", ""))
    else:
        return f"Unknown function: {function_name}"


@workflow("chatbot_conversation")
def process_message(session_id: str, user_message: str, conversation_history: list) -> tuple[str, list]:
    """Process a single message with tool support and chat_id association."""

    # Set a conversation_id to identify the conversation using the associations API
    traceloop.associations.set([(AssociationProperty.SESSION_ID, session_id)])

    # Add user message to conversation history
    conversation_history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # Keep trying until we get a final response (handle tool calls)
    while True:
        # Generate content with tools
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=conversation_history,
            config=types.GenerateContentConfig(
                tools=[weather_tool, time_tool, knowledge_tool],
                temperature=0.7,
            )
        )

        # Check if the model wants to use a tool
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            function_name = function_call.name
            function_args = dict(function_call.args)

            print(f"[Tool Call]: {function_name}({function_args})")

            # Execute the function
            function_result = execute_function(function_name, function_args)
            print(f"[Tool Result]: {function_result}")

            # Add the model's function call to history
            conversation_history.append({
                "role": "model",
                "parts": [{"function_call": function_call}]
            })

            # Add the function result to history
            conversation_history.append({
                "role": "user",
                "parts": [{
                    "function_response": types.FunctionResponse(
                        name=function_name,
                        response={"result": function_result}
                    )
                }]
            })
        else:
            # Got a text response, we're done with this turn
            assistant_message = response.text

            # Add assistant response to conversation history
            conversation_history.append({
                "role": "model",
                "parts": [{"text": assistant_message}]
            })

            return assistant_message, conversation_history


def main():
    """Main function for interactive chatbot."""

    # Generate a unique chat_id for this conversation
    chat_id = str(uuid.uuid4())

    print(f"Starting chatbot conversation (Chat ID: {chat_id})")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("=" * 80)

    conversation_history = []

    while True:
        # Get user input
        user_message = input("\nYou: ").strip()

        # Check for exit commands
        if user_message.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! Chat session ended.")
            break

        # Skip empty messages
        if not user_message:
            continue

        # Process the message
        try:
            assistant_message, conversation_history = process_message(
                chat_id, user_message, conversation_history
            )
            print(f"\nAssistant: {assistant_message}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
