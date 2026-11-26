import os
from openai import OpenAI
from traceloop.sdk import Traceloop

# 1. Initialize Traceloop before calling any LLM services
# We use disable_batch=True so traces are sent immediately (perfect for local testing!)
Traceloop.init(disable_batch=True)

def main():
    # 2. Ensure your OpenAI API key is set in your environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set your OPENAI_API_KEY environment variable to run this example.")
        return

    # 3. Initialize the standard OpenAI client
    # OpenLLMetry automatically instruments this client in the background!
    client = OpenAI()

    print("🚀 Sending a prompt to OpenAI... Traceloop is monitoring this request.")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain open-source software observability in one sentence."}
            ]
        )
        
        # Print the response out to the console
        print("\n🤖 AI Response:")
        print(response.choices[0].message.content)
        print("\n✅ Trace successfully captured! Check your Traceloop dashboard.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()