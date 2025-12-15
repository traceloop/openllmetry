import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails.guardrails import guardrail


Traceloop.init(
    app_name="medical-chat-example"
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it before running this example.")

client = AsyncOpenAI(api_key=api_key)


@guardrail(slug="medical-advice-given")
async def get_doctor_response(patient_message: str) -> str:
    """Get a doctor's response to patient input using GPT-4o."""
    
    system_prompt = """You are a medical AI assistant. Provide helpful, 
      general medical information and advice while being clear about your limitations.
      Always recommend consulting with qualified healthcare providers for proper diagnosis and treatment. 
      Be empathetic and professional in your responses."""
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": patient_message}
        ],
        max_tokens=500,
        temperature=0
    )

    ai_doc_answer = response.choices[0].message.content
    
    return ai_doc_answer


async def medical_chat_session():
    """Run an interactive medical chat session."""
    print("🏥 Welcome to the Medical Chat")
    print("=" * 50)
    print("This example simulates a conversation between a patient and a doctor.")
    print("The doctor's responses are processed through guardrails to ensure safety.")
    print("Type 'quit' to exit the chat.\n")
    
    while True:
        try:
            patient_input = input("Patient: ").strip()
            
            if patient_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the medical chat. Take care!")
                break
            
            if not patient_input:
                print("Please enter your symptoms or medical concern.")
                continue
            
            print("\n🤖 Processing your request through the medical AI system...\n")
            
            # Get the doctor's response with guardrails applied
            doctor_response = await get_doctor_response(patient_input)

            print(f"👨‍⚕️ Doctor response: {doctor_response}")
    
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")


async def main():
    """Main function to run the medical chat example."""
    await medical_chat_session()


if __name__ == "__main__":
    asyncio.run(main()) 
