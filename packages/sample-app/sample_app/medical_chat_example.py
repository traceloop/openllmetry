import asyncio
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails.guardrails import with_guardrails


Traceloop.init(
    app_name="medical-chat-example"
)


@with_guardrails(slug="medical_advice", client=Traceloop.get())
async def get_doctor_response(patient_message: str) -> str:
    """Simulate a doctor's response to patient input using an LLM."""
    # In a real implementation, this would call an actual LLM API
    # For this example, we'll simulate different responses based on keywords
    
    if "headache" in patient_message.lower():
        response = "I understand you're experiencing a headache. This could be due to stress, dehydration, or other factors. I recommend drinking plenty of water, getting adequate rest, and monitoring your symptoms. If the headache persists or becomes severe, please seek immediate medical attention."
    elif "fever" in patient_message.lower():
        response = "A fever can indicate an infection or other underlying condition. I suggest monitoring your temperature, staying hydrated, and getting plenty of rest. If your fever is above 103°F (39.4°C) or persists for more than 3 days, please contact a healthcare provider immediately."
    elif "chest pain" in patient_message.lower():
        response = "Chest pain can be a serious symptom that requires immediate medical attention. Please call emergency services or go to the nearest emergency room right away. This could be a sign of a heart condition or other serious medical issue."
    elif "dizziness" in patient_message.lower():
        response = "Dizziness can have various causes including dehydration, low blood pressure, or inner ear issues. Try sitting or lying down, and avoid sudden movements. If dizziness is severe or accompanied by other symptoms, please consult with a healthcare provider."
    else:
        response = "I understand you're not feeling well. While I can provide general information, I cannot provide a diagnosis. Please consult with a qualified healthcare provider for proper medical evaluation and treatment."
    
    return response


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
            
            print("\n🤖 Processing your request through the medical AI system...")
            
            # Get the doctor's response with guardrails applied
            doctor_response = await get_doctor_response(patient_input)
            
            print(f"👨‍⚕️ Guardrails result: {doctor_response['guardrails_result']}")
            print(f"\n👨‍⚕️ Doctor: {doctor_response['original_result']}")
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