import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails.guardrails import guardrail
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop


Traceloop.init(app_name="medical-chat-example")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required. Please set it before running this example."
    )

client = AsyncOpenAI(api_key=api_key)


# Custom callback function to handle evaluation results
def handle_medical_evaluation(evaluator_result, original_result):
    """
    Custom handler for medical advice evaluation.

    Args:
        evaluator_result: The evaluation result with 'success' and 'reason' fields
        original_result: The original AI response dict (e.g., {"text": "..."})

    Returns:
        Either the original result dict or a modified version
    """
    if not evaluator_result.success:
        # Return a modified dict with error message
        print(
            f"handle_medical_evaluation was activated - evaluator_result: {evaluator_result}"
        )
        return {"text": "There is an issue with the request. Please try again."}
    return original_result


@guardrail(
    evaluator=EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
    on_evaluation_complete=handle_medical_evaluation,
)
async def get_doctor_response_with_pii_check(patient_message: str) -> dict:
    """Get a doctor's response with PII detection guardrail and custom callback."""

    system_prompt = """You are a medical AI assistant. Provide helpful,
      general medical information and advice while being clear about your limitations.
      Always recommend consulting with qualified healthcare providers for proper diagnosis and treatment.
      Be empathetic and professional in your responses."""
    # This is the system prompt for the personal information case
    personal_info_system_prompt = """You are a medical AI assistant that provides helpful,
        general medical information tailored to the individual user.  # noqa: F841

        When personal information is available (such as age, sex, symptoms, medical history,
        lifestyle, medications, or concerns), actively incorporate it into your responses
        to make guidance more relevant and personalized.

        Adapt explanations, examples, and recommendations to the user’s context whenever possible.
        If key personal details are missing, ask concise and relevant follow-up questions
        before giving advice.

        Be clear about your limitations as an AI and do not provide diagnoses or definitive
        treatment plans. Always encourage consultation with qualified healthcare professionals
        for diagnosis, treatment, or urgent concerns.

        Maintain a professional, empathetic, and supportive tone.
        Avoid assumptions, respect privacy, and clearly distinguish general information
        from personalized considerations."""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": patient_message},
        ],
        max_tokens=500,
        temperature=0,
    )

    return {"text": response.choices[0].message.content}


# Main function using the simple example
@guardrail(evaluator="medicaladvicegiven")
async def get_doctor_response(patient_message: str) -> dict:
    """Get a doctor's response to patient input using GPT-4o."""

    system_prompt = """You are a medical AI assistant. Provide helpful,
      general medical information and advice while being clear about your limitations.
      Always recommend consulting with qualified healthcare providers for proper diagnosis and treatment.
      Be empathetic and professional in your responses."""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": patient_message},
        ],
        max_tokens=500,
        temperature=0,
    )

    # Return dict with 'text' field
    return {"text": response.choices[0].message.content}


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

            if patient_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using the medical chat. Take care!")
                break

            if not patient_input:
                print("Please enter your symptoms or medical concern.")
                continue

            print("\n🤖 Processing your request through the medical AI system...\n")

            # Get the doctor's response with guardrails applied
            doctor_response = await get_doctor_response_with_pii_check(patient_input)

            # Extract text from the response dict
            response_text = doctor_response.get("text", str(doctor_response))
            print(f"👨‍⚕️ Doctor response: {response_text}")

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
