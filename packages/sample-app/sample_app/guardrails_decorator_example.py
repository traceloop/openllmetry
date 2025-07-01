import asyncio
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails.guardrails import with_guardrails


Traceloop.init(
    app_name="guardrails-decorator-example"
)


@with_guardrails(slug="medical_advice", client=Traceloop.get())
async def generate_medical_response(patient_query: str) -> str:
    """Async function that generates a medical response."""
    # Simulate AI response generation
    response = f"Based on your symptoms: {patient_query}, I recommend consulting with a healthcare provider for proper diagnosis."
    print(f"ASYNC Original result from function: {response}")
    return response


@with_guardrails(slug="medical_advice", client=Traceloop.get())
def generate_medical_response_sync(patient_query: str) -> str:
    """Sync function that generates a medical response."""
    # Simulate AI response generation
    response = f"Based on your symptoms: {patient_query}, I recommend consulting with a healthcare provider for proper diagnosis."
    return response


async def main():
    # Example with async function
    print("=== Async Function Example ===")
    result = await generate_medical_response("I have a headache and fever")
    print(f"ASYNC FINAL Result:", result)
    
    # print("\n=== Sync Function Example ===")
    # # Example with sync function
    # result_sync = generate_medical_response_sync("I have chest pain")
    # print(f"Original result: {result_sync['original_result']}")
    # print(f"Evaluator result: {result_sync['evaluator_result']}")


if __name__ == "__main__":
    asyncio.run(main()) 