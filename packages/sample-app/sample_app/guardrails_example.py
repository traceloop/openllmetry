"""
Example usage of the guardrails decorator.

This example shows how to use the guardrails decorator to evaluate content
and make runtime decisions based on the calculated score.
"""

import asyncio

from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import InputExtractor

Traceloop.init(app_name="guardrails-example")


async def main():
    """Main function to demonstrate guardrails usage."""

    medical_answer = "Hello, I understand your concerns. Seeing blood in your stool can be alarming, but it doesn't always mean you have colon cancer. There are other conditions that can cause this symptom, such as hemorrhoids, anal fissures, or gastrointestinal infections. It's good that you've had blood tests done. In addition to blood tests, your healthcare provider may recommend a colonoscopy to further investigate the cause of the bleeding. It's important to follow up with your healthcare provider to discuss your symptoms and test results in detail. They will be able to provide you with an accurate diagnosis and appropriate treatment plan based on your individual situation. If you have any further questions or concerns, feel free to ask."
    
    print("Guardrails Example Starting")

    try:
        traceloop_client = Traceloop.get()
        result = await traceloop_client.guardrails.execute_evaluator(
            slug="medical_advice", 
            data={"completion": InputExtractor(source=medical_answer)}
        )
        print("Result: ", result)
    except Exception as e:
        print(f"Traceloop initialization failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 