"""
Guardrail Example: Medical Advice Detection with Custom Evaluators

This example demonstrates how to use custom evaluators (defined in Traceloop)
to detect and control medical advice in AI-generated responses. The examples show:

1. PASS Case: General health information that should be allowed
   - Educational content about hypertension and blood pressure
   - Uses medical-advice-detector evaluator
   - Demonstrates safe general health information

2. FAIL Case: Specific diagnosis requests that should be blocked
   - User asking for diagnosis based on symptoms
   - Uses diagnosis-blocker evaluator with direct execution
   - Shows how to block medical diagnosis attempts

Custom evaluators help distinguish between:
- Safe general health information (allowed)
- Specific medical diagnoses (blocked)

Requires a Traceloop API key and custom evaluators configured in your account.
See comments in each example for required evaluator specifications.
"""

import asyncio
import os

from openai import AsyncOpenAI
from pydantic import BaseModel

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.guardrail import (
    Condition,
    OnFailure,
    Guards,
    guard,
)
from traceloop.sdk.evaluator import EvaluatorDetails


# Input models for custom evaluators
# Note: You can use either Pydantic models (recommended) OR plain dictionaries
# for guard_input. Both are supported by as_guard().
class MedicalAdviceInput(BaseModel):
    """Input model for medical advice evaluator."""
    text: str


# Initialize Traceloop - returns client with guardrails access
client = Traceloop.init(app_name="guardrail-custom-evaluator", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: Medical Advice Quality Check (PASS Case)
# ===================================================
@workflow(name="medical_advice_quality_check")
async def medical_advice_quality_check():
    """
    Demonstrate safe general health information that passes the guard.

    This example shows content that SHOULD pass - general educational health
    information that doesn't provide specific medical diagnoses.

    Custom Evaluator Required: 'medical-advice-detector'
    Expected Input Fields:
      - text: The AI-generated response
    """

    async def generate_health_info() -> str:
        """Generate general health information about hypertension."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a health educator. Provide general educational "
                               "information only. Never diagnose or prescribe.",
                },
                {
                    "role": "user",
                    "content": "What is hypertension and how can I maintain healthy blood pressure?",
                },
            ],
        )
        return completion.choices[0].message.content or ""


    guardrail = client.guardrails.create(
        guards=[Guards.custom_evaluator_guard(evaluator_slug="medicaladvice")],
        on_failure=OnFailure.return_value(value="Sorry, I can't help you with that."),
    )
    result = await guardrail.run(
        generate_health_info,
        input_mapper=lambda text: [MedicalAdviceInput(text=text)],
    )
    print(f"Health information (passed guard): {result[:200]}...")


# Example 2: Diagnosis Request Blocker (FAIL Case)
# =================================================
@workflow(name="diagnosis_request_blocker")
async def diagnosis_request_blocker():
    """
    Demonstrate blocking specific medical diagnosis requests.

    This example shows content that SHOULD fail - user requesting specific
    diagnosis based on symptoms.

    Custom Evaluator Required: 'diagnosis-blocker'
    Expected Input Fields:
      - text: The AI-generated response
    """

    async def attempt_diagnosis_request() -> str:
        """Generate response to diagnosis request (will be blocked)."""
        user_question = "I have chest pain, shortness of breath, and dizziness. Do I have a heart attack?"

        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
        )
        return completion.choices[0].message.content

    guardrail = client.guardrails.create(
        guards=[Guards.custom_evaluator_guard(evaluator_slug="diagnosis-blocker", condition_field="pass")],
        on_failure=OnFailure.raise_exception(
            "This appears to be a request for medical diagnosis. "
            "Please consult a qualified healthcare professional for symptoms that concern you."
        ),
    )
    result = await guardrail.run(
        attempt_diagnosis_request,
        input_mapper=lambda text: [{"text": text}],
    )
    print(f"Response: {result[:200]}...")



async def main():
    """Run all medical advice guardrail examples."""
    print("=" * 70)
    print("Example 1: Medical Advice Quality Check (PASS Case)")
    print("=" * 70)
    print("Note: Requires custom evaluator 'medical-advice-detector' in Traceloop")
    print("Tests: General health information that SHOULD pass the guard\n")
    try:
        await medical_advice_quality_check()
    except Exception as e:
        print(f"Skipped: {e}")

    # print("\n" + "=" * 70)
    # print("Example 2: Diagnosis Request Blocker (FAIL Case)")
    # print("=" * 70)
    # print("Note: Requires custom evaluator 'diagnosis-blocker' in Traceloop")
    # print("Tests: Specific diagnosis request that SHOULD fail the guard\n")
    # try:
    #     await diagnosis_request_blocker()
    # except Exception as e:
    #     print(f"Expected failure - guard blocked diagnosis request: {e}")

if __name__ == "__main__":
    asyncio.run(main())
