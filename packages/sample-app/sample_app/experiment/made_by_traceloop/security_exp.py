"""
Security Evaluators Experiment

This example demonstrates Traceloop's security evaluators:
- PII Detector: Identifies personal information exposure
- Secrets Detector: Monitors for credential and key leaks
- Prompt Injection: Detects prompt injection attempts

These evaluators help ensure your AI applications don't leak sensitive data
or fall victim to prompt injection attacks.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_response(prompt: str) -> str:
    """Generate a response using OpenAI"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )

    return response.choices[0].message.content


async def security_task(row):
    """
    Task function that processes user queries.
    Returns text that will be evaluated for security issues.
    """
    user_query = row.get("query", "")

    # Generate response
    response = await generate_response(user_query)

    # Return data for evaluation
    return {
        "text": response,  # The text to check for PII, secrets, and prompt injection
        "query": user_query,
    }


async def run_security_experiment():
    """
    Run experiment with security evaluators.

    This experiment will evaluate responses for:
    1. PII (Personal Identifiable Information)
    2. Secrets (API keys, passwords, tokens)
    3. Prompt Injection attempts
    """

    print("\n" + "="*80)
    print("SECURITY EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment will test three critical security evaluators:\n")
    print("1. PII Detector - Identifies personal information (names, emails, SSN, etc.)")
    print("2. Secrets Detector - Finds API keys, passwords, and credentials")
    print("3. Prompt Injection - Detects attempts to manipulate the AI system")
    print("\n" + "-"*80 + "\n")

    # Configure security evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.7,),
        EvaluatorMadeByTraceloop.secrets_detector(),
        EvaluatorMadeByTraceloop.prompt_injection(threshold=0.6),
    ]

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="security",  # Set a dataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=security_task,
        evaluators=evaluators,
        experiment_slug="security-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Security experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nSecurity Evaluators Experiment\n")

    # To run with actual dataset, uncomment:
    asyncio.run(run_security_experiment())
