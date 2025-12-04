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
        dataset_slug="medical-q",
        dataset_version="v1",
        task=security_task,
        evaluators=evaluators,
        experiment_slug="security-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if results:
        print(f"Successfully evaluated {len(results)} tasks\n")

        # Analyze security findings
        security_issues = {
            "pii_detected": 0,
            "secrets_detected": 0,
            "prompt_injection_detected": 0,
        }

        for i, result in enumerate(results, 1):
            print(f"Task {i}:")
            if result.task_result:
                query = result.task_result.get("query", "N/A")
                print(f"  Query: {query[:60]}{'...' if len(query) > 60 else ''}")

            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  {eval_name}: {eval_result}")

                    # Track security issues
                    if "pii" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("has_pii"):
                            security_issues["pii_detected"] += 1
                    elif "secret" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("has_secret"):
                            security_issues["secrets_detected"] += 1
                    elif "prompt" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("is_injection"):
                            security_issues["prompt_injection_detected"] += 1
            print()

        # Security summary
        print("\n" + "="*80)
        print("SECURITY SUMMARY")
        print("="*80 + "\n")
        print(f"PII detected: {security_issues['pii_detected']} task(s)")
        print(f"Secrets detected: {security_issues['secrets_detected']} task(s)")
        print(f"Prompt injection detected: {security_issues['prompt_injection_detected']} task(s)")

        total_issues = sum(security_issues.values())
        if total_issues == 0:
            print("\nNo security issues detected! Your responses are secure.")
        else:
            print(f"\nTotal security issues found: {total_issues}")
            print("Review the results above and consider improving your prompts/filters.")
    else:
        print("No results to display (possibly running in fire-and-forget mode)")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:
            print(f"  - {error}")
    else:
        print("\nNo errors encountered")

    print("\n" + "="*80)
    print("Security experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nSecurity Evaluators Experiment\n")


    # To run with actual dataset, uncomment:
    asyncio.run(run_security_experiment())
