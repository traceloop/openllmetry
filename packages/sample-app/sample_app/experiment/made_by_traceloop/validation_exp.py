"""
Validation Evaluators Experiment

This example demonstrates Traceloop's validation evaluators:
- JSON Validation: Validates JSON format and optional schema compliance
- SQL Validation: Validates SQL query syntax
- Regex Validation: Validates text matches regex patterns
- Placeholder Regex: Validates dynamic text with placeholders

These evaluators are essential for structured output validation,
ensuring AI-generated code, queries, and formatted data are correct.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import Predefined

# Initialize Traceloop
client = Traceloop.init()


async def generate_structured_output(prompt: str, output_format: str) -> str:
    """
    Generate structured output (JSON, SQL, etc.) using OpenAI.

    Args:
        prompt: The user's request
        output_format: Expected format (json, sql, regex, etc.)
    """
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompts = {
        "json": "You are a helpful assistant that responds ONLY with valid JSON. No explanation, just JSON.",
        "sql": "You are a SQL expert. Generate ONLY valid SQL queries. No explanation, just the SQL query.",
        "regex": "You are a regex expert. Generate ONLY the regex pattern. No explanation, just the pattern.",
        "text": "You are a helpful assistant. Generate the requested text content.",
    }

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompts.get(output_format, system_prompts["text"])},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more structured outputs
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


async def validation_task(row):
    """
    Task function that generates structured outputs for validation.
    Returns different types of structured data based on the task type.
    """
    prompt = row.get("prompt", "")
    task_type = row.get("task_type", "json")  # json, sql, regex, text
    placeholder_value = row.get("placeholder_value", "")

    # Generate the appropriate output
    output = await generate_structured_output(prompt, task_type)

    # Return data for validation evaluators
    return {
        "text": output,
        "placeholder_value": placeholder_value,
    }


async def run_validation_experiment():
    """
    Run experiment with validation evaluators.

    This experiment validates:
    1. JSON Validation - Proper JSON format and schema
    2. SQL Validation - Valid SQL syntax
    3. Regex Validation - Pattern matching
    4. Placeholder Regex - Dynamic pattern validation
    """

    print("\n" + "="*80)
    print("VALIDATION EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment validates structured AI outputs:\n")
    print("1. JSON Validation - Ensures valid JSON format")
    print("2. SQL Validation - Verifies SQL query syntax")
    print("3. Regex Validation - Validates pattern matching")
    print("4. Placeholder Regex - Dynamic text validation")
    print("\n" + "-"*80 + "\n")

    # Configure validation evaluators
    json_schema = '''{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"}
        },
        "required": ["name", "email"]
    }'''

    evaluators = [
        Predefined.json_validator(
            enable_schema_validation=True,
            schema_string=json_schema
        ),
        Predefined.sql_validator(),
        Predefined.regex_validator(
            regex=r"^\d{3}-\d{2}-\d{4}$",  # SSN format
            should_match=True,
            case_sensitive=True
        ),
        Predefined.placeholder_regex(
            regex=r"^user_.*",
            placeholder_name="username",
            should_match=True
        ),
    ]

    print("Running experiment with validation evaluators:")
    for evaluator in evaluators:
        config_str = ", ".join(f"{k}={v}" for k, v in evaluator.config.items() if k != "description" and len(str(v)) < 50)
        print(f"  - {evaluator.slug}")
        if config_str:
            print(f"    Config: {config_str}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="validation-dataset",
        dataset_version="v1",
        task=validation_task,
        evaluators=evaluators,
        experiment_slug="validation-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if results:
        print(f"Successfully evaluated {len(results)} tasks\n")

        # Track validation results
        validation_stats = {
            "valid_json": 0,
            "valid_sql": 0,
            "regex_pass": 0,
            "placeholder_pass": 0,
            "total": len(results),
        }

        for i, result in enumerate(results, 1):
            print(f"Task {i}:")
            if result.task_result:
                text = result.task_result.get("text", "")
                print(f"  Output: {text[:60]}{'...' if len(text) > 60 else ''}")

            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  {eval_name}: {eval_result}")

                    # Track validation success
                    if "json" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("is_valid_json"):
                            validation_stats["valid_json"] += 1
                    elif "sql" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("is_valid_sql"):
                            validation_stats["valid_sql"] += 1
                    elif "regex" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("regex_pass"):
                            if "placeholder" in eval_name.lower():
                                validation_stats["placeholder_pass"] += 1
                            else:
                                validation_stats["regex_pass"] += 1
            print()

        # Validation Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80 + "\n")
        print(f"Valid JSON: {validation_stats['valid_json']}/{validation_stats['total']} "
              f"({validation_stats['valid_json']/validation_stats['total']*100:.0f}%)")
        print(f"Valid SQL: {validation_stats['valid_sql']}/{validation_stats['total']} "
              f"({validation_stats['valid_sql']/validation_stats['total']*100:.0f}%)")
        print(f"Regex Pass: {validation_stats['regex_pass']}/{validation_stats['total']} "
              f"({validation_stats['regex_pass']/validation_stats['total']*100:.0f}%)")
        print(f"Placeholder Pass: {validation_stats['placeholder_pass']}/{validation_stats['total']} "
              f"({validation_stats['placeholder_pass']/validation_stats['total']*100:.0f}%)")

        total_valid = sum([validation_stats['valid_json'], validation_stats['valid_sql'],
                          validation_stats['regex_pass'], validation_stats['placeholder_pass']])
        total_checks = validation_stats['total'] * 4
        overall_rate = (total_valid / total_checks) * 100

        print(f"\nOverall validation rate: {overall_rate:.1f}%")

        if overall_rate >= 90:
            print("Excellent! Structured outputs are highly reliable.")
        elif overall_rate >= 70:
            print("Good validation rate. Some improvements possible.")
        else:
            print("Low validation rate. Review prompts and output formatting.")

    else:
        print("No results to display")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:
            print(f"  - {error}")
    else:
        print("\nNo errors encountered")

    print("\n" + "="*80)
    print("Validation experiment completed!")
    print("="*80 + "\n")



async def run_validation_examples():
    """
    Run live examples showing each validator in action.
    """
    print("\n" + "="*80)
    print("LIVE VALIDATION EXAMPLES")
    print("="*80 + "\n")

    # Example 1: JSON Validation
    print("Example 1: JSON Validation\n")
    json_prompts = [
        "Generate a user profile JSON with name, age, and email for John Doe, age 30",
        "Create a product JSON with id, title, and price",
    ]

    print("Generating JSON outputs...")
    for prompt in json_prompts[:1]:  # Just show one for demo
        output = await generate_structured_output(prompt, "json")
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    # Example 2: SQL Validation
    print("\n" + "-"*80 + "\n")
    print("Example 2: SQL Validation\n")
    sql_prompts = [
        "Write a SQL query to select all users older than 18",
        "Create a query to count active customers",
    ]

    print("Generating SQL queries...")
    for prompt in sql_prompts[:1]:  # Just show one for demo
        output = await generate_structured_output(prompt, "sql")
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    # Example 3: Regex patterns
    print("\n" + "-"*80 + "\n")
    print("Example 3: Pattern Validation\n")

    patterns = [
        ("Email validation", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        ("Phone number", r"^\(\d{3}\) \d{3}-\d{4}$"),
        ("URL validation", r"^https?://[^\s/$.?#].[^\s]*$"),
    ]

    print("Common validation patterns:")
    for name, pattern in patterns:
        print(f"  {name}: {pattern}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\nValidation Evaluators Experiment\n")

    asyncio.run(run_validation_experiment())

