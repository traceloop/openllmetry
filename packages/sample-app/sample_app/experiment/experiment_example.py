"""
Example usage of the Experiment context manager
"""

import asyncio
import os

from openai import AsyncOpenAI
from medical_prompts import refuse_medical_advice_prompt, provide_medical_info_prompt
from traceloop.sdk import Traceloop

client = Traceloop.init()


async def generate_medical_answer(prompt_text: str) -> str:
    """
    Generate a medical answer using OpenAI and the clinical guidance prompt
    """
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content


async def medical_task_refuse_advice(row):
    """Task function for refusing medical advice prompt"""
    prompt_text = refuse_medical_advice_prompt(row["question"])
    answer = await generate_medical_answer(prompt_text)
    user_description = row["question"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")
    return {"completion": answer, "prompt": prompt_text}


async def medical_task_provide_info(row):
    """Task function for providing medical info prompt"""
    prompt_text = provide_medical_info_prompt(row["question"])
    answer = await generate_medical_answer(prompt_text)
    user_description = row["question"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")
    return {"completion": answer, "prompt": prompt_text}


async def run_experiment_example():
    """Example using the new run_experiment API to compare two different prompt approaches"""

    print(
        "\033[95m🔬 Running experiment with clinical guidance prompt (refuses medical advice)...\033[0m"
    )
    results_1, errors_1 = await client.experiment.run(
        dataset_slug="medical-q",
        dataset_version="v1",
        task=medical_task_refuse_advice,
        evaluators=["medical_advice"],
        experiment_slug="medical-advice-exp",
        stop_on_error=False,
    )

    print(f"Medical Refuse Advice Results: {results_1}")
    if errors_1:
        print(f"Medical Refuse Advice Errors: {errors_1}")

    print(
        "\n\033[95m🔬 Running experiment with educational prompt (provides medical info)...\033[0m"
    )
    results_2, errors_2 = await client.experiment.run(
        dataset_slug="medical-q",
        dataset_version="v1",
        task=medical_task_provide_info,
        evaluators=["medical_advice"],
        experiment_slug="medical-advice-exp",
        stop_on_error=False,
    )

    print(f"Medical Provide Info Results: {results_2}")
    if errors_2:
        print(f"Medical Provide Info Errors: {errors_2}")

    print(
        "\n\033[92m✅ Both experiments completed! Compare the results to see "
        "how different prompting affects medical advice generation.\033[0m"
    )


if __name__ == "__main__":
    print("\033[95m🚀 Running experiment example\033[0m")
    asyncio.run(run_experiment_example())
