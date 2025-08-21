"""
Example usage of the Experiment context manager
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traceloop.sdk import Traceloop
from openai import OpenAI
from medical_prompts import refuse_medical_advice_prompt, provide_medical_info_prompt

# Initialize Traceloop
client = Traceloop.init()

def generate_medical_answer(prompt_text: str) -> str:
    """
    Generate a medical answer using OpenAI and the clinical guidance prompt
    """
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def medical_task_refuse_advice(row):
    """Task function for refusing medical advice prompt"""
    prompt_text = refuse_medical_advice_prompt(row["user-description"])
    answer = generate_medical_answer(prompt_text)
    user_description = row["user-description"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")
    return {"completion": answer, "prompt": prompt_text}

def medical_task_provide_info(row):
    """Task function for providing medical info prompt"""
    prompt_text = provide_medical_info_prompt(row["user-description"])
    answer = generate_medical_answer(prompt_text)
    user_description = row["user-description"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")
    return {"completion": answer, "prompt": prompt_text}


def run_experiment_example():
    """Example using the new run_experiment API to compare two different prompt approaches"""
    
    print("\033[95m🔬 Running experiment with clinical guidance prompt (refuses medical advice)...\033[0m")
    experiment_id_1, results_1 = asyncio.run(client.experiment.run(
        dataset_slug="medical",
        dataset_version="v1",
        task=medical_task_refuse_advice,
        evaluators=["medical_advice"],
        experiment_slug="medical-clinical-guidance-refuse",
        exit_on_error=False,
    ))

    print(f"\nExperiment ID (Clinical - Refuses): {experiment_id_1}")
    print(f"Results: {results_1}")

    print("\n\033[95m🔬 Running experiment with educational prompt (provides medical info)...\033[0m")
    experiment_id_2, results_2 = asyncio.run(client.experiment.run(
        dataset_slug="medical",
        dataset_version="v1",
        task=medical_task_provide_info,
        evaluators=["medical_advice"],
        experiment_slug="medical-educational-comprehensive",
        exit_on_error=False,
    ))

    print(f"\nExperiment ID (Educational - Comprehensive): {experiment_id_2}")
    print(f"Results: {results_2}")
    
    print("\n\033[92m✅ Both experiments completed! Compare the results to see how different prompting affects medical advice generation.\033[0m")


if __name__ == "__main__":
    print("\033[95m🚀 Running experiment example with new API...\033[0m")
    run_experiment_example()