"""
Example usage of the Experiment context manager
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traceloop.sdk import Traceloop
from openai import OpenAI
from medical_prompts import clinical_guidance_prompt

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

def medical_task_clinical(row):
    """Task function for clinical guidance prompt"""
    prompt_text = clinical_guidance_prompt(row["user-description"])
    answer = generate_medical_answer(prompt_text)
    # print(f"\033[94mMedical user input:\033[0m {row.values['user-description']}")
    # print(f"\033[96mMedical LLM answer:\033[0m {answer}")
    user_description = row["user-description"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")
    return {"completion": answer, "prompt": prompt_text}

def medical_task_educational(row):
    # """Task function for educational prompt"""
    # answer = generate_medical_answer(row.values["user-description"], educational_prompt)
    # print(f"\033[94mMedical user input:\033[0m {row.values['user-description']}")
    # print(f"\033[96mMedical LLM answer:\033[0m {answer}")
    user_description = row.values["user-description"]
    print(f"\033[94mMedical user input:\033[0m {user_description}")

    return "This is A generated answer for AI doctor"


def run_experiment_example():
    """Example using the new run_experiment API"""
    
    # Run experiment with clinical guidance prompt
    print("\033[95m🔬 Running experiment with clinical guidance prompt...\033[0m")
    experiment_id, results = asyncio.run(client.experiment.run(
        dataset_slug="medical",
        dataset_version="v1",
        task=medical_task_clinical,
        evaluators=[("medical_advice", None), ("medical_advice", None)],
        experiment_slug="medical-clinical-guidance-1",
        exit_on_error=False,
    ))

    print(f"\nExperiment ID: {experiment_id}\n")
    print(f"\nResults: {results}")
    
    # if results:
    #     print(f"\033[92m✅ Experiment {experiment_id} completed with {len(results['results'])} results!\033[0m")
    #     if results['errors']:
    #         print(f"\033[91m❌ {len(results['errors'])} errors occurred\033[0m")
    
    # Run experiment with educational prompt  
    # print("\033[95m📚 Running experiment with educational prompt...\033[0m")
    # asyncio.run(client.experiment.run(
    #     dataset_slug="medical",
    #     task=medical_task_educational,
    #     evaluators=["medical_advice"],
    #     experiment_slug="medical-educational",
    #     exit_on_error=False,
    # ))
    
    # if results_2:
    #     print(f"\033[92m✅ Experiment {experiment_id_2} completed with {len(results_2['results'])} results!\033[0m")
    #     if results_2['errors']:
    #         print(f"\033[91m❌ {len(results_2['errors'])} errors occurred\033[0m")


if __name__ == "__main__":
    print("\033[95m🚀 Running experiment example with new API...\033[0m")
    run_experiment_example()