"""
Example usage of the Experiment context manager
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traceloop.sdk import Traceloop
from traceloop.sdk.experiment import Experiment
from openai import OpenAI
from medical_prompts import clinical_guidance_prompt, educational_prompt
from typing import Callable

# Initialize Traceloop
client = Traceloop.init()

def generate_medical_answer(question: str, prompt: Callable[[str], str]) -> str:
    """
    Generate a medical answer using OpenAI and the clinical guidance prompt
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = prompt(question)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


async def run_exp(experiment: Experiment, prompt: Callable[[str], str]):
    """Simple synchronous example without evaluator API calls"""

    with experiment.run() as experiment:
        dataset = client.datasets.get_by_slug("medical")
        
        for row in dataset.rows:

            answer = generate_medical_answer(row.values["user-description"], prompt)
            print(f"\033[94mMedical user input:\033[0m {row.values['user-description']}")
            print(f"\033[96mMedical LLM answer:\033[0m {answer}")
            
            eval_result = await client.evaluator.run(
                evaluator_slug="medical_advice",
                input={"completion": answer},
                timeout_in_sec=120
            )
            print(f"\033[93mMedical evaluation result:\033[0m {eval_result.result}")

    print("\033[92m✅ Experiment version 1 completed!\033[0m")


if __name__ == "__main__":
    print("\033[95m🚀 Running simple experiment example...\033[0m")
    experiment = Experiment(name="medical-experiment", run_data={"description": "This experiment verifies different prompt versions for a medical question answering model."})
    
    print("\033[95m🔬 Running experiment with clinical guidance prompt...\033[0m")
    asyncio.run(run_exp(experiment, clinical_guidance_prompt))

    # Run experiment with educational prompt
    print("\033[95m📚 Running experiment with educational prompt...\033[0m")
    asyncio.run(run_exp(experiment, educational_prompt))
    