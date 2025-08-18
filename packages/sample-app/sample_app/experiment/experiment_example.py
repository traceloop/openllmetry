"""
Example usage of the Experiment context manager
"""
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


def run_exp(experiment: Experiment, prompt: Callable[[str], str]):
    """Simple synchronous example without evaluator API calls"""

    with experiment.run() as experiment:
        dataset = client.datasets.get_by_slug("medical")
        
        for row in dataset.rows:

            answer = generate_medical_answer(row.values["user-description"], prompt)
            
            eval_result = client.evaluator.run(
                evaluator_slug="medical_advice",
                input={"completion": answer},
                timeout_in_sec=120
            )
            print(eval_result.result)

    print("✅ Experiment version 1 completed!")


if __name__ == "__main__":
    print("Running simple experiment example...")
    experiment = Experiment(name="medical-experiment", run_data={"description": "This experiment verifies different prompt versions for a medical question answering model."})
    
    print("Running experiment with clinical guidance prompt...")
    run_exp(experiment, clinical_guidance_prompt)

    # Run experiment with educational prompt
    print("Running experiment with educational prompt...")
    run_exp(experiment, educational_prompt)