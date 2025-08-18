"""
Generate medical answers using OpenAI and the clinical_guidance_prompt
"""
import os
from openai import OpenAI
from medical_prompts import clinical_guidance_prompt



if __name__ == "__main__":
    # The dataset question about hydration
    question = "Hi, could you explain the importance of hydration for overall health?"
    
    print("Question:", question)
    print("\n" + "="*50 + "\n")
    
    try:
        answer = generate_medical_answer(question)
        print("Generated Answer:")
        print(answer)
    except Exception as e:
        print(f"Error generating answer: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")