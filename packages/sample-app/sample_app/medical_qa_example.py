#!/usr/bin/env python3
"""
Medical Doctor Q&A Example using LLM with OpenTelemetry instrumentation.
Supports both single question mode and batch processing of 20 sample questions.

Usage:
    python medical_qa_example.py --mode single
    python medical_qa_example.py --mode batch --save
"""

import argparse
import json
import os
from typing import List, Dict
import openai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
import dotenv

dotenv.load_dotenv()

# Sample medical questions dataset
SAMPLE_QUESTIONS = [
    "What are the common symptoms of the flu?",
    "How long should I take antibiotics prescribed by my doctor?",
    "What's the difference between a cold and allergies?",
    "When should I see a doctor for a headache?",
    "What are the warning signs of a heart attack?",
    "How can I lower my blood pressure naturally?",
    "What vaccines do adults need?",
    "How much sleep do I need each night?",
    "What are the symptoms of diabetes?",
    "How can I prevent food poisoning?",
    "What should I do if I have chest pain?",
    "How often should I get a physical exam?",
    "What are the signs of dehydration?",
    "When is fever dangerous?",
    "How can I boost my immune system?",
    "What are the symptoms of a concussion?",
    "How do I know if a cut needs stitches?",
    "What's the recommended daily water intake?",
    "How can I manage stress effectively?",
    "What are the early signs of skin cancer?"
]

# Initialize OpenAI client (will be configured in main())
client = None

# Initialize Traceloop for OpenTelemetry instrumentation
Traceloop.init()


@workflow(name="medical_response_generation")
def get_medical_response(question: str) -> Dict[str, str]:
    """Generate a medical response to a patient question."""
    system_prompt = """You are a knowledgeable and compassionate medical doctor.
    Provide helpful, accurate medical information while being clear about limitations:
    - Give evidence-based medical advice when appropriate
    - Always recommend consulting a healthcare provider for serious symptoms
    - Be empathetic and professional
    - Clearly state when something requires immediate medical attention
    - Avoid giving specific diagnoses without examination
    - Include relevant disclaimers about seeking professional medical care"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        return {
            "question": question,
            "answer": answer,
            "status": "success"
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"Error generating response: {str(e)}",
            "status": "error"
        }


def run_single_mode(question: str = None) -> Dict[str, str]:
    """Run in single question mode."""
    if not question:
        question = input("Please enter your medical question: ")

    print(f"\nPatient Question: {question}")
    print("Doctor is thinking...")

    result = get_medical_response(question)

    print("\nDoctor's Response:")
    print(result['answer'])

    return result


def run_batch_mode(questions: List[str] = None) -> List[Dict[str, str]]:
    """Run in batch mode with sample questions."""
    questions = questions or SAMPLE_QUESTIONS
    results = []

    print(f"Processing {len(questions)} medical questions in batch mode...")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question}")
        result = get_medical_response(question)
        results.append(result)

        if result['status'] == 'success':
            print("✓ Response generated")
        else:
            print(f"✗ Error: {result['answer']}")

    return results


def save_results(results: List[Dict[str, str]], filename: str = "medical_qa_results.json"):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Medical Doctor Q&A using LLM")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="Run mode: single question or batch processing")
    parser.add_argument("--question", type=str, help="Question for single mode")
    parser.add_argument("--save", action="store_true", help="Save batch results to JSON file")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    try:
        # Initialize OpenAI client
        global client
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")

        client = openai.OpenAI(api_key=api_key)

        if args.mode == "single":
            # Single question mode
            result = run_single_mode(args.question)
            if args.save:
                save_results([result], "single_qa_result.json")

        else:
            # Batch mode
            results = run_batch_mode()

            # Print summary
            successful = sum(1 for r in results if r['status'] == 'success')
            print("\n" + "=" * 60)
            print(f"Batch processing complete: {successful}/{len(results)} successful responses")

            if args.save:
                save_results(results)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
