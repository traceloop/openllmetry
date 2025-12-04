"""
RAG (Retrieval-Augmented Generation) Evaluators Experiment

This example demonstrates Traceloop's RAG-specific evaluators:
- Answer Relevancy: Verifies the answer addresses the question
- Faithfulness: Ensures the answer is grounded in the retrieved context
- Agent Goal Accuracy: Validates the agent achieves its goal

These evaluators are essential for RAG applications to ensure responses
are both relevant and factually grounded in the source material.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import Predefined

# Initialize Traceloop
client = Traceloop.init()


async def retrieve_context(query: str) -> str:
    """
    Simulate retrieving context from a knowledge base.
    In a real application, this would query a vector database or search engine.
    """
    # Mock context retrieval
    knowledge_base = {
        "python": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991.",
        "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computers the ability to learn from data.",
        "rag": "Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed responses.",
    }

    # Simple keyword matching for demo
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value

    return "No relevant context found in the knowledge base."


async def generate_rag_response(question: str, context: str) -> str:
    """Generate a RAG response using OpenAI with retrieved context"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say so."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=200,
    )

    return response.choices[0].message.content


async def rag_task(row):
    """
    RAG task function that:
    1. Retrieves relevant context
    2. Generates a response grounded in that context
    3. Returns data for RAG evaluation
    """
    question = row.get("question", "")
    reference = row.get("reference_answer", "This is a demo reference answer")  # Expected/ideal answer

    # Step 1: Retrieve context
    context = await retrieve_context(question)

    # Step 2: Generate response
    completion = await generate_rag_response(question, context)

    # Return data for RAG evaluation
    return {
        "question": question,
        "completion": completion,
        "reference": reference,
    }


async def run_rag_experiment():
    """
    Run experiment with RAG evaluators.

    This experiment evaluates:
    1. Answer Relevancy - Does the answer address the question?
    2. Faithfulness - Is the answer grounded in the retrieved context?
    3. Agent Goal Accuracy - How well does the response match the reference?
    """

    print("\n" + "="*80)
    print("RAG EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment tests RAG-specific quality:\n")
    print("1. Answer Relevancy - Answer addresses the question")
    print("2. Faithfulness - Answer stays grounded in context (no hallucinations)")
    print("3. Agent Goal Accuracy - Response quality vs. reference answer")
    print("\n" + "-"*80 + "\n")

    # Configure RAG evaluators
    evaluators = [
        Predefined.agent_goal_accuracy(
            description="Compare response quality against reference answer"
        ),
    ]

    print("Running RAG experiment with evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="medical-q",
        dataset_version="v1",
        task=rag_task,
        evaluators=evaluators,
        experiment_slug="rag-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if results:
        print(f"Successfully evaluated {len(results)} RAG tasks\n")

        # Analyze RAG quality
        rag_metrics = {
            "relevant_count": 0,
            "faithful_count": 0,
            "avg_goal_accuracy": 0.0,
            "total_tasks": len(results),
        }

        accuracy_scores = []

        for i, result in enumerate(results, 1):
            print(f"Task {i}:")
            if result.task_result:
                question = result.task_result.get("question", "N/A")
                context = result.task_result.get("context", "N/A")
                print(f"  Question: {question[:50]}{'...' if len(question) > 50 else ''}")
                print(f"  Context: {context[:50]}{'...' if len(context) > 50 else ''}")

            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  {eval_name}: {eval_result}")

                    # Track metrics
                    if "relevancy" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("is_relevant"):
                            rag_metrics["relevant_count"] += 1
                    elif "faithful" in eval_name.lower() and isinstance(eval_result, dict):
                        if eval_result.get("is_faithful"):
                            rag_metrics["faithful_count"] += 1
                    elif "goal" in eval_name.lower() and isinstance(eval_result, dict):
                        if "accuracy_score" in eval_result:
                            accuracy_scores.append(eval_result["accuracy_score"])
            print()

        # Calculate average goal accuracy
        if accuracy_scores:
            rag_metrics["avg_goal_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)

        # RAG Quality Summary
        print("\n" + "="*80)
        print("RAG QUALITY SUMMARY")
        print("="*80 + "\n")
        print(f"Relevant answers: {rag_metrics['relevant_count']}/{rag_metrics['total_tasks']}")
        print(f"Faithful answers: {rag_metrics['faithful_count']}/{rag_metrics['total_tasks']}")
        print(f"Average goal accuracy: {rag_metrics['avg_goal_accuracy']:.2f}")

        relevancy_rate = (rag_metrics['relevant_count'] / rag_metrics['total_tasks']) * 100
        faithfulness_rate = (rag_metrics['faithful_count'] / rag_metrics['total_tasks']) * 100

        print(f"\nRelevancy rate: {relevancy_rate:.1f}%")
        print(f"Faithfulness rate: {faithfulness_rate:.1f}%")

        if relevancy_rate >= 90 and faithfulness_rate >= 90:
            print("\nExcellent RAG performance!")
        elif relevancy_rate >= 70 and faithfulness_rate >= 70:
            print("\nGood RAG performance, room for improvement.")
        else:
            print("\nRAG system needs improvement - check retrieval and generation.")
    else:
        print("No results to display")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:
            print(f"  - {error}")
    else:
        print("\nNo errors encountered")

    print("\n" + "="*80)
    print("RAG experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nRAG Evaluators Experiment\n")

    asyncio.run(run_rag_experiment())
