"""
Example usage of the Experiment context manager
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traceloop.sdk.experiment import Experiment


class MockDataset:
    """Mock dataset for example purposes"""
    def __init__(self):
        self.id = "mock-dataset-id"
        self.name = "mock-dataset"
        self.rows = [
            {"id": "row-1", "question": "What is the capital of France?", "context": "Geography knowledge", "expected": "Paris"},
            {"id": "row-2", "question": "What is 2+2?", "context": "Basic math", "expected": "4"},
            {"id": "row-3", "question": "Who wrote Romeo and Juliet?", "context": "Literature", "expected": "Shakespeare"}
        ]
    
    def iter_rows(self):
        for row in self.rows:
            yield row


class MockRAGPipeline:
    """Mock RAG pipeline for example purposes"""
    @staticmethod
    def run(question: str, context: str, variant: str = "default") -> str:
        # Mock responses based on variant
        if variant == "v1":
            return f"Answer to '{question}' using {context}: Basic response"
        elif variant == "v2":
            return f"Enhanced answer to '{question}' with {context}: Detailed response"
        else:
            return f"Default answer to '{question}'"


async def run_experiment_example():
    """Example of running an experiment with the context manager"""
    
    # Initialize the experiment context
    with Experiment.create("example-experiment") as experiment:
        # Load mock dataset
        dataset = MockDataset()
        
        # Process each row in the dataset
        for row in dataset.iter_rows():
            # Start tracking this evaluation run
            experiment.start_eval_run(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                dataset_row_id=row["id"],
                variant_name="rag_v1"
            )
            
            # Run mock RAG pipeline
            question = row["question"]
            context = row["context"]
            answer = MockRAGPipeline.run(question, context, "v1")
            
            # Get tracked evaluator (this would call real evaluators)
            try:
                relevancy_evaluator = Experiment.get_evaluator("relevancy-evaluator", experiment)
                
                # This would normally call the real evaluator API
                # For demo purposes, we'll simulate the call
                relevancy_score = await relevancy_evaluator.evaluate(
                    question=question,
                    context=context,
                    answer=answer
                )
                
                print(f"✅ Processed row {row['id']}: {relevancy_score}")
                
            except Exception as e:
                print(f"❌ Failed to evaluate row {row['id']}: {e}")
                # Manually log the error for demo
                experiment.log_evaluator_run(
                    evaluator_id="relevancy-evaluator",
                    evaluator_name="Relevancy Evaluator",
                    input_data={"question": question, "context": context, "answer": answer},
                    output_data={},
                    runtime_ms=100.0,
                    success=False,
                    error_message=str(e)
                )
            
            # Complete this evaluation run
            experiment.complete_eval_run()
            
            # Test variant 2
            experiment.start_eval_run(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                dataset_row_id=row["id"],
                variant_name="rag_v2"
            )
            
            answer_v2 = MockRAGPipeline.run(question, context, "v2")
            
            # Manually log evaluator run for demo (simulating evaluator call)
            experiment.log_evaluator_run(
                evaluator_id="relevancy-evaluator",
                evaluator_name="Relevancy Evaluator",
                input_data={"question": question, "context": context, "answer": answer_v2},
                output_data={"score": 0.85, "confidence": 0.92},
                runtime_ms=150.5,
                success=True
            )
            
            experiment.complete_eval_run()

    # When context exits, all data is automatically POSTed to the API
    print("🎉 Experiment completed! All data has been logged.")


def run_simple_experiment_example():
    """Simple synchronous example without evaluator API calls"""
    
    with Experiment.create("simple-experiment") as experiment:
        dataset = MockDataset()
        
        for row in dataset.iter_rows():
            experiment.start_eval_run(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                dataset_row_id=row["id"],
                variant_name="basic"
            )
            
            # Simulate some processing
            question = row["question"]
            answer = MockRAGPipeline.run(question, "basic context")
            
            # Manually log results (simulating evaluator execution)
            experiment.log_evaluator_run(
                evaluator_id="accuracy-evaluator",
                evaluator_name="Accuracy Evaluator", 
                input_data={"question": question, "expected": row["expected"], "actual": answer},
                output_data={"accuracy": 0.95, "exact_match": False},
                runtime_ms=75.0,
                success=True
            )
            
            experiment.complete_eval_run()

    print("✅ Simple experiment completed!")


if __name__ == "__main__":
    print("Running simple experiment example...")
    run_simple_experiment_example()
    
    print("\nRunning async experiment example...")
    # asyncio.run(run_experiment_example())  # Uncomment to test async version