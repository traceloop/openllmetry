"""
Example usage of the Experiment context manager
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traceloop.sdk import Traceloop
from traceloop.sdk.experiment import Experiment

# Initialize Traceloop
client = Traceloop.init()

def run_simple_experiment_example():
    """Simple synchronous example without evaluator API calls"""
    
    with Experiment.create("simple-experiment") as experiment:
        dataset = client.datasets.get_dataset_by_slug("medical")
        
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