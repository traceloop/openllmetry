#!/usr/bin/env python3
"""
Example script demonstrating how to use the Traceloop Experiment class.

This example shows:
1. Running a single evaluator on multiple inputs
2. Error handling for individual runs
3. Result aggregation and analysis
4. Different input scenarios

Requirements:
- TRACELOOP_API_KEY environment variable must be set
- Valid evaluator slug that exists in your Traceloop account
"""

from traceloop.sdk import Traceloop
from traceloop.sdk.experiments import Experiment


# Initialize Traceloop
Traceloop.init()


def run_basic_experiment_example():
    """Run a basic experiment with multiple inputs"""
    print("=== Basic Experiment Example ===")
    
    try:
        # Multiple test inputs for the same evaluator
        test_inputs = [
            {
                "love_only": "apples",
                "love_sentence": "My favorite fruit is apples and I love them"
            },
            {
                "love_only": "books",
                "love_sentence": "I really enjoy reading books in my spare time"
            },
            {
                "love_only": "music",
                "love_sentence": "Music brings joy to my life every day"
            },
            {
                "love_only": "coffee",
                "love_sentence": "I can't start my day without a good cup of coffee"
            }
        ]
        
        # Run experiment
        result = Experiment.run(
            evaluator_slug="What I Hate",
            inputs=test_inputs,
            timeout_in_sec=120
        )
        
        print(f"\n=== Experiment Results ===")
        print(f"Evaluator: {result.evaluator_slug}")
        print(f"Total runs: {result.total_runs}")
        print(f"Successful runs: {result.successful_runs}")
        print(f"Failed runs: {result.failed_runs}")
        print(f"Total execution time: {result.total_execution_time:.2f} seconds")
        
        # Display individual results
        print(f"\n=== Individual Results ===")
        for run_result in result.results:
            print(f"\nInput {run_result.input_index + 1}:")
            print(f"  Data: {run_result.input_data}")
            print(f"  Execution time: {run_result.execution_time:.2f}s")
            
            if run_result.error:
                print(f"  Error: {run_result.error}")
            else:
                print(f"  Result: {run_result.result}")
        
        return result
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        raise


def run_experiment_with_error_handling():
    """Demonstrate experiment with some inputs that might fail"""
    print("\n=== Experiment with Error Handling ===")
    
    try:
        # Mix of valid and potentially problematic inputs
        test_inputs = [
            {
                "love_only": "pizza",
                "love_sentence": "Pizza is my favorite food of all time"
            },
            {
                "love_only": "",  # Empty input that might cause issues
                "love_sentence": "This might fail due to empty love_only"
            },
            {
                "love_only": "travel",
                "love_sentence": "I love exploring new places and cultures"
            }
        ]
        
        result = Experiment.run(
            evaluator_slug="What I Hate",
            inputs=test_inputs,
            timeout_in_sec=60  # Shorter timeout
        )
        
        print(f"\nResults Summary:")
        print(f"Success rate: {result.successful_runs}/{result.total_runs} ({result.successful_runs/result.total_runs*100:.1f}%)")
        
        # Show only failed runs
        failed_runs = [r for r in result.results if r.error]
        if failed_runs:
            print(f"\nFailed runs:")
            for run_result in failed_runs:
                print(f"  Input {run_result.input_index + 1}: {run_result.error}")
        
        return result
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        raise


def analyze_experiment_results(result):
    """Analyze and display insights from experiment results"""
    print("\n=== Experiment Analysis ===")
    
    if result.successful_runs == 0:
        print("No successful runs to analyze")
        return
    
    # Calculate average execution time for successful runs
    successful_results = [r for r in result.results if r.error is None]
    avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
    
    print(f"Average execution time: {avg_time:.2f} seconds")
    print(f"Fastest run: {min(r.execution_time for r in successful_results):.2f}s")
    print(f"Slowest run: {max(r.execution_time for r in successful_results):.2f}s")
    
    # Show result patterns (if any)
    print(f"\nResult patterns:")
    for i, run_result in enumerate(successful_results):
        if run_result.result:
            print(f"  Run {run_result.input_index + 1}: {type(run_result.result).__name__}")


if __name__ == "__main__":
    print("Traceloop Experiment Examples")
    print("=" * 50)
    
    try:
        basic_result = run_basic_experiment_example()
        
        # error_handling_result = run_experiment_with_error_handling()
        
        # analyze_experiment_results(basic_result)
        
        print("\n" + "=" * 50)
        print("Experiment examples completed!")
        
    except Exception as e:
        print(f"Example failed with error: {e}")