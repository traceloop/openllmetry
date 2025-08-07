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

import asyncio
from colorama import Fore, Style, init
from traceloop.sdk import Traceloop
from traceloop.sdk.experiments import Experiment

# Initialize colorama for cross-platform color support
init()


# Initialize Traceloop
Traceloop.init()


async def run_basic_experiment_example():
    """Run a basic experiment with multiple inputs"""
    print(f"{Fore.CYAN}=== Basic Experiment Example ==={Style.RESET_ALL}")
    
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
        result = await Experiment.run(
            evaluator_slug="What I Hate",
            inputs=test_inputs,
            timeout_in_sec=120
        )
        
        print(f"\n{Fore.CYAN}=== Experiment Results ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Evaluator:{Style.RESET_ALL} {result.evaluator_slug}")
        print(f"{Fore.GREEN}Total runs:{Style.RESET_ALL} {result.total_runs}")
        print(f"{Fore.GREEN}Successful runs:{Style.RESET_ALL} {result.successful_runs}")
        print(f"{Fore.GREEN}Failed runs:{Style.RESET_ALL} {result.failed_runs}")
        print(f"{Fore.GREEN}Total execution time:{Style.RESET_ALL} {result.total_execution_time:.2f} seconds")
        
        # Display individual results
        print(f"\n{Fore.CYAN}=== Individual Results ==={Style.RESET_ALL}")
        for run_result in result.results:
            print(f"\n{Fore.YELLOW}Input {run_result.input_index + 1}:{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}Data:{Style.RESET_ALL} {run_result.input_data}")
            print(f"  {Fore.BLUE}Execution time:{Style.RESET_ALL} {run_result.execution_time:.2f}s")
            
            if run_result.error:
                print(f"  {Fore.RED}Error:{Style.RESET_ALL} {run_result.error}")
            else:
                print(f"  {Fore.GREEN}Result:{Style.RESET_ALL} {run_result.result}")
        
        return result
        
    except Exception as e:
        print(f"{Fore.RED}Error running experiment:{Style.RESET_ALL} {e}")
        raise


def run_experiment_with_error_handling():
    """Demonstrate experiment with some inputs that might fail"""
    print(f"\n{Fore.CYAN}=== Experiment with Error Handling ==={Style.RESET_ALL}")
    
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
        
        print(f"\n{Fore.GREEN}Results Summary:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Success rate:{Style.RESET_ALL} {result.successful_runs}/{result.total_runs} ({result.successful_runs/result.total_runs*100:.1f}%)")
        
        # Show only failed runs
        failed_runs = [r for r in result.results if r.error]
        if failed_runs:
            print(f"\n{Fore.RED}Failed runs:{Style.RESET_ALL}")
            for run_result in failed_runs:
                print(f"  {Fore.YELLOW}Input {run_result.input_index + 1}:{Style.RESET_ALL} {run_result.error}")
        
        return result
        
    except Exception as e:
        print(f"{Fore.RED}Error running experiment:{Style.RESET_ALL} {e}")
        raise


def analyze_experiment_results(result):
    """Analyze and display insights from experiment results"""
    print(f"\n{Fore.CYAN}=== Experiment Analysis ==={Style.RESET_ALL}")
    
    if result.successful_runs == 0:
        print(f"{Fore.YELLOW}No successful runs to analyze{Style.RESET_ALL}")
        return
    
    # Calculate average execution time for successful runs
    successful_results = [r for r in result.results if r.error is None]
    avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
    
    print(f"{Fore.GREEN}Average execution time:{Style.RESET_ALL} {avg_time:.2f} seconds")
    print(f"{Fore.GREEN}Fastest run:{Style.RESET_ALL} {min(r.execution_time for r in successful_results):.2f}s")
    print(f"{Fore.GREEN}Slowest run:{Style.RESET_ALL} {max(r.execution_time for r in successful_results):.2f}s")
    
    print(f"\n{Fore.BLUE}Result patterns:{Style.RESET_ALL}")
    for _, run_result in enumerate(successful_results):
        if run_result.result:
            print(f"  {Fore.YELLOW}Run {run_result.input_index + 1}:{Style.RESET_ALL} {type(run_result.result).__name__}")


if __name__ == "__main__":
    print(f"{Fore.MAGENTA}Traceloop Experiment Examples{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        basic_result = asyncio.run(run_basic_experiment_example())
        
        # error_handling_result = run_experiment_with_error_handling()
        
        # analyze_experiment_results(basic_result)
        
        print("\n" + "=" * 50)
        print(f"{Fore.GREEN}Experiment examples completed!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Example failed with error:{Style.RESET_ALL} {e}")