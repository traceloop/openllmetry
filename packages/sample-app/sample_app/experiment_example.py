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

        
        # Run experiment (automatically batches into groups of 100 for better resource management)
        result = await Experiment.evaluate(
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


if __name__ == "__main__":
    print(f"{Fore.MAGENTA}Traceloop Experiment Examples{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        basic_result = asyncio.run(run_basic_experiment_example())
                        
        print("\n" + "=" * 50)
        print(f"{Fore.GREEN}Experiment examples completed!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Example failed with error:{Style.RESET_ALL} {e}")