#!/usr/bin/env python3
"""
Example script demonstrating how to use the Traceloop Evaluator.

This example shows:
1. Basic evaluator execution with synchronous result waiting
2. Error handling
3. Different input schema mappings
4. Custom timeout configuration

Requirements:
- TRACELOOP_API_KEY environment variable must be set
- Valid evaluator slug that exists in your Traceloop account
"""

import asyncio
from colorama import Fore, Style, init
from traceloop.sdk.evaluators import Evaluator

# Initialize colorama for cross-platform color support
init()

# Debug/Run configuration
async def run_basic_example():
    """Run a basic evaluator example"""
    try:
        result = await Evaluator.run(
            evaluator_slug="What I Hate",
            input={
                "love_only": "apples",
                "love_sentence": "My favorite fruit is apples and I love them", 
            },
            timeout_in_sec=120
        )
        print("\n" + "*" * 100)
        print(f"{Fore.GREEN}Result from evaluator:{Style.RESET_ALL} {result}")
    except Exception as e:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} {e}")
        raise
    return result   

if __name__ == "__main__":    
    result = asyncio.run(run_basic_example())
    print(f"\n{Fore.BLUE}Final Result:{Style.RESET_ALL} {result}")