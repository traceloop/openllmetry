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

import os
from traceloop.sdk.evaluators import Evaluator, run_evaluator, InputExtractor

# Debug/Run configuration
def run_basic_example():
    """Run a basic evaluator example"""
    try:
        result = Evaluator.run(
            evaluator_slug="What I Hate",
            input={
                "love_only": "apples",
                "love_sentence": "My favorite fruit is apples and I love them", 
            },
            timeout_in_sec=120
        )
        print("\n" + "*" * 100)
        print(f"Result from evaluator: {result}")
    except Exception as e:
        print(f"Error: {e}")
        raise
    return result   

if __name__ == "__main__":    
    result = run_basic_example()
    print(f"Result: {result}")