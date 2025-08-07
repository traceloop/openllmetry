#!/usr/bin/env python3
"""
Simple test script for the evaluators feature
"""

import os
import sys

# Add the SDK to the path for testing
sys.path.insert(0, '/Users/ninakollman/Traceloop/openllmetry/packages/traceloop-sdk')

def test_evaluators_import():
    """Test that all evaluators components can be imported"""
    try:
        from traceloop.sdk.evaluators import (
            evaluator,
            create_evaluator,
            InputExtractor,
            ExecuteEvaluatorRequest,
            ExecuteEvaluatorResponse
        )
        print("✅ Successfully imported evaluators module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import evaluators module: {e}")
        return False

def test_evaluator_creation():
    """Test creating an evaluator instance"""
    try:
        from traceloop.sdk.evaluators import create_evaluator
        
        evaluator = create_evaluator(
            slug="test-evaluator",
            evaluator_slug="accuracy-evaluator",
            name="Test evaluator",
            description="Testing the evaluators feature"
        )
        
        assert evaluator.slug == "test-evaluator"
        assert evaluator.evaluator_slug == "accuracy-evaluator"
        assert evaluator.name == "Test evaluator"
        assert evaluator.description == "Testing the evaluators feature"
        
        print("✅ Successfully created evaluator instance")
        return True
    except Exception as e:
        print(f"❌ Failed to create evaluator: {e}")
        return False

def test_model_validation():
    """Test that pydantic models work correctly"""
    try:
        from traceloop.sdk.evaluators.model import (
            InputExtractor, 
            ExecuteEvaluatorResponse,
            InputSchemaMapping
        )
        
        # Test InputExtractor
        extractor = InputExtractor(source="user_input")
        assert extractor.source == "user_input"
        
        # Test ExecuteEvaluatorResponse
        response = ExecuteEvaluatorResponse(
            execution_id="test-123",
            stream_url="/stream/test-123"
        )
        assert response.execution_id == "test-123"
        assert response.stream_url == "/stream/test-123"
        
        # Test InputSchemaMapping
        mapping = InputSchemaMapping(__root__={
            "question": InputExtractor(source="user_query"),
            "answer": InputExtractor(source="model_output")
        })
        
        print("✅ Model validation works correctly")
        return True
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing evaluators Feature Implementation\n")
    
    tests = [
        test_evaluators_import,
        test_evaluator_creation,
        test_model_validation
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! evaluators feature is ready.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())