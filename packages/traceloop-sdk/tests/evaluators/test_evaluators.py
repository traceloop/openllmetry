#!/usr/bin/env python3
"""
Simple test script for the evaluators feature
"""

from traceloop.sdk.evaluators import create_evaluator


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
        mapping = InputSchemaMapping(root={
            "question": InputExtractor(source="user_query"),
            "answer": InputExtractor(source="model_output")
        })
        
        print("✅ Model validation works correctly")
        return True
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def test_evaluator_run_signature():
    """Test that the Evaluator.run method has the correct signature"""
    try:
        from traceloop.sdk.evaluators.evaluator import Evaluator
        import inspect
        
        # Get the run method signature
        sig = inspect.signature(Evaluator.run)
        params = list(sig.parameters.keys())
        
        # Check that it has the expected parameters (without callback and wait_for_result)
        expected_params = ['evaluator_slug', 'input_schema_mapping', 'timeout']
        actual_params = [p for p in params if p != 'cls']  # Exclude cls parameter
        
        assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
        
        # Check return type annotation
        return_annotation = sig.return_annotation
        expected_return = "Dict[str, Any]"
        
        # Check if it's the correct type (handle different representation formats)
        if hasattr(return_annotation, '__origin__'):
            # For typing.Dict[str, Any]
            assert str(return_annotation).startswith('typing.Dict'), f"Expected Dict return type, got {return_annotation}"
        else:
            # For dict[str, Any] or other formats
            assert 'Dict' in str(return_annotation) or 'dict' in str(return_annotation), f"Expected Dict return type, got {return_annotation}"
        
        print("✅ Evaluator.run method has correct synchronous-only signature")
        return True
    except Exception as e:
        print(f"❌ Evaluator.run signature test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing evaluators Feature Implementation\n")
    
    tests = [
        test_evaluators_import,
        test_evaluator_creation,
        test_model_validation,
        test_evaluator_run_signature
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