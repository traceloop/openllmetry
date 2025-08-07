from traceloop.sdk.evaluators import Evaluator


def test_evaluator_import():
    """Test that Evaluator can be imported successfully"""
    assert Evaluator is not None

def test_evaluator_run_method_exists():
    """Test that Evaluator.run method exists and is callable"""
    assert hasattr(Evaluator, 'run')
    assert callable(getattr(Evaluator, 'run'))