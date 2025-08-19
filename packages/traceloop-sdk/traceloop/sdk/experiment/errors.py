"""
Experiment-related error classes
"""


class ExperimentError(Exception):
    """Base exception for experiment-related errors"""
    pass


class TaskRequiredError(ExperimentError):
    """Task function is required to run an experiment"""
    
    def __init__(self, message: str = "Task function is required"):
        super().__init__(message)


class ExperimentExecutionError(ExperimentError):
    """Experiment execution failed"""
    pass


class EvaluatorError(ExperimentError):
    """Evaluator execution failed"""
    pass