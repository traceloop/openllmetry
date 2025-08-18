from datetime import datetime
from typing import TYPE_CHECKING, Dict, Any, Optional
import uuid

if TYPE_CHECKING:
    from traceloop.sdk.experiment.experiment import Experiment


class ExperimentContext:
    """Context manager for experiments that automatically logs results on exit"""
    
    def __init__(self, experiment: 'Experiment', run_name: str = ""):
        self.experiment = experiment
        self.start_time = datetime.now()
        self.run_id = str(uuid.uuid4())
        self.run_name = run_name

    def __enter__(self):
        return self
    
    async def run_evaluator(self, 
                           evaluator_slug: str,
                           input: Dict[str, str], 
                           timeout_in_sec: int = 120):
        """
        Run an evaluator within the experiment context with the experiment's run ID
        
        Args:
            evaluator_slug: Slug of the evaluator to execute
            input: Dict mapping evaluator input field names to their values
            timeout_in_sec: Timeout in seconds for execution
            
        Returns:
            ExecutionResponse: The evaluation result
        """
        from traceloop.sdk.evaluator.evaluator import Evaluator
        from .model import ExperimentContextData
        
        context_data = ExperimentContextData(
            experiment_id=self.experiment.id,
            experiment_name=self.experiment.name,
            experiment_run_data=self.experiment.run_data,
            run_id=self.run_id,
            run_name=self.run_name
        )
        
        return await Evaluator.run(
            evaluator_slug=evaluator_slug,
            input=input,
            timeout_in_sec=timeout_in_sec,
            context_data=context_data
        )
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically log experiment data via POST API"""
        self.duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        print(f"\033[92m✅ Experiment '{self.experiment.name}' logged successfully\033[0m")
            
        return False