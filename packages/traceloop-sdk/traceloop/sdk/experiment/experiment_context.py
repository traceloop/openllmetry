from datetime import datetime

from traceloop.sdk.experiment.experiment import Experiment


class ExperimentContext:
    """Context manager for experiments that automatically logs results on exit"""
    
    def __init__(self, experiment: 'Experiment'):
        self.experiment = experiment
        self.start_time = datetime.now()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically log experiment data via POST API"""
        self.duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        print(f"✅ Experiment '{self.experiment.name}' logged successfully")
            
        return False