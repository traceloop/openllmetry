import threading
from typing import Optional

class GuardrailsContext:
    def __init__(self):
        self._local = threading.local()
    
    def set_score(self, score: float):
        self._local.score = score
    
    def get_score(self) -> Optional[float]:
        return getattr(self._local, 'score', None)

_guardrails_context = GuardrailsContext()

def get_current_score() -> Optional[float]:
    """Get the current guardrails score from the context."""
    return _guardrails_context.get_score()

def set_current_score(score: float):
    """Set the current guardrails score in the context."""
    _guardrails_context.set_score(score) 