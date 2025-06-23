from typing import Any, Dict, Optional, Union, Callable

from ..client.http import HTTPClient
from .base_evaluator import BaseEvaluator
from .models import EvaluationInputData, EvaluationConfig, EvaluationResult, StreamEventData


class EvaluatorClient(BaseEvaluator):
    """
    Evaluator client for executing evaluations in Traceloop.
    
    This class provides functionality to execute evaluators and get
    raw evaluation results.
    """
    
    def __init__(
        self, 
        http: HTTPClient, 
        app_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: int = 30
    ):
        super().__init__(http, app_name, api_key, api_url, project_id, timeout)

    async def evaluate_content(
        self,
        evaluator_slug: str,
        content: str,
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate content using an evaluator.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            content (str): The content to evaluate
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
            
        Example:
            ```python
            evaluator = EvaluatorClient(http, app_name)
            result = await evaluator.evaluate_content(
                evaluator_slug="toxicity-checker",
                content="This is some text to check"
            )
            
            print(f"Score: {result.score}, Reason: {result.reason}")
            ```
        """
        input_data = EvaluationInputData(content=content)
        return await self.evaluate(evaluator_slug, input_data, config, timeout, on_progress)

    def evaluate_content_sync(
        self,
        evaluator_slug: str,
        content: str,
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate content using an evaluator (synchronous version).
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            content (str): The content to evaluate
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
        """
        input_data = EvaluationInputData(content=content)
        return self.evaluate_sync(evaluator_slug, input_data, config, timeout, on_progress)

    async def evaluate_messages(
        self,
        evaluator_slug: str,
        messages: list,
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate messages using an evaluator.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            messages (list): The messages to evaluate
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
        """
        input_data = EvaluationInputData(messages=messages)
        return await self.evaluate(evaluator_slug, input_data, config, timeout, on_progress)

    def evaluate_messages_sync(
        self,
        evaluator_slug: str,
        messages: list,
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate messages using an evaluator (synchronous version).
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            messages (list): The messages to evaluate
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
        """
        input_data = EvaluationInputData(messages=messages)
        return self.evaluate_sync(evaluator_slug, input_data, config, timeout, on_progress) 