import asyncio
import logging
from typing import Any, Dict, Optional, Union, Callable

from ..client.http import HTTPClient
from ..evaluator.evaluator_client import EvaluatorClient
from ..evaluator.models import EvaluationInputData, EvaluationConfig, EvaluationResult
from .models import (
    GuardrailAction,
    GuardrailConfig,
    GuardrailInputData,
    GuardrailResult,
)


class BaseGuardrails:
    """
    Base guardrails class for executing guardrails in Traceloop.
    
    This class uses evaluators to get evaluation results and makes
    guardrail decisions based on those results (block/pass/retry).
    """
    
    _evaluator: EvaluatorClient
    _app_name: str
    _timeout: int
    _logger: logging.Logger
    
    def __init__(
        self, 
        http: HTTPClient, 
        app_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: int = 30
    ):
        self._app_name = app_name
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        
        # Create the underlying evaluator client
        self._evaluator = EvaluatorClient(
            http, app_name, api_key, api_url, project_id, timeout
        )
    
    async def execute(
        self,
        evaluator_slug: str,
        input_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable] = None
    ) -> GuardrailResult:
        """Execute a guardrail asynchronously.
        
        This method uses an evaluator to assess the input data and then
        makes a guardrail decision based on the evaluation result.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            input_data (Union[GuardrailInputData, Dict[str, Any]]): The input data for evaluation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail decision
            
        Example:
            ```python
            client = Client(api_key="your-key")
            result = await client.guardrails.execute(
                evaluator_slug="toxicity-checker",
                input_data={
                    "content": "This is some text to check",
                    "context": {"user_id": "123"}
                }
            )
            
            if result.blocked:
                print(f"Content blocked: {result.reason}")
            elif result.pass_through:
                print("Content is safe")
            ```
        """
        
        timeout = timeout or self._timeout
        
        # Convert guardrail input to evaluation input
        if isinstance(input_data, dict):
            input_data = GuardrailInputData(**input_data)
        
        eval_input = EvaluationInputData(
            content=input_data.content,
            messages=input_data.messages,
            context=input_data.context,
            metadata=input_data.metadata
        )
        
        # Convert guardrail config to evaluation config if provided
        eval_config = None
        if config:
            if isinstance(config, dict):
                config = GuardrailConfig(**config)
            eval_config = EvaluationConfig(
                parameters=config.parameters,
                thresholds=config.thresholds,
                settings=config.settings
            )
        
        try:
            # Use the evaluator to get the evaluation result
            eval_result = await self._evaluator.evaluate(
                evaluator_slug, eval_input, eval_config, timeout, on_progress
            )
            
            # Make guardrail decision based on evaluation result
            return self._make_guardrail_decision(eval_result)
            
        except Exception as e:
            self._logger.error(f"Error executing guardrail {evaluator_slug}: {str(e)}")
            return GuardrailResult(action=GuardrailAction.PASS, reason=f"Error: {str(e)}")
    
    def execute_sync(
        self,
        evaluator_slug: str,
        input_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable] = None
    ) -> GuardrailResult:
        """Execute a guardrail synchronously.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            input_data (Union[GuardrailInputData, Dict[str, Any]]): The input data for evaluation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail decision
        """
        
        return asyncio.run(self.execute(
            evaluator_slug, input_data, config, timeout, on_progress
        ))
    
    def _make_guardrail_decision(self, eval_result: EvaluationResult) -> GuardrailResult:
        """Make a guardrail decision based on the evaluation result.
        
        This method interprets the evaluation result and decides whether to
        block, pass, or retry based on common evaluation patterns.
        
        Args:
            eval_result (EvaluationResult): The result from the evaluator
            
        Returns:
            GuardrailResult: The guardrail decision
        """
        
        # Default to pass if no evaluation result
        if not eval_result or not eval_result.raw_result:
            return GuardrailResult(
                action=GuardrailAction.PASS,
                result=eval_result.result if eval_result else None,
                reason="No evaluation result available",
                score=eval_result.score if eval_result else None,
                metadata=eval_result.metadata if eval_result else {}
            )
        
        raw_result = eval_result.raw_result
        
        # Check for explicit pass/block/retry signals in the result
        if raw_result.get("pass", True):
            action = GuardrailAction.PASS
        elif raw_result.get("retry", False):
            action = GuardrailAction.RETRY
        elif raw_result.get("block", False):
            action = GuardrailAction.BLOCK
        else:
            # Use score-based decision if available
            score = eval_result.score
            if score is not None:
                # Default thresholds - can be overridden via config
                if score < 0.3:  # Low score = block
                    action = GuardrailAction.BLOCK
                elif score > 0.7:  # High score = pass
                    action = GuardrailAction.PASS
                else:  # Medium score = retry for more evaluation
                    action = GuardrailAction.RETRY
            else:
                # Default to pass if no clear signals
                action = GuardrailAction.PASS
        
        return GuardrailResult(
            action=action,
            result=eval_result.result,
            reason=eval_result.reason,
            score=eval_result.score,
            metadata=eval_result.metadata
        ) 