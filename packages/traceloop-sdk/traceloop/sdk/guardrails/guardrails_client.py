from typing import Any, Dict, Optional, Union, Callable

from ..client.http import HTTPClient
from .base_guardrails import BaseGuardrails
from .models import GuardrailInputData, GuardrailConfig, GuardrailResult
from ..evaluator.models import StreamEventData


class GuardrailsClient(BaseGuardrails):
    """
    Guardrails client for executing guardrails in Traceloop.
    
    This class provides functionality to execute guardrails for content validation,
    safety checks, and other evaluations.
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

    async def validate_input(
        self,
        evaluator_slug: str,
        input_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> GuardrailResult:
        """Validate input data using a guardrail evaluator.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to execute
            input_data (Union[GuardrailInputData, Dict[str, Any]]): The input data for validation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail validation
            
        Example:
            ```python
            client = Client(api_key="your-key")
            result = await client.guardrails.validate_input(
                evaluator_slug="toxicity-checker",
                input_data={
                    "content": "This is some text to check",
                    "context": {"user_id": "123"}
                }
            )
            
            if result.blocked:
                raise ValueError(f"Input blocked: {result.reason}")
            ```
        """
        return await self.execute(evaluator_slug, input_data, config, timeout, on_progress)

    def validate_input_sync(
        self,
        evaluator_slug: str,
        input_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> GuardrailResult:
        """Validate input data using a guardrail evaluator (synchronous version).
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to execute
            input_data (Union[GuardrailInputData, Dict[str, Any]]): The input data for validation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail validation
        """
        return self.execute_sync(evaluator_slug, input_data, config, timeout, on_progress)

    async def validate_output(
        self,
        evaluator_slug: str,
        output_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> GuardrailResult:
        """Validate output data using a guardrail evaluator.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to execute
            output_data (Union[GuardrailInputData, Dict[str, Any]]): The output data for validation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail validation
        """
        return await self.execute(evaluator_slug, output_data, config, timeout, on_progress)

    def validate_output_sync(
        self,
        evaluator_slug: str,
        output_data: Union[GuardrailInputData, Dict[str, Any]],
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> GuardrailResult:
        """Validate output data using a guardrail evaluator (synchronous version).
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to execute
            output_data (Union[GuardrailInputData, Dict[str, Any]]): The output data for validation
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            GuardrailResult: The result of the guardrail validation
        """
        return self.execute_sync(evaluator_slug, output_data, config, timeout, on_progress) 