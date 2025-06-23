import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Union, Callable

import httpx

from ..client.http import HTTPClient
from .models import (
    EvaluationConfig,
    EvaluationInputData,
    EvaluationResult,
    EvaluationRequest,
    EvaluationResponse,
    StreamEvent,
    StreamEventData,
)


class BaseEvaluator:
    """
    Base evaluator class for executing evaluations in Traceloop.
    
    This class provides the core functionality for calling evaluator APIs
    and processing evaluation results.
    """
    
    _http: HTTPClient
    _app_name: str
    _api_key: str
    _api_url: str
    _project_id: str
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
        self._http = http
        self._app_name = app_name
        
        # Handle nullable environment variables
        import os
        env_api_key = os.getenv("TRACELOOP_API_KEY")
        env_api_url = os.getenv("TRACELOOP_BASE_URL")
        env_project_id = os.getenv("TRACELOOP_PROJECT_ID")
        
        self._api_key = api_key or env_api_key or ""
        self._api_url = "http://localhost:8000" # api_url or env_api_url or "https://api.traceloop.com"
        self._project_id = project_id or env_project_id or ""
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        
        if not self._api_key:
            raise ValueError("API key is required")
        if not self._api_url:
            raise ValueError("API URL is required")
        # if not self._project_id:
        #     raise ValueError("Project ID is required")
    
    async def evaluate(
        self,
        evaluator_slug: str,
        input_data: Union[EvaluationInputData, Dict[str, Any]],
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate data using an evaluator.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            input_data (Union[EvaluationInputData, Dict[str, Any]]): The input data for evaluation
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
            
        Example:
            ```python
            evaluator = EvaluatorClient(http, app_name)
            result = await evaluator.evaluate(
                evaluator_slug="toxicity-checker",
                input_data={
                    "content": "This is some text to check",
                    "context": {"user_id": "123"}
                }
            )
            
            print(f"Score: {result.score}, Reason: {result.reason}")
            ```
        """
        
        timeout = timeout or self._timeout
        
        # Convert dict to Pydantic model if needed
        if isinstance(input_data, dict):
            input_data = EvaluationInputData(**input_data)
        
        if isinstance(config, dict):
            config = EvaluationConfig(**config)
        
        try:
            print("Starting evaluation")
            evaluation_response = await self._start_evaluation(evaluator_slug, input_data, config, timeout)
            execution_id = evaluation_response.execution_id
            stream_url = evaluation_response.stream_url
            
            result = await self._wait_for_result(execution_id, stream_url, timeout, on_progress)
            return self._parse_result(result)
            
        except asyncio.TimeoutError:
            self._logger.warning(f"Evaluation timed out for evaluator {evaluator_slug}")
            return EvaluationResult(reason="Evaluation timed out")
            
        except Exception as e:
            self._logger.error(f"Error evaluating with {evaluator_slug}: {str(e)}")
            return EvaluationResult(reason=f"Error: {str(e)}")
    
    def evaluate_sync(
        self,
        evaluator_slug: str,
        input_data: Union[EvaluationInputData, Dict[str, Any]],
        config: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[StreamEventData], None]] = None
    ) -> EvaluationResult:
        """Evaluate data using an evaluator (synchronous version).
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            input_data (Union[EvaluationInputData, Dict[str, Any]]): The input data for evaluation
            config (Optional[Union[EvaluationConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds, defaults to instance timeout
            on_progress (Optional[Callable[[StreamEventData], None]]): Progress callback
            
        Returns:
            EvaluationResult: The result of the evaluation
        """
        
        return asyncio.run(self.evaluate(
            evaluator_slug, input_data, config, timeout, on_progress
        ))
    
    async def _start_evaluation(
        self,
        evaluator_slug: str,
        input_data: EvaluationInputData,
        config: Optional[EvaluationConfig],
        timeout: int
    ) -> EvaluationResponse:
        """Start the evaluation and return execution details."""
        
        url = f"projects/{self._project_id}/evaluators/{evaluator_slug}/evaluate-async"
        
        request_data = EvaluationRequest(
            input_data=input_data,
            timeout_ms=timeout * 1000,
            config=config
        )
        
        # Use httpx directly for async requests since HTTPClient is sync
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            response = await client.post(
                f"{self._api_url}/v2/{url}",
                json=request_data.dict(exclude_none=True),
                headers=headers
            )
            
            if response.status_code != 202:
                raise Exception(f"Failed to start evaluation: {response.status_code} {response.text}")
                
            return EvaluationResponse(**response.json())
    
    async def _wait_for_result(
        self,
        execution_id: str,
        stream_url: str,
        timeout: int,
        on_progress: Optional[Callable[[StreamEventData], None]]
    ) -> Dict[str, Any]:
        """Wait for the evaluation result via server-sent events."""
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            
            async with client.stream("GET", stream_url, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to stream results: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if time.time() - start_time > timeout:
                        raise asyncio.TimeoutError()
                    
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            event = StreamEvent(**data)
                            
                            if on_progress:
                                on_progress(event.data)
                            
                            if event.type == "completed":
                                return event.data.result or {}
                            elif event.type == "error":
                                error_msg = event.data.error or "Unknown error"
                                raise Exception(error_msg)
                                
                        except json.JSONDecodeError:
                            continue
                
                raise Exception("Stream ended without completion")
    
    def _parse_result(self, result: Dict[str, Any]) -> EvaluationResult:
        """Parse the raw result into an EvaluationResult object."""
        
        return EvaluationResult(
            result=result.get("result"),
            score=result.get("score"),
            reason=result.get("reason"),
            metadata=result.get("metadata", {}),
            raw_result=result
        ) 