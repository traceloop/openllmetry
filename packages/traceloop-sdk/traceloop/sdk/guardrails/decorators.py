import asyncio
from typing import Any, Dict, Optional, Union, Callable

from .guardrails_client import GuardrailsClient
from .models import GuardrailInputData, GuardrailConfig, GuardrailResult, GuardrailAction


class GuardrailsDecorator:
    """
    Decorator class for applying guardrails to functions.
    
    This class provides decorators for input and output validation
    using guardrail evaluators.
    """
    
    def __init__(self, client: GuardrailsClient):
        self.client = client
        
    def validate_input(
        self,
        evaluator_slug: str,
        input_extractor: Optional[Callable[[Any], Union[GuardrailInputData, Dict[str, Any]]]] = None,
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_block: Optional[Callable[[GuardrailResult], Any]] = None,
        on_retry: Optional[Callable[[GuardrailResult], Any]] = None,
        max_retries: int = 3
    ):
        """Decorator to validate function inputs using a guardrail.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            input_extractor (Optional[Callable]): Function to extract input data from function args
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds
            on_block (Optional[Callable[[GuardrailResult], Any]]): Callback when input is blocked
            on_retry (Optional[Callable[[GuardrailResult], Any]]): Callback when retry is needed
            max_retries (int): Maximum number of retries
            
        Example:
            ```python
            @client.guardrails.validate_input("toxicity-checker")
            def process_text(text: str) -> str:
                return text.upper()
            ```
        """
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                if input_extractor:
                    input_data = input_extractor(*args, **kwargs)
                else:
                    input_data = GuardrailInputData(
                        content=str(args[0]) if args else None,
                        context={"args": list(args), "kwargs": kwargs}
                    )
                
                result = await self.client.execute(
                    evaluator_slug, input_data, config, timeout
                )
                
                if result.blocked:
                    if on_block:
                        return on_block(result)
                    raise Exception(f"Input blocked by guardrail: {result.reason}")
                
                if result.retry_required:
                    if on_retry:
                        return on_retry(result)
                    raise Exception(f"Input requires retry: {result.reason}")
                
                return await func(*args, **kwargs)
            
            def sync_wrapper(*args, **kwargs):
                if input_extractor:
                    input_data = input_extractor(*args, **kwargs)
                else:
                    input_data = GuardrailInputData(
                        content=str(args[0]) if args else None,
                        context={"args": list(args), "kwargs": kwargs}
                    )
                
                result = self.client.execute_sync(
                    evaluator_slug, input_data, config, timeout
                )
                
                if result.blocked:
                    if on_block:
                        return on_block(result)
                    raise Exception(f"Input blocked by guardrail: {result.reason}")
                
                if result.retry_required:
                    if on_retry:
                        return on_retry(result)
                    raise Exception(f"Input requires retry: {result.reason}")
                
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def validate_output(
        self,
        evaluator_slug: str,
        output_extractor: Optional[Callable[[Any], Union[GuardrailInputData, Dict[str, Any]]]] = None,
        config: Optional[Union[GuardrailConfig, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        on_block: Optional[Callable[[GuardrailResult], Any]] = None,
        on_retry: Optional[Callable[[GuardrailResult], Any]] = None,
        max_retries: int = 3
    ):
        """Decorator to validate function outputs using a guardrail.
        
        Args:
            evaluator_slug (str): The slug/name of the evaluator to use
            output_extractor (Optional[Callable]): Function to extract output data from function result
            config (Optional[Union[GuardrailConfig, Dict[str, Any]]]): Optional configuration
            timeout (Optional[int]): Timeout in seconds
            on_block (Optional[Callable[[GuardrailResult], Any]]): Callback when output is blocked
            on_retry (Optional[Callable[[GuardrailResult], Any]]): Callback when retry is needed
            max_retries (int): Maximum number of retries
            
        Example:
            ```python
            @client.guardrails.validate_output("content-safety")
            def generate_text(prompt: str) -> str:
                return llm.generate(prompt)
            ```
        """
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                if output_extractor:
                    output_data = output_extractor(result)
                else:
                    output_data = GuardrailInputData(
                        content=str(result) if result else None,
                        context={"result": result}
                    )
                
                guardrail_result = await self.client.execute(
                    evaluator_slug, output_data, config, timeout
                )
                
                if guardrail_result.blocked:
                    if on_block:
                        return on_block(guardrail_result)
                    raise Exception(f"Output blocked by guardrail: {guardrail_result.reason}")
                
                if guardrail_result.retry_required:
                    if on_retry:
                        return on_retry(guardrail_result)
                    
                    for attempt in range(max_retries):
                        retry_result = await func(*args, **kwargs)
                        
                        if output_extractor:
                            retry_output_data = output_extractor(retry_result)
                        else:
                            retry_output_data = GuardrailInputData(
                                content=str(retry_result) if retry_result else None,
                                context={"result": retry_result}
                            )
                        
                        retry_guardrail_result = await self.client.execute(
                            evaluator_slug, retry_output_data, config, timeout
                        )
                        
                        if not retry_guardrail_result.retry_required:
                            return retry_result
                    
                    raise Exception(f"Max retries exceeded for guardrail: {guardrail_result.reason}")
                
                return result
            
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                if output_extractor:
                    output_data = output_extractor(result)
                else:
                    output_data = GuardrailInputData(
                        content=str(result) if result else None,
                        context={"result": result}
                    )
                
                guardrail_result = self.client.execute_sync(
                    evaluator_slug, output_data, config, timeout
                )
                
                if guardrail_result.blocked:
                    if on_block:
                        return on_block(guardrail_result)
                    raise Exception(f"Output blocked by guardrail: {guardrail_result.reason}")
                
                if guardrail_result.retry_required:
                    if on_retry:
                        return on_retry(guardrail_result)
                    
                    for attempt in range(max_retries):
                        retry_result = func(*args, **kwargs)
                        
                        if output_extractor:
                            retry_output_data = output_extractor(retry_result)
                        else:
                            retry_output_data = GuardrailInputData(
                                content=str(retry_result) if retry_result else None,
                                context={"result": retry_result}
                            )
                        
                        retry_guardrail_result = self.client.execute_sync(
                            evaluator_slug, retry_output_data, config, timeout
                        )
                        
                        if not retry_guardrail_result.retry_required:
                            return retry_result
                    
                    raise Exception(f"Max retries exceeded for guardrail: {guardrail_result.reason}")
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator 