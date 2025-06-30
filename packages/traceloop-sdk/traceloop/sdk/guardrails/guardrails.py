from typing import Dict, Any, Optional, Callable
from aiohttp.client import ClientSession
from traceloop.sdk.client.http import HTTPClient
from dataclasses import asdict
from .types import ExecuteEvaluatorRequest, InputExtractor, EvaluatorResponse, StreamEventData, StreamEvent
from typing import Any, Dict, List, Union
import time

import httpx

class Guardrails:        
    def __init__(self, http: HTTPClient, app_name: str, api_key: str):
        self._http = http
        self._app_name = app_name
        self._flow = "guardrails"
        self._api_key = api_key

    async def execute_evaluator(self, slug: str, data: Dict[str, InputExtractor]) -> Dict[str, Any]:
        """Execute evaluator and return accumulated SSE event data."""
        url = f"evaluators/slug/{slug}/execute"
        
        try:
            # Make POST request to evaluator endpoint
            response = await self._make_post_request(url, data)

            print("Response: ", response)
            
            if response:
                # Handle SSE streaming
                response_from_stream = await self._wait_for_result(stream_url=response["stream_url"], execution_id=response["execution_id"], timeout=120)
                print("Response from stream: ", response_from_stream)
                return response_from_stream
            else:
                # Handle direct response
                return response or {}
                
        except Exception as e:
            # Log error and return empty data
            print(f"Error executing evaluator {slug}. Error: {str(e)}")
            return {}
    
    async def _make_post_request(self, url: str, data: Dict[str, InputExtractor]) -> Optional[Dict[str, Any]]:
        """Make POST request using the HTTP client."""
        try:
            print("Making POST request to url: ", url)

            request_body = ExecuteEvaluatorRequest(input_schema_mapping=data)

            body = asdict(request_body)

            print("Request body: ", body)

            response = self._http.post(url, body)

            return response
        except Exception as e:
            print(f"Error making POST request to {url}: {str(e)}")
            return None
    
    
    async def _wait_for_result(
        self,
        execution_id: str,
        stream_url: str,
        timeout: int,
    ) -> Dict[str, Any]:
        """Wait for the evaluation result via server-sent events."""
        print("Waiting for result with execution_id: ", execution_id, "and stream_url: ", stream_url)
        
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache"
                }
                print("Headers: ", headers)

                stream_url = f"http://localhost:3002{stream_url}"
                print("Stream URL: ", stream_url)

                async with client.stream("GET", stream_url, headers=headers) as response:
                    print(f"Response status: {response.status_code}")
                    print("Response from stream: ", response)


                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"Failed to stream results: {response.status_code}, body: {error_text}")
                    
                    response_text = await response.aread()
                    print("GAL GAL - Response text: ", response_text)

                    print("Stream ended without explicit completion")
                    return response_text

        except httpx.ConnectError as e:
            print(f"Connection error: {e}")
            raise Exception(f"Failed to connect to stream URL: {stream_url}. Error: {e}")
        except httpx.TimeoutException as e:
            print(f"Timeout error: {e}")
            raise Exception(f"Stream request timed out: {e}")
        except Exception as e:
            print(f"Unexpected error in _wait_for_result: {e}")
            raise

    def _parse_result(self, result: Dict[str, Any]) -> EvaluatorResponse:
        """Parse the raw result into an EvaluatorResponse object."""
        return EvaluatorResponse(
            result=result.get("result"),
            score=result.get("score"),
            reason=result.get("reason"),
            metadata=result.get("metadata", {}),
            raw_result=result
        )