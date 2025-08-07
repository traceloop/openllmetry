import asyncio
import httpx
import json
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class SSEResultClient:
    """Handles Server-Sent Events streaming for evaluator results - based on guardrails implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.active_streams: Dict[str, asyncio.Task] = {}
    
    async def wait_for_result(
        self,
        execution_id: str,
        stream_url: str, 
        timeout: int = 120,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for evaluation result via SSE streaming.
        Based on guardrails._wait_for_result implementation.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "text/event-stream", 
                    "Cache-Control": "no-cache"
                }
                
                # Construct full stream URL
                api_endpoint = os.getenv("TRACELOOP_BASE_URL", "https://api.traceloop.com")
                full_stream_url = f"{api_endpoint}/v2{stream_url}"
                
                async with client.stream("GET", full_stream_url, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(
                            f"Failed to stream results: {response.status_code}, body: {error_text}"
                        )
                    
                    # Read the complete SSE response
                    response_text = await response.aread()
                    parsed_result = self._parse_sse_result(response_text.decode())
                    
                    # Invoke callback if provided
                    if callback:
                        callback(parsed_result)
                    
                    return parsed_result
                    
        except httpx.ConnectError as e:
            raise Exception(f"Failed to connect to stream URL: {full_stream_url}. Error: {e}")
        except httpx.TimeoutException as e:
            raise Exception(f"Stream request timed out: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error in SSE stream: {e}")
    
    def _parse_sse_result(self, response_text: str) -> Dict[str, Any]:
        """Parse SSE response text into result data"""
        try:
            # Parse the JSON response (following guardrails pattern)
            response_data = json.loads(response_text)
            return response_data
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse SSE result as JSON: {e}")
    
    def start_async_stream(
        self, 
        execution_id: str,
        stream_url: str,
        callback: Callable[[Dict[str, Any]], None],
        timeout: int = 120
    ) -> None:
        """Start streaming results asynchronously with callback"""
        task = asyncio.create_task(
            self.wait_for_result(execution_id, stream_url, timeout, callback)
        )
        self.active_streams[execution_id] = task
    
    def stop_stream(self, execution_id: str) -> None:
        """Stop streaming for a specific execution"""
        if execution_id in self.active_streams:
            self.active_streams[execution_id].cancel()
            self.active_streams.pop(execution_id)