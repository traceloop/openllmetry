import httpx
import json
import os

from .model import ExecutionResponse


class SSEClient:
    """Handles Server-Sent Events streaming"""
    
    def __init__(self):
        self._api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
        self._api_key = os.environ.get("TRACELOOP_API_KEY", "")
    
    async def wait_for_result(
        self,
        execution_id: str,
        stream_url: str, 
        timeout_in_sec: int = 120,
    ) -> ExecutionResponse:
        """
        Wait for execution result via SSE streaming.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "text/event-stream", 
                "Cache-Control": "no-cache"
            }
            
            full_stream_url = f"{self._api_endpoint}/v2{stream_url}"
            
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", full_stream_url, headers=headers, timeout=httpx.Timeout(timeout_in_sec)) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(
                            f"Failed to stream results: {response.status_code}, body: {error_text}"
                        )
                    
                    response_text = await response.aread()
                    parsed_result = self._parse_sse_result(response_text.decode())
                
                if parsed_result.execution_id != execution_id:
                    raise Exception(f"Execution ID mismatch: {parsed_result.execution_id} != {execution_id}")
                
                return parsed_result
                    
        except httpx.ConnectError as e:
            raise Exception(f"Failed to connect to stream URL: {full_stream_url}. Error: {e}")
        except httpx.TimeoutException as e:
            raise Exception(f"Stream request timed out: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error in SSE stream: {e}")
    
    def _parse_sse_result(self, response_text: str) -> ExecutionResponse:
        """Parse SSE response text into ExecutionResponse"""
        try:
            response_data = json.loads(response_text)
            return ExecutionResponse(**response_data)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse SSE result as JSON: {e}")
        except Exception as e:
            raise Exception(f"Failed to parse response into ExecutionResponse: {e}")
    
