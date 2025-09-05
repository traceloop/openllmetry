import httpx
import json

from .model import ExecutionResponse


class SSEClient:
    """Handles Server-Sent Events streaming"""

    def __init__(self, shared_client: httpx.AsyncClient):
        self.client = shared_client

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
                "Authorization": f"Bearer {self.client.headers.get('Authorization')}",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            }

            full_stream_url = f"{self.client.base_url}/v2{stream_url}"

            async with self.client.stream(
                "GET",
                full_stream_url,
                headers=headers,
                timeout=httpx.Timeout(timeout_in_sec),
            ) as response:
                parsed_result = await self._handle_sse_response(response)

            if parsed_result.execution_id != execution_id:
                raise Exception(
                    f"Execution ID mismatch: {parsed_result.execution_id} != {execution_id}"
                )

            return parsed_result

        except httpx.ConnectError as e:
            raise Exception(
                f"Failed to connect to stream URL: {full_stream_url}. Error: {e}"
            )
        except httpx.TimeoutException as e:
            raise Exception(f"Stream request timed out: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error in SSE stream: {e}")

    async def _handle_sse_response(self, response) -> ExecutionResponse:
        """Handle SSE response: check status and parse result"""
        if response.status_code != 200:
            error_text = await response.aread()
            raise Exception(
                f"Failed to stream results: {response.status_code}, body: {error_text}"
            )

        response_text = await response.aread()
        return self._parse_sse_result(response_text.decode())

    def _parse_sse_result(self, response_text: str) -> ExecutionResponse:
        """Parse SSE response text into ExecutionResponse"""
        try:
            response_data = json.loads(response_text)
            return ExecutionResponse(**response_data)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse SSE result as JSON: {e}")
        except Exception as e:
            raise Exception(f"Failed to parse response into ExecutionResponse: {e}")
