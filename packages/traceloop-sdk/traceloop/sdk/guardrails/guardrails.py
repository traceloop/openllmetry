from typing import Dict, Any, Optional
from traceloop.sdk.client.http import HTTPClient
from .types import ExecuteEvaluatorRequest, InputExtractor, OutputSchema
import os
import httpx
import json


class Guardrails:
    def __init__(self, http: HTTPClient, app_name: str, api_key: str):
        self._http = http
        self._app_name = app_name
        self._flow = "guardrails"
        self._api_key = api_key

    async def execute_evaluator(self, slug: str, data: Dict[str, InputExtractor]) -> Dict[str, Any]:
        """Execute evaluator and return accumulated SSE event data."""
        try:
            response = await self._post_request(slug, data)

            if response:
                # Handle SSE streaming
                response_from_stream = await self._wait_for_result(
                    stream_url=response["stream_url"],
                    execution_id=response["execution_id"],
                    timeout=120
                )

                return response_from_stream
            else:
                # Handle direct response
                return response or {}

        except Exception as e:
            # Log error and return empty data
            print(f"Error executing evaluator {slug}. Error: {str(e)}")
            return {}

    async def _post_request(self, slug: str, data: Dict[str, InputExtractor]) -> Optional[Dict[str, Any]]:
        """Make POST request using the HTTP client."""
        try:
            url = f"evaluators/slug/{slug}/execute"

            request_body = ExecuteEvaluatorRequest(input_schema_mapping=data)
            body = request_body.model_dump()

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
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache"
                }
                api_endpoint = os.getenv("TRACELOOP_BASE_URL")
                stream_url = f"{api_endpoint}/v2{stream_url}"

                async with client.stream("GET", stream_url, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(
                            f"Failed to stream results: {response.status_code}, body: {error_text}"
                        )

                    response_text = await response.aread()
                    parsed_response = self._parse_result(response_text)

                    return parsed_response

        except httpx.ConnectError as e:
            print(f"Connection error: {e}")
            raise Exception(f"Failed to connect to stream URL: {stream_url}. Error: {e}")
        except httpx.TimeoutException as e:
            print(f"Timeout error: {e}")
            raise Exception(f"Stream request timed out: {e}")
        except Exception as e:
            print(f"Unexpected error in _wait_for_result: {e}")
            raise

    def _parse_result(self, response_text: str) -> OutputSchema:
        """Parse the response text into an EvaluatorResponse object using Pydantic."""
        try:
            response_data = json.loads(response_text)

            inner_result = response_data.get("result", {}).get("result", {})
            evaluator_response = OutputSchema.model_validate(inner_result)

            return evaluator_response

        except Exception as e:
            print(f"Error parsing result: {e}")
            raise
