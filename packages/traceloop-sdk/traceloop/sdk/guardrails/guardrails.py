import json
from typing import Dict, Any, Optional, Callable
import aiohttp
from aiohttp.client import ClientSession
from traceloop.sdk.client.http import HTTPClient
from dataclasses import asdict
from .types import ExecuteEvaluatorRequest, InputExtractor, EvaluatorResponse, StreamEventData, StreamEvent
import asyncio
import json
from typing import Any, Dict, List, Union
import time

import httpx

class Guardrails:        
    def __init__(self, http: HTTPClient, app_name: str, api_key: str):
        self._http = http
        self._app_name = app_name
        self._flow = "guardrails"
        self._api_key = api_key
        # re-use one client per process so we don't waste sockets
        # self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

    async def execute_evaluator(self, slug: str, data: Dict[str, InputExtractor]) -> Dict[str, Any]:
        """Execute evaluator and return accumulated SSE event data."""
        slug = slug.replace(" ", "%20")
        url = f"projects/default/evaluators/slug/{slug}/execute"
        
        try:
            # Make POST request to evaluator endpoint
            response = await self._make_post_request(url, data)

            print("Response: ", response)
            
            if response:
                # Handle SSE streaming
                return await self._wait_for_result(stream_url=response["stream_url"], execution_id=response["execution_id"], timeout=120, on_progress=None)
            else:
                # Handle direct response
                return response or {}
                
        except Exception as e:
            # Log error and return empty data
            print(f"Error executing evaluator {slug}: {str(e)}")
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
        on_progress: Optional[Callable[[StreamEventData], None]]
    ) -> Dict[str, Any]:
        """Wait for the evaluation result via server-sent events."""

        print("Waiting for result with execution_id: ", execution_id, "and stream_url: ", stream_url)

        start_time = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"Authorization": f"Bearer {self._api_key}"}

            async with client.stream("GET", stream_url, headers=headers) as response:
                print("Response: ", response)
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

    def _parse_result(self, result: Dict[str, Any]) -> EvaluatorResponse:
        """Parse the raw result into an EvaluatorResponse object."""

        return EvaluatorResponse(
            result=result.get("result"),
            score=result.get("score"),
            reason=result.get("reason"),
            metadata=result.get("metadata", {}),
            raw_result=result
        )

    async def _handle_sse_stream(
        self,
        stream_url: str,
        *,
        expected_event_types: Union[None, List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Open an SSE stream at ``stream_url`` and aggregate JSON payloads
        until a "[DONE]" sentinel (or the connection closes).

        Parameters
        ----------
        stream_url : str
            Absolute *or* relative URL where the SSE stream lives.
        expected_event_types : list[str] | None
            Optional white-list of SSE `event:` names to store; if None,
            everything is kept.

        Returns
        -------
        Dict[str, Any]
            A dictionary keyed by event name (or `"message"` if none was
            supplied) whose values are the last JSON object received for that
            event.  Adjust as you need – for many use-cases you might want a
            list per key instead.
        """
        # Accumulate the server's answers here
        aggregated: Dict[str, Any] = {}

        # Ensure we're speaking SSE
        headers = {"Accept": "text/event-stream"}

        print("Stream URL: ", stream_url)

        # A relative URL is made absolute against the client's base_url
        async with self._http.stream("GET", stream_url, headers=headers) as resp:
            resp.raise_for_status()

            print("Headers: ", resp.headers)

            # httpx gives us an async iterator over *decoded* lines
            async for raw_line in resp.aiter_lines():
                print("Raw line: ", raw_line)
                if not raw_line:          # keep-alives are blank
                    continue

                # ------------------------------------------------------------------
                # Simple, dependency-free parser for the 3 fields we care about:
                #   event: <name>
                #   data:  <anything>
                #   id:    <ignored>
                # ------------------------------------------------------------------
                if raw_line.startswith("event:"):
                    event_name = raw_line.split("event:", 1)[1].strip()
                    # next line *should* be a data line
                    data_line = await resp.aiter_lines().__anext__()
                    if not data_line.startswith("data:"):
                        # Malformed stream; bail or ignore
                        continue
                    payload = data_line.split("data:", 1)[1].strip()

                elif raw_line.startswith("data:"):
                    # event field omitted ⇒ default event name is "message"
                    event_name = "message"
                    payload = raw_line.split("data:", 1)[1].strip()

                else:
                    # ignore comments (lines starting with ':') and id/other fields
                    continue

                # Stop condition
                if payload == "[DONE]":
                    break

                # Filter if requested
                if expected_event_types and event_name not in expected_event_types:
                    continue

                # Decode & stash
                try:
                    aggregated[event_name] = json.loads(payload)
                except json.JSONDecodeError:
                    # Non-JSON payloads are kept as raw strings
                    aggregated[event_name] = payload
        
        print("Aggregated: ", aggregated)
        return aggregated
