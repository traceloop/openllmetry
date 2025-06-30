import json
from typing import Dict, Any, Optional
import aiohttp
from aiohttp.client import ClientSession
from traceloop.sdk.client.http import HTTPClient
from dataclasses import asdict
from .types import ExecuteEvaluatorRequest, InputExtractor

class Guardrails:        
    def __init__(self, http: HTTPClient, app_name: str):
        self._http = http
        self._app_name = app_name
        self._flow = "guardrails"

    async def execute_evaluator(self, slug: str, data: Dict[str, InputExtractor]) -> Dict[str, Any]:
        """Execute evaluator and return accumulated SSE event data."""
        slug = slug.replace(" ", "%20")
        url = f"projects/default/evaluators/slug/{slug}/execute"
        
        try:
            # Make POST request to evaluator endpoint
            response = await self._make_post_request(url, data)

            print("Response: ", response)
            
            if response and "stream_url" in response:
                # Handle SSE streaming
                return await self._handle_sse_stream(response["stream_url"])
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

            print("Response: ", response)
            return response
        except Exception as e:
            print(f"Error making POST request to {url}: {str(e)}")
            return None
    
    async def _handle_sse_stream(self, stream_url: str) -> Dict[str, Any]:
        """Handle Server-Sent Events stream and accumulate data."""
        accumulated_data = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream_url) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                event_data = json.loads(data_str)
                                # Merge event data into accumulated data
                                self._merge_event_data(accumulated_data, event_data)
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                                
        except Exception as e:
            print(f"Error handling SSE stream: {str(e)}")
        
        return accumulated_data
    
    def _merge_event_data(self, accumulated: Dict[str, Any], event: Dict[str, Any]):
        """Merge event data into accumulated data."""
        for key, value in event.items():
            if key in accumulated:
                # If key exists, merge or append based on type
                if isinstance(accumulated[key], list) and isinstance(value, list):
                    accumulated[key].extend(value)
                elif isinstance(accumulated[key], dict) and isinstance(value, dict):
                    accumulated[key].update(value)
                else:
                    # Replace with new value
                    accumulated[key] = value
            else:
                # New key, add directly
                accumulated[key] = value 