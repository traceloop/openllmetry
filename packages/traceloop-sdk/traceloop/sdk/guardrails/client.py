import json
import re
import asyncio
from typing import Dict, Any, Optional
import aiohttp
from traceloop.sdk.client.http import HTTPClient

class GuardrailsClient:
    def __init__(self, http_client: HTTPClient):
        self.http_client = http_client
    
    async def execute_evaluator(self, slug: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluator and return accumulated SSE event data."""
        url = f"projects/default/evaluators/{slug}/execute"
        
        try:
            print("NOMI - GuardrailsClient - execute_evaluator - url:", url)
            print("NOMI - GuardrailsClient - execute_evaluator - data:", data)
            # Make POST request to evaluator endpoint
            response = await self._make_post_request(url, data)
            print("NOMI - GuardrailsClient - execute_evaluator - response:", response)
            if response and "stream_url" in response:
                # Handle SSE streaming
                print("NOMI - GuardrailsClient - execute_evaluator - handling SSE streaming")
                return await self._handle_sse_stream(response["stream_url"])
            
            else:
                # Handle direct response
                return response or {}
                
        except Exception as e:
            # Log error and return empty data
            print(f"Error executing evaluator {slug}: {str(e)}")
            return {}
    
    async def _make_post_request(self, url: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make POST request using the HTTP client."""
        try:
            # Convert sync HTTP client to async request
            import requests
            response = requests.post(
                f"{self.http_client.base_url}/v2/{url}",
                json=data,
                headers=self.http_client._headers(),
                timeout=30
            )
            response.raise_for_status()
            return response.json()
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