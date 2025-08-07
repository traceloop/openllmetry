# Evaluator Execution Feature Planning

## Overview  
The evaluator execution feature will add evaluator execution capabilities to the traceloop-sdk, following a similar structure to the existing datasets module. This feature will enable users to execute evaluators against data through the API service, with results delivered via Server-Sent Events (SSE) streaming.

## Reference Implementation
This feature will follow the same SSE streaming pattern as the existing guardrails implementation in `traceloop/sdk/guardrails/guardrails.py:139-177`, which successfully handles evaluator execution with streaming responses.

## Architecture Analysis - Datasets Pattern

Based on the existing datasets structure at `packages/traceloop-sdk/traceloop/sdk/datasets/`, the following pattern is established:

### File Structure
```
traceloop/sdk/datasets/
├── __init__.py          # Exports main classes
├── dataset.py           # Main Dataset class with API methods
├── model.py             # Pydantic models and data structures
├── column.py            # Column entity class
└── row.py               # Row entity class
```

### Key Components
1. **Main Entity Class** (`Dataset`): Inherits from `DatasetBaseModel`, contains business logic
2. **HTTP Client Integration**: Uses `traceloop.sdk.client.http.HTTPClient` for API communication
3. **Pydantic Models**: Strong typing with validation in `model.py`
4. **Environment-based Configuration**: Uses `TRACELOOP_API_KEY` and `TRACELOOP_BASE_URL`
5. **Class Methods**: Static methods for operations that don't require instance state

## Proposed Experiments Structure

### File Structure
```
traceloop/sdk/experiments/
├── __init__.py          # Exports main classes
├── experiment.py        # Main Experiment class
├── model.py             # Pydantic models for experiments
├── stream_client.py     # Stream URL client for result polling/streaming
└── result_handler.py    # Async result processing and callbacks
```

### Core Classes

#### 1. Evaluator Class (`evaluator.py`)
```python
class Evaluator(DatasetBaseModel):
    """Evaluator class for executing evaluators with SSE streaming"""
    
    @classmethod
    def run(cls, 
            evaluator_slug: str,
            input_schema_mapping: Dict[str, str], 
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            wait_for_result: bool = False,
            timeout: int = 300) -> Union[ExecuteEvaluatorResponse, Dict[str, Any]]:
        """Execute evaluator with input schema mapping"""
        
    @classmethod 
    async def run_async(cls, 
                       evaluator_slug: str,
                       input_schema_mapping: Dict[str, str],
                       callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                       timeout: int = 120) -> ExecuteEvaluatorResponse:
        """Async version of run method"""
```

#### 2. Model Definitions (`model.py`)
Based on the actual API specification:

```python
# Input schema mapping models
class InputExtractor(BaseModel):
    source: str = Field(..., description="Source field name")

class InputSchemaMapping(BaseModel):
    """Map of field names to input extractors"""
    __root__: Dict[str, InputExtractor]

class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: InputSchemaMapping = Field(..., alias="input_schema_mapping")

class ExecuteEvaluatorResponse(BaseModel):
    """Response from execute API matching actual structure"""
    execution_id: str
    stream_url: str

class EvaluatorResult(BaseModel):
    """Final result from stream"""
    execution_id: str
    status: str  # completed, failed, running
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None

class StreamEvent(BaseModel):
    """Individual event from stream"""
    event_type: str  # progress, result, error
    data: Dict[str, Any]
    timestamp: datetime
```

#### 3. SSE Stream Client (`stream_client.py`)
Based on the existing guardrails SSE implementation:

```python
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
```

#### 4. Result Handler (`result_handler.py`)
```python
class ResultHandler:
    """Manages async result processing and callbacks"""
    
    @staticmethod
    def create_callback(experiment: 'Experiment', on_complete: Optional[Callable] = None) -> Callable:
        """Create callback function for result processing"""
        def callback(result: EvaluatorResult):
            # Process result
            if on_complete:
                on_complete(result)
        return callback
```

### Key Methods

#### Main Run Function (Stream-based Pattern)
```python
def run(self, 
        input_schema_mapping: Dict[str, str], 
        callback: Optional[Callable[[EvaluatorResult], None]] = None,
        wait_for_result: bool = False,
        timeout: int = 300) -> Union[ExecuteEvaluatorResponse, EvaluatorResult]:
    """
    Execute evaluator with input schema mapping
    
    Args:
        input_schema_mapping: Dict mapping field names to source fields
        callback: Optional callback function for async result handling
        wait_for_result: If True, blocks until result is received via stream
        timeout: Timeout in seconds for synchronous execution
    
    Returns:
        ExecuteEvaluatorResponse (immediate) or EvaluatorResult (if waiting)
    """
    # Convert dict to proper model format
    mapping = InputSchemaMapping(__root__={
        field: InputExtractor(source=source) 
        for field, source in input_schema_mapping.items()
    })
    
    request = ExecuteEvaluatorRequest(input_schema_mapping=mapping)
    
    # Make API call to trigger evaluator
    result = self._http.post(
        f"v2/evaluators/slug/{self.evaluator_slug}/execute",
        request.model_dump(by_alias=True)
    )
    
    if result is None:
        raise Exception(f"Failed to execute evaluator {self.evaluator_slug}")
    
    response = ExecuteEvaluatorResponse(**result)
    
    # Set up SSE client for result delivery
    if not self._sse_client:
        api_key = os.environ.get("TRACELOOP_API_KEY", "")
        self._sse_client = SSEResultClient(api_key)
    
    # Handle SSE streaming results
    if callback or wait_for_result:
        if wait_for_result:
            # Synchronous execution - wait for result
            return self._wait_for_sse_result(response.stream_url, response.execution_id, timeout)
        else:
            # Asynchronous execution - start stream with callback
            self._sse_client.start_async_stream(response.execution_id, response.stream_url, callback, timeout)
    
    return response

def _wait_for_sse_result(self, stream_url: str, execution_id: str, timeout: int) -> Dict[str, Any]:
    """Wait synchronously for result from SSE stream - based on guardrails pattern"""
    import asyncio
    import threading
    
    result_container = {"result": None, "received": False, "error": None}
    
    # Run async SSE stream in thread
    def run_sse_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._sse_client.wait_for_result(execution_id, stream_url, timeout)
            )
            result_container["result"] = result
            result_container["received"] = True
        except Exception as e:
            result_container["error"] = str(e)
            result_container["received"] = True
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_sse_stream)
    thread.start()
    thread.join(timeout + 5)  # Add small buffer to thread timeout
    
    if not result_container["received"]:
        self._sse_client.stop_stream(execution_id)
        raise TimeoutError(f"Evaluator execution {execution_id} timed out after {timeout}s")
    
    if result_container["error"]:
        raise Exception(f"SSE stream error: {result_container['error']}")
    
    return result_container["result"]
```

#### Async Support
```python
async def run_async(self, 
                   input_schema_mapping: Dict[str, str],
                   callback: Optional[Callable[[EvaluatorResult], None]] = None) -> ExecuteEvaluatorResponse:
    """Async version of run method"""
    # Similar implementation using async HTTP client
```

## API Integration Details

### Execution Endpoint
- **URL**: `/v2/evaluators/slug/:slug/execute`
- **Method**: `POST`
- **Path Parameter**: `slug` - The evaluator slug to execute

### Request Payload
```json
{
  "input_schema_mapping": {
    "field_name": {
      "source": "source_field_name"
    }
  }
}
```

### Response Structure
```json
{
  "execution_id": "12345-67890-abcdef",
  "stream_url": "https://api.traceloop.com/v2/executions/events/12345-67890-abcdef"
}
```

### SSE-based Result Delivery
The API service provides a streaming URL for Server-Sent Events result delivery:

1. **API Response**: Immediate response with `execution_id` and `stream_url`
2. **SSE Connection**: SDK connects to `stream_url` with appropriate headers (`Accept: text/event-stream`)
3. **Event Stream**: Results delivered as Server-Sent Events with JSON payload
4. **Result Parsing**: SDK parses SSE response and extracts evaluation results
5. **Callback Execution**: SDK processes results and executes registered callbacks

**SSE Headers Used** (following guardrails pattern):
- `Authorization: Bearer {api_key}`
- `Accept: text/event-stream`
- `Cache-Control: no-cache`

## Example Usage

### Synchronous Execution
```python
from traceloop.sdk.experiments import Evaluator

# Execute evaluator and wait for result via SSE stream
result = Evaluator.run(
    evaluator_slug="accuracy-evaluator",
    input_schema_mapping={
        "question": "user_query",
        "answer": "model_response", 
        "expected": "ground_truth"
    },
    wait_for_result=True,
    timeout=300
)

print(f"Final result: {result}")

# Or using the convenience function
from traceloop.sdk.experiments import run_evaluator

result = run_evaluator(
    evaluator_slug="accuracy-evaluator",
    input_schema_mapping={
        "question": "user_query",
        "answer": "model_response"
    },
    wait_for_result=True
)
```

### Asynchronous Execution with Callback
```python
def handle_result(result: Dict[str, Any]):
    print(f"SSE result received: {result}")
    # Process the parsed SSE result as needed

# Execute with callback - SSE stream will be processed in background
response = Evaluator.run(
    evaluator_slug="accuracy-evaluator",
    input_schema_mapping={
        "question": "user_query",
        "answer": "model_response"
    },
    callback=handle_result,
    timeout=120  # Optional timeout for SSE connection
)

print(f"Execution initiated: {response.execution_id}")
print(f"Stream URL: {response.stream_url}")
# Continue with other work while result is streamed asynchronously via SSE
```

### Async/Await Pattern
```python
import asyncio

async def main():
    # Direct async evaluator execution
    response = await Evaluator.run_async(
        evaluator_slug="accuracy-evaluator",
        input_schema_mapping={"question": "user_query"},
        callback=lambda result: print(f"Result: {result}"),
        timeout=120
    )
    print(f"Execution started: {response.execution_id}")
    
    # Or using the convenience function
    from traceloop.sdk.experiments import run_evaluator_async
    
    response = await run_evaluator_async(
        evaluator_slug="accuracy-evaluator",
        input_schema_mapping={"question": "user_query"}
    )

asyncio.run(main())
```

## Implementation Steps

1. **Create experiments directory structure**
   - Set up all required files
   - Configure `__init__.py` with proper exports

2. **Implement SSE client infrastructure**
   - Server-Sent Events connection management based on guardrails pattern
   - Result delivery handling with proper headers and parsing
   - Error handling for connection and timeout issues

3. **Create model definitions**
   - Request/response models matching API specification
   - Result handling models for SSE data
   - Input schema mapping validation

4. **Implement Experiment class**
   - HTTP client integration for triggering execution (similar to datasets)
   - SSE client integration for result delivery (following guardrails)
   - Both sync and async execution patterns

5. **Add result handling system**
   - Callback registration and execution
   - Timeout handling for synchronous SSE calls
   - Error handling for SSE connection issues

6. **Testing**
   - Mock SSE connections for unit tests using VCR.py pattern
   - Integration tests with actual SSE streaming
   - Timeout and error scenario testing
   - Follow existing test patterns from guardrails implementation

## Configuration

Standard configuration for stream-based result delivery:

### Environment Variables
- `TRACELOOP_API_KEY`: Required for API authentication
- `TRACELOOP_BASE_URL`: Custom endpoint URL (default: https://api.traceloop.com)
- `TRACELOOP_STREAM_TIMEOUT`: Stream connection timeout (default: 300s)

## Dependencies

Dependencies for SSE functionality (already available in guardrails):
- `httpx`: Async HTTP client for SSE streaming support (already in use)
- `asyncio`: Built-in async support (Python 3.7+)
- `json`: JSON parsing for SSE response data (built-in)
- `os`: Environment variable access (built-in)

## Future Enhancements

- **Stream Reconnection**: Handle connection failures with automatic retry
- **Result Persistence**: Store results in local database/cache  
- **Progress Events**: Support intermediate progress updates via stream
- **Monitoring**: Add telemetry for execution and stream connection metrics
- **Batch Execution**: Support for running multiple evaluations with shared streams
- **WebSocket Alternative**: Optional WebSocket-based streaming for real-time updates