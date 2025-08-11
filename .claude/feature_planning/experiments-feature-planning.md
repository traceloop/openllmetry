# Experiments Feature Planning

## Overview
The experiments feature will add evaluator execution capabilities to the traceloop-sdk, following a similar structure to the existing datasets module. This feature will enable users to execute evaluators against data through the API service.

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
```

### Core Classes

#### 1. Experiment Class (`experiment.py`)
```python
class Experiment(DatasetBaseModel):
    """Main experiment class for evaluator execution"""
    
    # Core attributes
    
    
    # HTTP client for API communication
    _http: HTTPClient = PrivateAttr(default=None)
```

#### 2. Model Definitions (`model.py`)
Based on the API payload specification:

```python
# Input schema mapping models
class InputExtractor(BaseModel):
    source: str = Field(..., description="Source field name")

class InputSchema(BaseModel):
    """Map of field names to input extractors"""
    __root__: Dict[str, InputExtractor]

class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: InputSchemaMapping = Field(..., alias="input_schema_mapping")

class ExecuteEvaluatorResponse(BaseModel):
    # Response structure to be defined based on API response
    execution_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
```


#### Main Run Function
```python
def run(self, input_schema_mapping: Dict[str, str]) -> ExecuteEvaluatorResponse:
    """
    Execute evaluator with input schema mapping
    
    Args:
        input_schema_mapping: Dict mapping field names to source fields
                              e.g., {"question": "user_input", "answer": "model_output"}
    
    Returns:
        ExecuteEvaluatorResponse with execution results
    """
    # Convert dict to proper model format
    mapping = InputSchemaMapping(__root__={
        field: InputExtractor(source=source) 
        for field, source in input_schema_mapping.items()
    })
    
    request = ExecuteEvaluatorRequest(input_schema_mapping=mapping)
    
    # Make API call to /v2/evaluators/slug/{evaluator_slug}/execute
    result = self._http.post(
        f"v2/evaluators/slug/{self.evaluator_slug}/execute",
        request.model_dump(by_alias=True)
    )
    
    if result is None:
        raise Exception(f"Failed to execute evaluator {self.evaluator_slug}")
    
    return ExecuteEvaluatorResponse(**result)
```

## API Integration Details

### Endpoint
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

### Example Usage
```python
from traceloop.sdk.experiments import Experiment

# Create experiment instance
experiment = Experiment(
    slug="my-experiment",
    evaluator_slug="accuracy-evaluator"
)

# Execute evaluator
result = experiment.run({
    "question": "user_query",
    "answer": "model_response", 
    "expected": "ground_truth"
})

print(f"Execution ID: {result.execution_id}")
print(f"Status: {result.status}")
```

## Implementation Steps

1. **Create experiments directory structure** 
   - Mirror datasets structure
   - Set up `__init__.py` with proper exports

2. **Implement model definitions**
   - Define Pydantic models for request/response
   - Handle input schema mapping validation

3. **Create Experiment class**
   - Inherit from DatasetBaseModel  
   - Implement HTTP client integration
   - Add run() method for evaluator execution

4. **Add class methods**
   - CRUD operations following datasets pattern
   - HTTP client management

5. **Update package exports**
   - Add experiments to main SDK `__init__.py`
   - Ensure proper module importing

6. **Testing**
   - Unit tests for model validation
   - Integration tests with VCR cassettes
   - Error handling tests

## Configuration

The experiments feature will reuse the existing configuration pattern:
- `TRACELOOP_API_KEY`: Required for API authentication
- `TRACELOOP_BASE_URL`: Custom endpoint URL (defaults to https://api.traceloop.com)

## Future Enhancements

- **Async Support**: Add async variants of API methods
- **Batch Execution**: Support for running multiple evaluations
- **Result Storage**: Integration with datasets for storing results
- **Monitoring**: Add telemetry and metrics for experiment execution