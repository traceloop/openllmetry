# Guardrails Decorator

The guardrails decorator allows you to evaluate function inputs and outputs using external evaluators and make runtime decisions based on the calculated scores.

## Overview

The guardrails decorator integrates with Traceloop's evaluator system to:
- Extract data from function arguments and return values
- Send data to evaluators via API
- Process Server-Sent Events (SSE) responses
- Calculate scores using custom functions
- Make the score available within the decorated function for runtime decisions

## Basic Usage

```python
from traceloop.sdk.guardrails import (
    guardrails,
    get_current_score,
    InputExtractor,
)

def calculate_sentiment_score(event_data: dict) -> float:
    """Calculate sentiment score from event data."""
    return event_data.get("sentiment_score", 0.5)

@guardrails(
    evaluator_slug="sentiment-analyzer",
    score_calculator=calculate_sentiment_score,
    input_schema={
        "text": InputExtractor(source="input", key="message"),
        "user_id": InputExtractor(source="input", key="user_id")
    }
)
def process_message(message: str, user_id: str) -> str:
    # Access the score within the function
    score = get_current_score()
    
    if score < 0.3:
        return "REJECTED"
    elif score > 0.7:
        return "APPROVED"
    else:
        return "REVIEW"
```

## Components

### InputExtractor

Defines how to extract data from function arguments or return values:

```python
@dataclass
class InputExtractor:
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply
```

### InputSchemaMapping

A dictionary mapping field names to `InputExtractor` instances:

```python
InputSchemaMapping = Dict[str, InputExtractor]
```

### Score Calculator

A function that takes event data and returns a score:

```python
def calculate_score(event_data: Dict[str, Any]) -> float:
    return event_data.get("score", 0.0)
```

## Data Extraction

### Input Source

Extract data from function arguments:

```python
@guardrails(
    evaluator_slug="analyzer",
    score_calculator=calculate_score,
    input_schema={
        "text": InputExtractor(source="input", key="message"),
        "user": InputExtractor(source="input", key="user_id")
    }
)
def process(message: str, user_id: str):
    pass
```

### Output Source

Extract data from function return value:

```python
@guardrails(
    evaluator_slug="analyzer",
    score_calculator=calculate_score,
    input_schema={
        "result": InputExtractor(source="output", key="content")
    }
)
def generate_content(prompt: str) -> dict:
    return {"content": "Generated text", "metadata": {...}}
```

### Regex Pattern

Use regex to extract specific patterns:

```python
@guardrails(
    evaluator_slug="email-validator",
    score_calculator=calculate_score,
    input_schema={
        "email": InputExtractor(
            source="input",
            key="contact_info",
            use_regex=True,
            regex_pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
    }
)
def process_contact(contact_info: str):
    pass
```

## Accessing Scores

### Within Decorated Function

Use `get_current_score()` to access the calculated score:

```python
@guardrails(...)
def my_function():
    score = get_current_score()
    
    if score < 0.3:
        # Handle low score
        pass
    elif score > 0.8:
        # Handle high score
        pass
```

### Outside Context

When called outside a guardrails context, `get_current_score()` returns `None`:

```python
score = get_current_score()  # Returns None
```

## Async Support

The decorator supports both sync and async functions:

```python
@guardrails(...)
async def async_function():
    score = get_current_score()
    # Process with score
    return result
```

## Tracing Integration

The guardrails decorator integrates with OpenTelemetry tracing:

- Creates spans for guardrails evaluation
- Adds guardrails-specific attributes:
  - `traceloop.guardrails.score`
  - `traceloop.guardrails.evaluator_slug`
  - `traceloop.guardrails.input_schema`
  - `traceloop.guardrails.event_data`

## Error Handling

The decorator handles errors gracefully:

- API connection errors
- SSE parsing errors
- Score calculation errors
- Invalid input schema errors

Errors are logged and the function continues execution with a default score.

## Examples

See `examples/guardrails_example.py` for comprehensive usage examples including:

- Sentiment analysis
- Content moderation
- Fact checking
- Error handling

## API Endpoints

The decorator uses the following API endpoints:

- **POST** `/v2/projects/default/evaluators/{slug}/execute`
- **SSE** Stream from the response URL

## Configuration

The decorator uses the same configuration as the main Traceloop SDK:

- API endpoint from `Traceloop.init()`
- API key from environment or initialization
- Headers and authentication from client configuration 