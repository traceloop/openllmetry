# Guardrails Sample Applications

This directory contains sample applications demonstrating the Traceloop Guardrails feature, which provides content validation and safety checks for AI applications.

## Overview

Guardrails in Traceloop help you:

- ✅ **Validate Input**: Check user inputs for safety, appropriateness, and policy compliance
- ✅ **Validate Output**: Ensure AI-generated content meets quality and safety standards
- ✅ **Configurable Rules**: Set custom thresholds and validation parameters
- ✅ **Multiple Actions**: Handle validation results with pass/block/retry actions
- ✅ **Async & Sync**: Support for both synchronous and asynchronous workflows
- ✅ **Decorator Support**: Easy integration with existing functions using decorators

## Sample Applications

### 1. Simple Guardrails Example (`guardrails_simple_example.py`)

A straightforward example demonstrating:

- Basic input validation before text generation
- Synchronous guardrails validation
- Error handling and fallback strategies
- Different validation outcomes (pass/block/retry)

**Key Features:**

- Simple synchronous workflow
- Clear validation result handling
- Multiple test cases
- Environment variable checks

### 2. Comprehensive Guardrails Example (`guardrails_example.py`)

An advanced example showcasing:

- Manual input and output validation
- Decorator-based validation
- Asynchronous workflows
- Custom progress callbacks
- Complex configuration options

**Key Features:**

- Multiple validation patterns
- Async/await support
- Traceloop workflow integration
- Custom callbacks and configuration

## Prerequisites

Before running the samples, ensure you have:

```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export TRACELOOP_API_KEY="your-traceloop-api-key"  # Recommended for full functionality
```

## Installation

The samples use dependencies already configured in the sample-app project:

```bash
cd openllmetry/packages/sample-app
poetry install
```

## Running the Examples

### Simple Example

```bash
# Run the basic guardrails demo
python sample_app/guardrails_simple_example.py
```

### Comprehensive Example

```bash
# Run the advanced guardrails demo
python sample_app/guardrails_example.py
```

## Understanding Guardrails

### Core Concepts

1. **Guardrail Actions**:

   - `PASS`: Content is safe and appropriate
   - `BLOCK`: Content violates policies and should be rejected
   - `RETRY`: Content needs modification or re-evaluation

2. **Input Data Structure**:

   ```python
   GuardrailInputData(
       content="text to validate",
       context={"user_id": "123", "session": "abc"},
       metadata={"timestamp": "2024-01-01T10:00:00Z"}
   )
   ```

3. **Configuration Options**:
   ```python
   GuardrailConfig(
       thresholds={"safety": 0.8, "toxicity": 0.7},
       parameters={"check_categories": ["safety", "hate", "violence"]},
       settings={"strict_mode": True}
   )
   ```

### Usage Patterns

#### 1. Manual Validation

```python
# Validate input manually
result = await guardrails.validate_input(
    evaluator_slug="content-safety",
    input_data=input_data,
    config=config
)

if result.pass_through:
    # Proceed with content generation
    pass
elif result.blocked:
    # Handle blocked content
    pass
```

#### 2. Decorator-Based Validation

```python
@guardrails_decorator.validate_input("toxicity-checker")
def generate_content(prompt: str) -> str:
    return llm.generate(prompt)
```

#### 3. Output Validation

```python
# Validate generated output
output_result = await guardrails.validate_output(
    evaluator_slug="output-quality-checker",
    output_data=output_data
)
```

## Evaluator Slugs

The examples use various evaluator slugs for different validation types:

- `content-safety`: General content safety validation
- `toxicity-checker`: Toxicity and harmful content detection
- `output-quality-checker`: Output quality and relevance validation
- `content-appropriateness`: General appropriateness checking
- `ai-content-validator`: AI-specific content validation

> **Note**: Actual evaluator slugs depend on your Traceloop configuration and available evaluators.

## Error Handling

The samples demonstrate different error handling strategies:

1. **Conservative Approach**: Block requests when validation fails
2. **Graceful Degradation**: Allow requests to proceed with warnings
3. **Retry Logic**: Attempt validation multiple times with backoff

## Configuration Examples

### Basic Configuration

```python
config = GuardrailConfig(
    thresholds={"safety": 0.8},
    parameters={"check_categories": ["safety"]}
)
```

### Advanced Configuration

```python
config = GuardrailConfig(
    thresholds={
        "toxicity": 0.7,
        "hate": 0.8,
        "violence": 0.9,
        "appropriateness": 0.75
    },
    parameters={
        "check_categories": ["toxicity", "hate", "violence", "adult"],
        "language": "en",
        "detailed_analysis": True
    },
    settings={
        "strict_mode": True,
        "enable_caching": True,
        "cache_ttl": 3600
    }
)
```

## Best Practices

1. **Environment Setup**: Always check for required environment variables
2. **Error Handling**: Implement robust error handling for validation failures
3. **Timeouts**: Set appropriate timeouts for validation requests
4. **Logging**: Use proper logging for validation results and errors
5. **Testing**: Test with various input types and edge cases
6. **Performance**: Consider caching and async operations for high-throughput applications

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure both OpenAI and Traceloop API keys are set
2. **Network Timeouts**: Increase timeout values for slow networks
3. **Evaluator Not Found**: Verify evaluator slugs match your Traceloop configuration
4. **Import Errors**: Ensure traceloop-sdk is properly installed

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Applications

To integrate guardrails into your existing application:

1. **Initialize Traceloop** with guardrails support
2. **Add validation points** at input and output boundaries
3. **Handle validation results** according to your application's needs
4. **Configure evaluators** based on your content policies
5. **Monitor and adjust** thresholds based on validation results

## Further Resources

- [Traceloop Documentation](https://docs.traceloop.com)
- [Guardrails API Reference](https://docs.traceloop.com/api/guardrails)
- [Best Practices Guide](https://docs.traceloop.com/guides/guardrails-best-practices)
- [Evaluator Configuration](https://docs.traceloop.com/guides/evaluators)
