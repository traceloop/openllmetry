# Medical Doctor Q&A Example

This example demonstrates a medical doctor chatbot that answers patient questions using an LLM, instrumented with OpenTelemetry for observability.

## Features

- **Single Mode**: Ask one question at a time interactively
- **Batch Mode**: Process 20 predefined medical questions automatically
- **OpenTelemetry Integration**: Full tracing and observability
- **Medical Expert Persona**: Responses as a knowledgeable, compassionate doctor
- **Safety Disclaimers**: Appropriate medical disclaimers and recommendations

## Setup

1. Install dependencies (from the sample-app directory):
```bash
poetry install
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. (Optional) Set Traceloop API key for dashboard:
```bash
export TRACELOOP_API_KEY="your-traceloop-key"
```

## Usage

### Single Question Mode

Interactive mode - ask one question:
```bash
poetry run python sample_app/medical_qa_example.py --mode single
```

Ask a specific question:
```bash
poetry run python sample_app/medical_qa_example.py --mode single --question "What are the symptoms of flu?"
```

Save the result:
```bash
poetry run python sample_app/medical_qa_example.py --mode single --question "How much sleep do I need?" --save
```

### Batch Mode

Process all 20 sample questions:
```bash
poetry run python sample_app/medical_qa_example.py --mode batch
```

Save results to JSON:
```bash
poetry run python sample_app/medical_qa_example.py --mode batch --save
```

## Sample Questions

The script includes 20 common medical questions covering:
- Symptoms and conditions (flu, diabetes, heart attack signs)
- Preventive care (vaccines, physical exams, immune system)
- Emergency situations (chest pain, fever, concussion)
- General health (sleep, hydration, stress management)

## OpenTelemetry Tracing

The script uses Traceloop decorators to provide comprehensive instrumentation:

- `@workflow` decorators trace the overall consultation workflows (single and batch modes)
- `@task` decorators trace individual operations (medical response generation, result saving)
- Automatic instrumentation of OpenAI API calls with request/response tracing, token usage metrics, error tracking, and performance monitoring

### Key Instrumented Functions:
- `@workflow("single_medical_consultation")` - Single question consultation flow
- `@workflow("batch_medical_consultation")` - Batch processing flow
- `@task("medical_response_generation")` - Individual question processing
- `@task("save_medical_results")` - Result saving operation

Use the console span exporter for debugging:
```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

Traceloop.init(exporter=ConsoleSpanExporter())
```

## Safety Notes

⚠️ **Important**: This is a demonstration example. The responses:
- Should not replace professional medical advice
- Include appropriate disclaimers
- Recommend consulting healthcare providers for serious symptoms
- Are for educational/testing purposes only