# sample-app

Collection of sample applications demonstrating OpenLLMetry instrumentation with various AI/ML frameworks and services.

## Examples

### Medical Doctor Q&A (`medical_qa_example.py`)
A medical chatbot that answers patient questions with proper medical disclaimers. Supports both single question and batch processing modes. See `README_medical_qa.md` for detailed usage instructions.

### Other Examples
The `sample_app/` directory contains numerous examples for different AI frameworks:
- OpenAI, Anthropic, Cohere, Gemini integrations
- LangChain, LlamaIndex, Haystack applications
- Vector databases (Chroma, Pinecone, Weaviate)
- Streaming, async, and structured output examples

## Usage
All examples can be run with Poetry:
```bash
poetry run python sample_app/<example_name>.py
```
