from setuptools import setup, find_packages

setup(
    name='traceloop-sdk',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # Core minimum dependencies required for all installs
        'opentelemetry-sdk>=0.52b1',
        # Add other core required packages here
    ],
    extras_require={
        'full': [
            'opentelemetry-instrumentation-alephalpha==0.39.0',
            'opentelemetry-instrumentation-anthropic==0.39.0',
            'opentelemetry-instrumentation-bedrock==0.39.0',
            'opentelemetry-instrumentation-chromadb==0.39.0',
            'opentelemetry-instrumentation-cohere==0.39.0',
            'opentelemetry-instrumentation-crewai==0.39.0',
            'opentelemetry-instrumentation-google-generativeai==0.39.0',
            'opentelemetry-instrumentation-groq==0.39.0',
            'opentelemetry-instrumentation-haystack==0.39.0',
            'opentelemetry-instrumentation-lancedb==0.39.0',
            'opentelemetry-instrumentation-langchain==0.39.0',
            'opentelemetry-instrumentation-llamaindex==0.39.0',
            'opentelemetry-instrumentation-logging==0.52b1',
            'opentelemetry-instrumentation-marqo==0.39.0',
            'opentelemetry-instrumentation-milvus==0.39.0',
            'opentelemetry-instrumentation-mistralai==0.39.0',
            'opentelemetry-instrumentation-ollama==0.39.0',
            'opentelemetry-instrumentation-openai==0.39.0',
            'opentelemetry-instrumentation-pinecone==0.39.0',
            'opentelemetry-instrumentation-qdrant==0.39.0',
            'opentelemetry-instrumentation-replicate==0.39.0',
            'opentelemetry-instrumentation-requests==0.52b1',
            'opentelemetry-instrumentation-sagemaker==0.39.0',
            'opentelemetry-instrumentation-sqlalchemy==0.52b1',
            'opentelemetry-instrumentation-threading==0.52b1',
            'opentelemetry-instrumentation-together==0.39.0',
            'opentelemetry-instrumentation-transformers==0.39.0',
            'opentelemetry-instrumentation-urllib3==0.52b1',
            'opentelemetry-instrumentation-vertexai==0.39.0',
            'opentelemetry-instrumentation-watsonx==0.39.0',
            'opentelemetry-instrumentation-weaviate==0.39.0',
            'opentelemetry-semantic-conventions-ai==0.4.3',
            'opentelemetry-util-http==0.52b1',
        ],
        'minimal': [
            'opentelemetry-instrumentation-requests==0.52b1',
            'opentelemetry-instrumentation-urllib3==0.52b1',
        ],
        'openai': [
            'opentelemetry-instrumentation-openai==0.39.0',
        ],
        'langchain': [
            'opentelemetry-instrumentation-langchain==0.39.0',
        ],
        # Add other targeted optional groups here
    }
)

