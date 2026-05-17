name: 🚀 Feature
description: "Submit a proposal for a new feature"
title: "🚀 Feature: "
labels: [feature]
body:
  - type: dropdown
    id: component
    validations:
      required: true
    attributes:
      label: Which component is this feature for?
      description: Which package does this feature request apply to?
      options:
        - "AlephAlpha Instrumentation"
        - "Anthropic Instrumentation"
        - "Bedrock Instrumentation"
        - "Chromadb Instrumentation"
        - "Cohere Instrumentation"
        - "Google Generative AI Instrumentation"
        - "Groq Instrumentation"
        - "Haystack Instrumentation"
        - "LanceDB Instrumentation"
        - "Langchain Instrumentation"
        - "LlamaIndex Instrumentation"
        - "Marqo Instrumentation"
        - "Milvus Instrumentation"
        - "Mistral Instrumentation"
        - "Ollama Instrumentation"
        - "OpenAI Instrumentation"
        - "Pinecone Instrumentation"
        - "Qdrant Instrumentation"
        - "Replicate Instrumentation"
        - "SageMaker Instrumentation"
        - "Together Instrumentation"
        - "Transformers Instrumentation"
        - "VertexAI Instrumentation"
        - "Watsonx Instrumentation"
        - "Weaviate Instrumentation"
        - "LLM Semantic Conventions"
        - "Traceloop SDK"
        - "All Packages"
  - type: markdown
    attributes:
      value: |
        We value your time and efforts to submit this Feature request form. 🙏
  - type: textarea
    id: feature-description
    validations:
      required: true
    attributes:
      label: "🔖 Feature description"
      description: "A clear and concise description of what the feature is."
      placeholder: "You should add ..."
  - type: textarea
    id: pitch
    validations:
      required: true
    attributes:
      label: "🎤 Why is this feature needed ?"
      description: "Please explain why this feature should be implemented and how it would be used. Add examples, if applicable."
      placeholder: "In my use-case, ..."
  - type: textarea
    id: solution
    validations:
      required: true
    attributes:
      label: "✌️ How do you aim to achieve this?"
      description: "A clear and concise description of what you want to happen."
      placeholder: "I want this feature to, ..."
  - type: textarea
    id: alternative
    validations:
      required: false
    attributes:
      label: "🔄️ Additional Information"
      description: "A clear and concise description of any alternative solutions or additional solutions you've considered."
      placeholder: "I tried, ..."
  - type: checkboxes
    id: no-duplicate-issues
    attributes:
      label: "👀 Have you spent some time to check if this feature request has been raised before?"
      options:
        - label: "I checked and didn't find similar issue"
          required: true
  - type: dropdown
    id: willing-to-submit-pr
    attributes:
      label: Are you willing to submit PR?
      description: This is absolutely not required, but we are happy to guide you in the contribution process.
      options:
        - "Yes I am willing to submit a PR!"
