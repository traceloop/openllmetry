from enum import Enum


class LLMVendor(Enum):
    """
    Supported LLM vendor names used across OpenLLMetry instrumentations.
    
    These values match the actual strings used in span attributes (LLM_SYSTEM)
    throughout the instrumentation packages.
    """
    
    # === Direct LLM Providers ===
    OPENAI = "openai"
    ANTHROPIC = "Anthropic"
    COHERE = "Cohere" 
    MISTRALAI = "MistralAI"
    OLLAMA = "Ollama"
    GROQ = "Groq"
    ALEPH_ALPHA = "AlephAlpha"
    REPLICATE = "Replicate"
    TOGETHER_AI = "TogetherAI"
    WATSONX = "Watsonx"  
    HUGGINGFACE = "HuggingFace"
    FIREWORKS = "Fireworks"
    
    # === Cloud Providers ===
    AZURE = "Azure"         # Azure OpenAI
    AWS = "AWS"             # AWS Bedrock  
    GOOGLE = "Google"       # Google VertexAI + Generative AI
    OPENROUTER = "OpenRouter" # Should not use this vendor name, rather report the actual vendor name
    
    # === Frameworks/Wrappers ===
    LANGCHAIN = "Langchain" # Should not use this vendor name, rather report the actual vendor name
    CREWAI = "crewai"
        
    @classmethod
    def values(cls):
        """Return all vendor values as a list of strings."""
        return [vendor.value for vendor in cls]
    
    def __str__(self):
        """Return the vendor value as string for easy usage."""
        return self.value 