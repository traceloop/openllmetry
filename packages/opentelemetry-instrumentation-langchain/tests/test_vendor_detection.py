import pytest
from opentelemetry.instrumentation.langchain.vendor_detection import detect_vendor_from_class


@pytest.mark.parametrize("class_name,expected", [
    # Exact matches
    ("AzureChatOpenAI", "az.ai.openai"),
    ("AzureOpenAI", "az.ai.openai"),
    ("AzureOpenAIEmbeddings", "az.ai.openai"),
    ("ChatOpenAI", "openai"),
    ("OpenAI", "openai"),
    ("OpenAIEmbeddings", "openai"),
    ("ChatBedrock", "aws.bedrock"),
    ("BedrockEmbeddings", "aws.bedrock"),
    ("Bedrock", "aws.bedrock"),
    ("BedrockChat", "aws.bedrock"),
    ("ChatAnthropic", "anthropic"),
    ("AnthropicLLM", "anthropic"),
    ("ChatVertexAI", "gcp.gen_ai"),
    ("VertexAI", "gcp.gen_ai"),
    ("ChatGoogleGenerativeAI", "gcp.gen_ai"),
    ("GoogleGenerativeAI", "gcp.gen_ai"),
    ("ChatCohere", "cohere"),
    ("Cohere", "cohere"),
    ("HuggingFacePipeline", "hugging_face"),
    ("HuggingFaceTextGenInference", "hugging_face"),
    ("ChatHuggingFace", "hugging_face"),
    ("ChatOllama", "ollama"),
    ("Ollama", "ollama"),
    ("Together", "together_ai"),
    ("ChatTogether", "together_ai"),
    ("Replicate", "replicate"),
    ("ChatReplicate", "replicate"),
    ("ChatFireworks", "fireworks"),
    ("Fireworks", "fireworks"),
    ("ChatGroq", "groq"),
    ("ChatMistralAI", "mistral_ai"),
    ("MistralAI", "mistral_ai"),
    # Pattern matches
    ("SomeAzureModel", "az.ai.openai"),
    ("CustomOpenAIModel", "openai"),
    ("AwsBedrockModel", "aws.bedrock"),
    ("AnthropicCustom", "anthropic"),
    ("VertexCustom", "gcp.gen_ai"),
    ("GeminiModel", "gcp.gen_ai"),
    ("CohereCustom", "cohere"),
    ("OllamaCustom", "ollama"),
    ("TogetherCustom", "together_ai"),
    ("ReplicateCustom", "replicate"),
    ("FireworksCustom", "fireworks"),
    ("GroqCustom", "groq"),
    ("MistralCustom", "mistral_ai"),
    # Default fallback
    ("UnknownModel", "langchain"),
    ("", "langchain"),
    (None, "langchain"),
])
def test_detect_vendor_from_class(class_name, expected):
    assert detect_vendor_from_class(class_name) == expected
