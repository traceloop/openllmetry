from dataclasses import dataclass
from typing import Set, List
from opentelemetry.semconv_ai import LLMVendor


@dataclass(frozen=True)
class VendorRule:
    exact_matches: Set[str]
    patterns: List[str]
    vendor_name: str

    def matches(self, class_name: str) -> bool:
        if class_name in self.exact_matches:
            return True
        class_lower = class_name.lower()
        return any(pattern in class_lower for pattern in self.patterns)


def _get_vendor_rules() -> List[VendorRule]:
    """
    Get vendor detection rules ordered by specificity (most specific first).

    Returns:
        List of VendorRule objects for detecting LLM vendors from class names
    """
    return [
        VendorRule(
            exact_matches={"AzureChatOpenAI", "AzureOpenAI", "AzureOpenAIEmbeddings"},
            patterns=["azure"],
            vendor_name=LLMVendor.AZURE.value
        ),
        VendorRule(
            exact_matches={"ChatOpenAI", "OpenAI", "OpenAIEmbeddings"},
            patterns=["openai"],
            vendor_name=LLMVendor.OPENAI.value
        ),
        VendorRule(
            exact_matches={"ChatBedrock", "BedrockEmbeddings", "Bedrock", "BedrockChat"},
            patterns=["bedrock", "aws"],
            vendor_name=LLMVendor.AWS.value
        ),
        VendorRule(
            exact_matches={"ChatAnthropic", "AnthropicLLM"},
            patterns=["anthropic"],
            vendor_name=LLMVendor.ANTHROPIC.value
        ),
        VendorRule(
            exact_matches={
                "ChatVertexAI", "VertexAI", "VertexAIEmbeddings", "ChatGoogleGenerativeAI",
                "GoogleGenerativeAI", "GooglePaLM", "ChatGooglePaLM"
            },
            patterns=["vertex", "google", "palm", "gemini"],
            vendor_name=LLMVendor.GOOGLE.value
        ),
        VendorRule(
            exact_matches={"ChatCohere", "CohereEmbeddings", "Cohere"},
            patterns=["cohere"],
            vendor_name=LLMVendor.COHERE.value
        ),
        VendorRule(
            exact_matches={
                "HuggingFacePipeline", "HuggingFaceTextGenInference",
                "HuggingFaceEmbeddings", "ChatHuggingFace"
            },
            patterns=["huggingface"],
            vendor_name=LLMVendor.HUGGINGFACE.value
        ),
        VendorRule(
            exact_matches={"ChatOllama", "OllamaEmbeddings", "Ollama"},
            patterns=["ollama"],
            vendor_name=LLMVendor.OLLAMA.value
        ),
        VendorRule(
            exact_matches={"Together", "ChatTogether"},
            patterns=["together"],
            vendor_name=LLMVendor.TOGETHER_AI.value
        ),
        VendorRule(
            exact_matches={"Replicate", "ChatReplicate"},
            patterns=["replicate"],
            vendor_name=LLMVendor.REPLICATE.value
        ),
        VendorRule(
            exact_matches={"ChatFireworks", "Fireworks"},
            patterns=["fireworks"],
            vendor_name=LLMVendor.FIREWORKS.value
        ),
        VendorRule(
            exact_matches={"ChatGroq"},
            patterns=["groq"],
            vendor_name=LLMVendor.GROQ.value
        ),
        VendorRule(
            exact_matches={"ChatMistralAI", "MistralAI"},
            patterns=["mistral"],
            vendor_name=LLMVendor.MISTRALAI.value
        ),
    ]


def detect_vendor_from_class(class_name: str) -> str:
    """
    Detect vendor from LangChain model class name.
    Uses unified detection rules combining exact matches and patterns.

    Args:
        class_name: The class name extracted from serialized model information

    Returns:
        Vendor string, defaults to "Langchain" if no match found
    """
    if not class_name:
        return LLMVendor.LANGCHAIN.value

    vendor_rules = _get_vendor_rules()

    for rule in vendor_rules:
        if rule.matches(class_name):
            return rule.vendor_name

    return "Langchain"
