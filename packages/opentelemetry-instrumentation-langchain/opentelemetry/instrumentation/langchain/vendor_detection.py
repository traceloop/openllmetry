from dataclasses import dataclass
from typing import Set, List


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
            vendor_name="Azure"
        ),
        VendorRule(
            exact_matches={"ChatOpenAI", "OpenAI", "OpenAIEmbeddings"},
            patterns=["openai"],
            vendor_name="openai"
        ),
        VendorRule(
            exact_matches={"ChatBedrock", "BedrockEmbeddings", "Bedrock", "BedrockChat"},
            patterns=["bedrock", "aws"],
            vendor_name="AWS"
        ),
        VendorRule(
            exact_matches={"ChatAnthropic", "AnthropicLLM"},
            patterns=["anthropic"],
            vendor_name="Anthropic"
        ),
        VendorRule(
            exact_matches={
                "ChatVertexAI", "VertexAI", "VertexAIEmbeddings", "ChatGoogleGenerativeAI",
                "GoogleGenerativeAI", "GooglePaLM", "ChatGooglePaLM"
            },
            patterns=["vertex", "google", "palm", "gemini"],
            vendor_name="Google"
        ),
        VendorRule(
            exact_matches={"ChatCohere", "CohereEmbeddings", "Cohere"},
            patterns=["cohere"],
            vendor_name="Cohere"
        ),
        VendorRule(
            exact_matches={
                "HuggingFacePipeline", "HuggingFaceTextGenInference",
                "HuggingFaceEmbeddings", "ChatHuggingFace"
            },
            patterns=["huggingface"],
            vendor_name="HuggingFace"
        ),
        VendorRule(
            exact_matches={"ChatOllama", "OllamaEmbeddings", "Ollama"},
            patterns=["ollama"],
            vendor_name="Ollama"
        ),
        VendorRule(
            exact_matches={"Together", "ChatTogether"},
            patterns=["together"],
            vendor_name="Together"
        ),
        VendorRule(
            exact_matches={"Replicate", "ChatReplicate"},
            patterns=["replicate"],
            vendor_name="Replicate"
        ),
        VendorRule(
            exact_matches={"ChatFireworks", "Fireworks"},
            patterns=["fireworks"],
            vendor_name="Fireworks"
        ),
        VendorRule(
            exact_matches={"ChatGroq"},
            patterns=["groq"],
            vendor_name="Groq"
        ),
        VendorRule(
            exact_matches={"ChatMistralAI", "MistralAI"},
            patterns=["mistral"],
            vendor_name="MistralAI"
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
        return "Langchain"

    vendor_rules = _get_vendor_rules()

    for rule in vendor_rules:
        if rule.matches(class_name):
            return rule.vendor_name

    return "Langchain"
