import atexit
import logging
import os
import importlib.util


from colorama import Fore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from opentelemetry.context import get_value, attach, set_value

from opentelemetry.semconv.ai import SpanAttributes
from traceloop.sdk import Telemetry
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.tracing.content_allow_list import ContentAllowList
from traceloop.sdk.utils import is_notebook
from typing import Dict, Optional, Set

TRACER_NAME = "traceloop.tracer"
EXCLUDED_URLS = """
    iam.cloud.ibm.com,
    dataplatform.cloud.ibm.com,
    ml.cloud.ibm.com,
    api.openai.com,
    openai.azure.com,
    api.anthropic.com,
    api.cohere.ai,
    pinecone.io,
    traceloop.com,
    posthog.com,
    bedrock-runtime,
    googleapis.com,
    githubusercontent.com"""


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    endpoint: str = None
    headers: Dict[str, str] = {}

    def __new__(
        cls,
        disable_batch=False,
        processor: SpanProcessor = None,
        propagator: TextMapPropagator = None,
        exporter: SpanExporter = None,
        instruments: Optional[Set[Instruments]] = None,
    ) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            if not TracerWrapper.endpoint:
                return obj

            obj.__resource = Resource(attributes=TracerWrapper.resource_attributes)
            obj.__tracer_provider: TracerProvider = init_tracer_provider(
                resource=obj.__resource
            )
            if processor:
                Telemetry().capture("tracer:init", {"processor": "custom"})
                obj.__spans_processor: SpanProcessor = processor
            else:
                if exporter:
                    Telemetry().capture(
                        "tracer:init",
                        {
                            "exporter": "custom",
                            "processor": "simple" if disable_batch else "batch",
                        },
                    )
                else:
                    Telemetry().capture(
                        "tracer:init",
                        {
                            "exporter": TracerWrapper.endpoint,
                            "processor": "simple" if disable_batch else "batch",
                        },
                    )

                obj.__spans_exporter: SpanExporter = (
                    exporter
                    if exporter
                    else init_spans_exporter(
                        TracerWrapper.endpoint, TracerWrapper.headers
                    )
                )
                if disable_batch or is_notebook():
                    obj.__spans_processor: SpanProcessor = SimpleSpanProcessor(
                        obj.__spans_exporter
                    )
                else:
                    obj.__spans_processor: SpanProcessor = BatchSpanProcessor(
                        obj.__spans_exporter
                    )

            obj.__spans_processor.on_start = obj._span_processor_on_start
            obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            if propagator:
                set_global_textmap(propagator)

            instrument_set = False
            if instruments is None:
                init_instrumentations()
                instrument_set = True
            else:
                for instrument in instruments:
                    if instrument == Instruments.OPENAI:
                        if not init_openai_instrumentor():
                            print(Fore.RED + "Warning: OpenAI library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.ANTHROPIC:
                        if not init_anthropic_instrumentor():
                            print(
                                Fore.RED + "Warning: Anthropic library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.COHERE:
                        if not init_cohere_instrumentor():
                            print(Fore.RED + "Warning: Cohere library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.PINECONE:
                        if not init_pinecone_instrumentor():
                            print(
                                Fore.RED + "Warning: Pinecone library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.CHROMA:
                        if not init_chroma_instrumentor():
                            print(Fore.RED + "Warning: Chroma library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.LANGCHAIN:
                        if not init_langchain_instrumentor():
                            print(
                                Fore.RED + "Warning: LangChain library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.LLAMA_INDEX:
                        if not init_llama_index_instrumentor():
                            print(
                                Fore.RED + "Warning: LlamaIndex library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.TRANSFORMERS:
                        if not init_transformers_instrumentor():
                            print(
                                Fore.RED
                                + "Warning: Transformers library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.REQUESTS:
                        if not init_requests_instrumentor():
                            print(
                                Fore.RED + "Warning: Requests library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.URLLIB3:
                        if not init_urllib3_instrumentor():
                            print(Fore.RED + "Warning: urllib3 library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.PYMYSQL:
                        if not init_pymysql_instrumentor():
                            print(Fore.RED + "Warning: PyMySQL library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.BEDROCK:
                        if not init_bedrock_instrumentor():
                            print(Fore.RED + "Warning: Bedrock library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.REPLICATE:
                        if not init_replicate_instrumentor():
                            print(
                                Fore.RED + "Warning: Replicate library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.VERTEXAI:
                        if not init_vertexai_instrumentor():
                            print(
                                Fore.RED + "Warning: Vertex AI library does not exist."
                            )
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.WATSONX:
                        if not init_watsonx_instrumentor():
                            print(Fore.RED + "Warning: Watsonx library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True
                    elif instrument == Instruments.WEAVIATE:
                        if not init_weaviate_instrumentor():
                            print(Fore.RED + "Warning: Weaviate library does not exist.")
                            print(Fore.RESET)
                        else:
                            instrument_set = True

                    else:
                        print(
                            Fore.RED
                            + "Warning: "
                            + instrument
                            + " library does not exist."
                        )
                        print(Fore.RESET)

            if not instrument_set:
                print(
                    Fore.RED + "Warning: No valid instruments set. Remove 'instrument' "
                    "argument to use all instruments, or set a valid instrument."
                )
                print(Fore.RESET)

            obj.__content_allow_list = ContentAllowList()

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        self.flush()

    def _span_processor_on_start(self, span, parent_context):
        workflow_name = get_value("workflow_name")
        if workflow_name is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)

        correlation_id = get_value("correlation_id")
        if correlation_id is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_CORRELATION_ID, correlation_id)

        association_properties = get_value("association_properties")
        if association_properties is not None:
            for key, value in association_properties.items():
                span.set_attribute(
                    f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.{key}", value
                )

            if not self.enable_content_tracing:
                if self.__content_allow_list.is_allowed(association_properties):
                    attach(set_value("override_enable_content_tracing", True))
                else:
                    attach(set_value("override_enable_content_tracing", False))

        if is_llm_span(span):
            prompt_key = get_value("prompt_key")
            if prompt_key is not None:
                span.set_attribute("traceloop.prompt.key", prompt_key)

            prompt_version = get_value("prompt_version")
            if prompt_version is not None:
                span.set_attribute("traceloop.prompt.version", prompt_version)

            prompt_version_name = get_value("prompt_version_name")
            if prompt_version_name is not None:
                span.set_attribute("traceloop.prompt.version_name", prompt_version_name)

            prompt_version_hash = get_value("prompt_version_hash")
            if prompt_version_hash is not None:
                span.set_attribute("traceloop.prompt.version_hash", prompt_version_hash)

            prompt_template_variables = get_value("prompt_template_variables")
            if prompt_version_hash is not None:
                for key, value in prompt_template_variables.items():
                    span.set_attribute(
                        f"traceloop.prompt.template_variables.{key}", value
                    )

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing
        TracerWrapper.endpoint = endpoint
        TracerWrapper.headers = headers

    @classmethod
    def verify_initialized(cls) -> bool:
        if hasattr(cls, "instance"):
            return True

        if (os.getenv("TRACELOOP_SUPPRESS_WARNINGS") or "false").lower() == "true":
            return False

        print(
            Fore.RED
            + "Warning: Traceloop not initialized, make sure you call Traceloop.init()"
        )
        print(Fore.RESET)
        return False

    def flush(self):
        self.__spans_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_correlation_id(correlation_id: str) -> None:
    attach(set_value("correlation_id", correlation_id))


def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))


def set_workflow_name(workflow_name: str) -> None:
    attach(set_value("workflow_name", workflow_name))


def set_prompt_tracing_context(
    key: str,
    version: int,
    version_name: str,
    version_hash: str,
    template_variables: dict,
) -> None:
    attach(set_value("prompt_key", key))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_version_name", version_name))
    attach(set_value("prompt_version_hash", version_hash))
    attach(set_value("prompt_template_variables", template_variables))


def is_llm_span(span) -> bool:
    return span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) is not None


def init_spans_exporter(api_endpoint: str, headers: Dict[str, str]) -> SpanExporter:
    if "http" in api_endpoint.lower() or "https" in api_endpoint.lower():
        return HTTPExporter(endpoint=f"{api_endpoint}/v1/traces", headers=headers)
    else:
        return GRPCExporter(endpoint=f"{api_endpoint}", headers=headers)


def init_tracer_provider(resource: Resource) -> TracerProvider:
    provider: TracerProvider = None
    default_provider: TracerProvider = get_tracer_provider()

    if isinstance(default_provider, ProxyTracerProvider):
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
    elif not hasattr(default_provider, "add_span_processor"):
        logging.error(
            "Cannot add span processor to the default provider since it doesn't support it"
        )
        return
    else:
        provider = default_provider

    return provider


def init_instrumentations():
    init_openai_instrumentor()
    init_anthropic_instrumentor()
    init_cohere_instrumentor()
    init_pinecone_instrumentor()
    init_qdrant_instrumentor()
    init_chroma_instrumentor()
    init_haystack_instrumentor()
    init_langchain_instrumentor()
    init_llama_index_instrumentor()
    init_transformers_instrumentor()
    init_requests_instrumentor()
    init_urllib3_instrumentor()
    init_pymysql_instrumentor()
    init_bedrock_instrumentor()
    init_replicate_instrumentor()
    init_vertexai_instrumentor()
    init_watsonx_instrumentor()
    init_weaviate_instrumentor()


def init_openai_instrumentor():
    if importlib.util.find_spec("openai") is not None:
        Telemetry().capture("instrumentation:openai:init")
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_anthropic_instrumentor():
    if importlib.util.find_spec("anthropic") is not None:
        Telemetry().capture("instrumentation:anthropic:init")
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_cohere_instrumentor():
    if importlib.util.find_spec("cohere") is not None:
        Telemetry().capture("instrumentation:cohere:init")
        from opentelemetry.instrumentation.cohere import CohereInstrumentor

        instrumentor = CohereInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_pinecone_instrumentor():
    if importlib.util.find_spec("pinecone") is not None:
        Telemetry().capture("instrumentation:pinecone:init")
        from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

        instrumentor = PineconeInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_qdrant_instrumentor():
    if importlib.util.find_spec("qdrant_client") is not None:
        Telemetry().capture("instrumentation:qdrant:init")
        from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

        instrumentor = QdrantInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


def init_chroma_instrumentor():
    if importlib.util.find_spec("chromadb") is not None:
        Telemetry().capture("instrumentation:chromadb:init")
        from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

        instrumentor = ChromaInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_haystack_instrumentor():
    if importlib.util.find_spec("haystack") is not None:
        Telemetry().capture("instrumentation:haystack:init")
        from opentelemetry.instrumentation.haystack import HaystackInstrumentor

        instrumentor = HaystackInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_langchain_instrumentor():
    if importlib.util.find_spec("langchain") is not None:
        Telemetry().capture("instrumentation:langchain:init")
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor

        instrumentor = LangchainInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_transformers_instrumentor():
    if importlib.util.find_spec("transformers") is not None:
        Telemetry().capture("instrumentation:transformers:init")
        from opentelemetry.instrumentation.transformers import TransformersInstrumentor

        instrumentor = TransformersInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_llama_index_instrumentor():
    if importlib.util.find_spec("llama_index") is not None:
        Telemetry().capture("instrumentation:llamaindex:init")
        from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

        instrumentor = LlamaIndexInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_requests_instrumentor():
    if importlib.util.find_spec("requests") is not None:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        instrumentor = RequestsInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
    return True


def init_urllib3_instrumentor():
    if importlib.util.find_spec("urllib3") is not None:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        instrumentor = URLLib3Instrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
    return True


def init_pymysql_instrumentor():
    if importlib.util.find_spec("sqlalchemy") is not None:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        instrumentor = SQLAlchemyInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_bedrock_instrumentor():
    if importlib.util.find_spec("boto3") is not None:
        from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

        instrumentor = BedrockInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_replicate_instrumentor():
    if importlib.util.find_spec("replicate") is not None:
        Telemetry().capture("instrumentation:replicate:init")
        from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

        instrumentor = ReplicateInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_vertexai_instrumentor():
    if importlib.util.find_spec("vertexai") is not None:
        Telemetry().capture("instrumentation:vertexai:init")
        from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

        instrumentor = VertexAIInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_watsonx_instrumentor():
    if importlib.util.find_spec("ibm_watson_machine_learning") is not None:
        Telemetry().capture("instrumentation:watsonx:init")
        from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

        instrumentor = WatsonxInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True


def init_weaviate_instrumentor():
    if importlib.util.find_spec("weaviate") is not None:
        Telemetry().capture("instrumentation:weaviate:init")
        from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

        instrumentor = WeaviateInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return True
