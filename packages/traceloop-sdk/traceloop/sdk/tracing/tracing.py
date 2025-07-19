import atexit
import importlib
import logging
import os


from colorama import Fore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from opentelemetry.context import get_value, attach, set_value
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk import Telemetry
from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.tracing.content_allow_list import ContentAllowList
from traceloop.sdk.utils import is_notebook
from traceloop.sdk.utils.package_check import is_package_installed
from typing import Callable, Dict, Optional, Set


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
    sentry.io,
    bedrock-runtime,
    sagemaker-runtime,
    googleapis.com,
    githubusercontent.com,
    openaipublic.blob.core.windows.net"""


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    endpoint: str = None
    headers: Dict[str, str] = {}
    __tracer_provider: TracerProvider = None
    __image_uploader: ImageUploader = None
    __disabled: bool = False

    def __new__(
        cls,
        disable_batch=False,
        processor: SpanProcessor = None,
        propagator: TextMapPropagator = None,
        exporter: SpanExporter = None,
        sampler: Optional[Sampler] = None,
        should_enrich_metrics: bool = True,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        image_uploader: ImageUploader = None,
        span_postprocess_callback: Optional[Callable[[ReadableSpan], None]] = None,
    ) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            if not TracerWrapper.endpoint:
                return obj

            obj.__image_uploader = image_uploader
            obj.__resource = Resource(attributes=TracerWrapper.resource_attributes)
            obj.__tracer_provider = init_tracer_provider(resource=obj.__resource, sampler=sampler)
            if processor:
                Telemetry().capture("tracer:init", {"processor": "custom"})
                obj.__spans_processor: SpanProcessor = processor
                obj.__spans_processor_original_on_start = processor.on_start
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
                obj.__spans_processor_original_on_start = None
                if span_postprocess_callback:
                    # Create a wrapper that calls both the custom and original methods
                    original_on_end = obj.__spans_processor.on_end

                    def wrapped_on_end(span):
                        # Call the custom on_end first
                        span_postprocess_callback(span)
                        # Then call the original to ensure normal processing
                        original_on_end(span)
                    obj.__spans_processor.on_end = wrapped_on_end

            obj.__spans_processor.on_start = obj._span_processor_on_start
            obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            if propagator:
                set_global_textmap(propagator)

            # this makes sure otel context is propagated so we always want it
            ThreadingInstrumentor().instrument()

            init_instrumentations(
                should_enrich_metrics,
                image_uploader.aupload_base64_image,
                instruments,
                block_instruments,
            )

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

        entity_path = get_value("entity_path")
        if entity_path is not None:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

        association_properties = get_value("association_properties")
        if association_properties is not None:
            _set_association_properties_attributes(span, association_properties)

            if not self.enable_content_tracing:
                if self.__content_allow_list.is_allowed(association_properties):
                    attach(set_value("override_enable_content_tracing", True))
                else:
                    attach(set_value("override_enable_content_tracing", False))

        if is_llm_span(span):
            managed_prompt = get_value("managed_prompt")
            if managed_prompt is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_PROMPT_MANAGED, managed_prompt
                )

            prompt_key = get_value("prompt_key")
            if prompt_key is not None:
                span.set_attribute(SpanAttributes.TRACELOOP_PROMPT_KEY, prompt_key)

            prompt_version = get_value("prompt_version")
            if prompt_version is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_PROMPT_VERSION, prompt_version
                )

            prompt_version_name = get_value("prompt_version_name")
            if prompt_version_name is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_PROMPT_VERSION_NAME, prompt_version_name
                )

            prompt_version_hash = get_value("prompt_version_hash")
            if prompt_version_hash is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_PROMPT_VERSION_HASH, prompt_version_hash
                )

            prompt_template = get_value("prompt_template")
            if prompt_template is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_PROMPT_TEMPLATE, prompt_template
                )

            prompt_template_variables = get_value("prompt_template_variables")
            if prompt_template_variables is not None:
                for key, value in prompt_template_variables.items():
                    span.set_attribute(
                        f"{SpanAttributes.TRACELOOP_PROMPT_TEMPLATE_VARIABLES}.{key}",
                        value,
                    )

        # Call original on_start method if it exists in custom processor
        if self.__spans_processor_original_on_start:
            self.__spans_processor_original_on_start(span, parent_context)

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
        if cls.__disabled:
            return False

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

    @classmethod
    def set_disabled(cls, disabled: bool) -> None:
        cls.__disabled = disabled

    def flush(self):
        self.__spans_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))

    # Attach association properties to the current span, if it's a workflow or a task
    span = trace.get_current_span()
    if get_value("workflow_name") is not None or get_value("entity_name") is not None:
        _set_association_properties_attributes(span, properties)


def _set_association_properties_attributes(span, properties: dict) -> None:
    for key, value in properties.items():
        span.set_attribute(
            f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.{key}", value
        )


def set_workflow_name(workflow_name: str) -> None:
    attach(set_value("workflow_name", workflow_name))


def set_entity_path(entity_path: str) -> None:
    attach(set_value("entity_path", entity_path))


def get_chained_entity_path(entity_name: str) -> str:
    parent = get_value("entity_path")
    if parent is None:
        return entity_name
    else:
        return f"{parent}.{entity_name}"


def set_managed_prompt_tracing_context(
    key: str,
    version: int,
    version_name: str,
    version_hash: str,
    template_variables: dict,
) -> None:
    attach(set_value("managed_prompt", True))
    attach(set_value("prompt_key", key))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_version_name", version_name))
    attach(set_value("prompt_version_hash", version_hash))
    attach(set_value("prompt_template_variables", template_variables))


def set_external_prompt_tracing_context(
    template: str, variables: dict, version: int
) -> None:
    attach(set_value("managed_prompt", False))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_template", template))
    attach(set_value("prompt_template_variables", variables))


def is_llm_span(span) -> bool:
    return span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) is not None


def init_spans_exporter(api_endpoint: str, headers: Dict[str, str]) -> SpanExporter:
    if "http" in api_endpoint.lower() or "https" in api_endpoint.lower():
        return HTTPExporter(endpoint=f"{api_endpoint}/v1/traces", headers=headers)
    else:
        return GRPCExporter(endpoint=f"{api_endpoint}", headers=headers)


def init_tracer_provider(resource: Resource, sampler: Optional[Sampler] = None) -> TracerProvider:
    provider: TracerProvider = None
    default_provider: TracerProvider = get_tracer_provider()

    if isinstance(default_provider, ProxyTracerProvider):
        if sampler is not None:
            provider = TracerProvider(resource=resource, sampler=sampler)
        else:
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


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str], str],
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
):
    block_instruments = block_instruments or set()
    instruments = instruments or set(
        Instruments
    )  # Use all instruments if none specified

    # Remove any instruments that were explicitly blocked
    instruments = instruments - block_instruments

    instrument_set = False
    for instrument in instruments:
        if instrument == Instruments.ALEPHALPHA:
            if init_alephalpha_instrumentor():
                instrument_set = True
        elif instrument == Instruments.ANTHROPIC:
            if init_anthropic_instrumentor(
                should_enrich_metrics, base64_image_uploader
            ):
                instrument_set = True
        elif instrument == Instruments.BEDROCK:
            if init_bedrock_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.CHROMA:
            if init_chroma_instrumentor():
                instrument_set = True
        elif instrument == Instruments.COHERE:
            if init_cohere_instrumentor():
                instrument_set = True
        elif instrument == Instruments.CREW:
            if init_crewai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.GOOGLE_GENERATIVEAI:
            if init_google_generativeai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.GROQ:
            if init_groq_instrumentor():
                instrument_set = True
        elif instrument == Instruments.HAYSTACK:
            if init_haystack_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANCEDB:
            if init_lancedb_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANGCHAIN:
            if init_langchain_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LLAMA_INDEX:
            if init_llama_index_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MARQO:
            if init_marqo_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MCP:
            if init_mcp_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MILVUS:
            if init_milvus_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MISTRAL:
            if init_mistralai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OLLAMA:
            if init_ollama_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OPENAI:
            if init_openai_instrumentor(should_enrich_metrics, base64_image_uploader):
                instrument_set = True
        elif instrument == Instruments.OPENAI_AGENTS:
            if init_openai_agents_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PINECONE:
            if init_pinecone_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PYMYSQL:
            if init_pymysql_instrumentor():
                instrument_set = True
        elif instrument == Instruments.QDRANT:
            if init_qdrant_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REDIS:
            if init_redis_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REPLICATE:
            if init_replicate_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REQUESTS:
            if init_requests_instrumentor():
                instrument_set = True
        elif instrument == Instruments.SAGEMAKER:
            if init_sagemaker_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.TOGETHER:
            if init_together_instrumentor():
                instrument_set = True
        elif instrument == Instruments.TRANSFORMERS:
            if init_transformers_instrumentor():
                instrument_set = True
        elif instrument == Instruments.URLLIB3:
            if init_urllib3_instrumentor():
                instrument_set = True
        elif instrument == Instruments.VERTEXAI:
            if init_vertexai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WATSONX:
            if init_watsonx_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WEAVIATE:
            if init_weaviate_instrumentor():
                instrument_set = True
        else:
            print(Fore.RED + f"Warning: {instrument} instrumentation does not exist.")
            print(
                "Usage:\n"
                "from traceloop.sdk.instruments import Instruments\n"
                "Traceloop.init(app_name='...', instruments=set([Instruments.OPENAI]))"
            )
            print(Fore.RESET)

    if not instrument_set:
        print(
            Fore.RED
            + "Warning: No valid instruments set. "
            + "Ensure the instrumented libraries are installed, specify valid instruments, "
            + "or remove 'instruments' argument to use all instruments."
        )
        print(Fore.RESET)

    return instrument_set


def init_openai_instrumentor(
    should_enrich_metrics: bool, base64_image_uploader: Callable[[str, str, str], str]
):
    return _init_generic_instrumentor(
        library_name="OpenAI",
        instrumentor_package="opentelemetry-instrumentation-openai",
        instrumentor_class_name="OpenAIInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.openai",
        library_packages=["openai"],
        extra_name="openai",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
            "enrich_assistant": should_enrich_metrics,
            "enrich_token_usage": should_enrich_metrics,
            "get_common_metrics_attributes": metrics_common_attributes,
            "upload_base64_image": base64_image_uploader,
        },
        telemetry_event="instrumentation:openai:init",
    )


def init_anthropic_instrumentor(
    should_enrich_metrics: bool, base64_image_uploader: Callable[[str, str, str], str]
):
    return _init_generic_instrumentor(
        library_name="Anthropic",
        instrumentor_package="opentelemetry-instrumentation-anthropic",
        instrumentor_class_name="AnthropicInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.anthropic",
        library_packages=["anthropic"],
        extra_name="anthropic",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
            "enrich_token_usage": should_enrich_metrics,
            "get_common_metrics_attributes": metrics_common_attributes,
            "upload_base64_image": base64_image_uploader,
        },
        telemetry_event="instrumentation:anthropic:init",
    )


def init_cohere_instrumentor():
    return _init_generic_instrumentor(
        library_name="Cohere",
        instrumentor_package="opentelemetry-instrumentation-cohere",
        instrumentor_class_name="CohereInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.cohere",
        library_packages=["cohere"],
        extra_name="cohere",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:cohere:init",
    )


def init_pinecone_instrumentor():
    return _init_generic_instrumentor(
        library_name="Pinecone",
        instrumentor_package="opentelemetry-instrumentation-pinecone",
        instrumentor_class_name="PineconeInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.pinecone",
        library_packages=["pinecone"],
        extra_name="pinecone",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:pinecone:init",
    )


def init_qdrant_instrumentor():
    return _init_generic_instrumentor(
        library_name="Qdrant",
        instrumentor_package="opentelemetry-instrumentation-qdrant",
        instrumentor_class_name="QdrantInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.qdrant",
        library_packages=["qdrant_client"],
        extra_name="qdrant",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:qdrant:init",
    )


def init_chroma_instrumentor():
    return _init_generic_instrumentor(
        library_name="ChromaDB",
        instrumentor_package="opentelemetry-instrumentation-chromadb",
        instrumentor_class_name="ChromaInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.chromadb",
        library_packages=["chromadb"],
        extra_name="chromadb",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:chromadb:init",
    )


def init_requests_instrumentor():
    return _init_generic_instrumentor(
        library_name="Requests",
        instrumentor_package="opentelemetry-instrumentation-requests",
        instrumentor_class_name="RequestsInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.requests",
        library_packages=["requests"],
        extra_name="requests",
        requires_instrumentation_package=False,
        excluded_urls=EXCLUDED_URLS,
    )


def init_urllib3_instrumentor():
    return _init_generic_instrumentor(
        library_name="urllib3",
        instrumentor_package="opentelemetry-instrumentation-urllib3",
        instrumentor_class_name="URLLib3Instrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.urllib3",
        library_packages=["urllib3"],
        extra_name="urllib3",
        requires_instrumentation_package=False,
        excluded_urls=EXCLUDED_URLS,
    )


def init_redis_instrumentor():
    return _init_generic_instrumentor(
        library_name="Redis",
        instrumentor_package="opentelemetry-instrumentation-redis",
        instrumentor_class_name="RedisInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.redis",
        library_packages=["redis"],
        extra_name="redis",
        requires_instrumentation_package=False,
        excluded_urls=EXCLUDED_URLS,
    )


def init_google_generativeai_instrumentor():
    return _init_generic_instrumentor(
        library_name="Google GenerativeAI",
        instrumentor_package="opentelemetry-instrumentation-google-generativeai",
        instrumentor_class_name="GoogleGenerativeAiInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.google_generativeai",
        custom_check_function=lambda: (
            is_package_installed("google-generativeai") and 
            is_package_installed("opentelemetry-instrumentation-google-generativeai")
        ) or is_package_installed("google-genai"),
        extra_name="google-generativeai",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:gemini:init",
    )


def init_langchain_instrumentor():
    return _init_generic_instrumentor(
        library_name="LangChain/LangGraph",
        instrumentor_package="opentelemetry-instrumentation-langchain",
        instrumentor_class_name="LangchainInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.langchain",
        custom_check_function=lambda: is_package_installed("langchain") or is_package_installed("langgraph"),
        extra_name="langchain",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:langchain:init",
    )


def init_mistralai_instrumentor():
    return _init_generic_instrumentor(
        library_name="MistralAI",
        instrumentor_package="opentelemetry-instrumentation-mistralai",
        instrumentor_class_name="MistralAiInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.mistralai",
        library_packages=["mistralai"],
        extra_name="mistralai",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:mistralai:init",
    )


def init_ollama_instrumentor():
    return _init_generic_instrumentor(
        library_name="Ollama",
        instrumentor_package="opentelemetry-instrumentation-ollama",
        instrumentor_class_name="OllamaInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.ollama",
        library_packages=["ollama"],
        extra_name="ollama",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:ollama:init",
    )


def init_transformers_instrumentor():
    return _init_generic_instrumentor(
        library_name="Transformers",
        instrumentor_package="opentelemetry-instrumentation-transformers",
        instrumentor_class_name="TransformersInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.transformers",
        library_packages=["transformers"],
        extra_name="transformers",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:transformers:init",
    )


def init_together_instrumentor():
    return _init_generic_instrumentor(
        library_name="Together",
        instrumentor_package="opentelemetry-instrumentation-together",
        instrumentor_class_name="TogetherAiInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.together",
        library_packages=["together"],
        extra_name="together",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:together:init",
    )


def init_llama_index_instrumentor():
    return _init_generic_instrumentor(
        library_name="LlamaIndex",
        instrumentor_package="opentelemetry-instrumentation-llamaindex",
        instrumentor_class_name="LlamaIndexInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.llamaindex",
        custom_check_function=lambda: is_package_installed("llama-index") or is_package_installed("llama_index"),
        extra_name="llamaindex",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:llamaindex:init",
    )


def init_milvus_instrumentor():
    return _init_generic_instrumentor(
        library_name="Pymilvus",
        instrumentor_package="opentelemetry-instrumentation-milvus",
        instrumentor_class_name="MilvusInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.milvus",
        library_packages=["pymilvus"],
        extra_name="milvus",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:milvus:init",
    )


def init_haystack_instrumentor():
    return _init_generic_instrumentor(
        library_name="Haystack",
        instrumentor_package="opentelemetry-instrumentation-haystack",
        instrumentor_class_name="HaystackInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.haystack",
        library_packages=["haystack"],
        extra_name="haystack",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:haystack:init",
    )


def init_pymysql_instrumentor():
    return _init_generic_instrumentor(
        library_name="SQLAlchemy",
        instrumentor_package="opentelemetry-instrumentation-sqlalchemy",
        instrumentor_class_name="SQLAlchemyInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.sqlalchemy",
        library_packages=["sqlalchemy"],
        extra_name="sqlalchemy",
        requires_instrumentation_package=False,
    )


def init_bedrock_instrumentor(should_enrich_metrics: bool):
    return _init_generic_instrumentor(
        library_name="Bedrock",
        instrumentor_package="opentelemetry-instrumentation-bedrock",
        instrumentor_class_name="BedrockInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.bedrock",
        library_packages=["boto3"],
        extra_name="bedrock",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
            "enrich_token_usage": should_enrich_metrics,
        },
    )


def init_sagemaker_instrumentor(should_enrich_metrics: bool):
    return _init_generic_instrumentor(
        library_name="SageMaker",
        instrumentor_package="opentelemetry-instrumentation-sagemaker",
        instrumentor_class_name="SageMakerInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.sagemaker",
        library_packages=["boto3"],
        extra_name="sagemaker",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
            "enrich_token_usage": should_enrich_metrics,
        },
    )


def init_replicate_instrumentor():
    return _init_generic_instrumentor(
        library_name="Replicate",
        instrumentor_package="opentelemetry-instrumentation-replicate",
        instrumentor_class_name="ReplicateInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.replicate",
        library_packages=["replicate"],
        extra_name="replicate",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:replicate:init",
    )


def init_vertexai_instrumentor():
    return _init_generic_instrumentor(
        library_name="Google Cloud AI Platform",
        instrumentor_package="opentelemetry-instrumentation-vertexai",
        instrumentor_class_name="VertexAIInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.vertexai",
        library_packages=["google-cloud-aiplatform"],
        extra_name="vertexai",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:vertexai:init",
        use_warning_log=True,
    )


def init_watsonx_instrumentor():
    return _init_generic_instrumentor(
        library_name="IBM WatsonX",
        instrumentor_package="opentelemetry-instrumentation-watsonx",
        instrumentor_class_name="WatsonxInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.watsonx",
        custom_check_function=lambda: is_package_installed("ibm-watsonx-ai") or is_package_installed("ibm_watson_machine_learning"),
        extra_name="watsonx",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:watsonx:init",
        use_warning_log=True,
    )


def init_weaviate_instrumentor():
    return _init_generic_instrumentor(
        library_name="Weaviate",
        instrumentor_package="opentelemetry-instrumentation-weaviate",
        instrumentor_class_name="WeaviateInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.weaviate",
        library_packages=["weaviate"],
        extra_name="weaviate",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:weaviate:init",
        use_warning_log=True,
    )


def init_alephalpha_instrumentor():
    return _init_generic_instrumentor(
        library_name="Aleph Alpha",
        instrumentor_package="opentelemetry-instrumentation-alephalpha",
        instrumentor_class_name="AlephAlphaInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.alephalpha",
        library_packages=["aleph_alpha_client"],
        extra_name="alephalpha",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:alephalpha:init",
    )


def init_marqo_instrumentor():
    return _init_generic_instrumentor(
        library_name="Marqo",
        instrumentor_package="opentelemetry-instrumentation-marqo",
        instrumentor_class_name="MarqoInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.marqo",
        library_packages=["marqo"],
        extra_name="marqo",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:marqo:init",
    )


def init_lancedb_instrumentor():
    return _init_generic_instrumentor(
        library_name="LanceDB",
        instrumentor_package="opentelemetry-instrumentation-lancedb",
        instrumentor_class_name="LanceInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.lancedb",
        library_packages=["lancedb"],
        extra_name="lancedb",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:lancedb:init",
    )


def init_groq_instrumentor():
    return _init_generic_instrumentor(
        library_name="Groq",
        instrumentor_package="opentelemetry-instrumentation-groq",
        instrumentor_class_name="GroqInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.groq",
        library_packages=["groq"],
        extra_name="groq",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:groq:init",
    )


def init_crewai_instrumentor():
    return _init_generic_instrumentor(
        library_name="CrewAI",
        instrumentor_package="opentelemetry-instrumentation-crewai",
        instrumentor_class_name="CrewAIInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.crewai",
        library_packages=["crewai"],
        extra_name="crewai",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:crewai:init",
    )


def init_mcp_instrumentor():
    return _init_generic_instrumentor(
        library_name="MCP",
        instrumentor_package="opentelemetry-instrumentation-mcp",
        instrumentor_class_name="McpInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.mcp",
        library_packages=["mcp"],
        extra_name="mcp",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:mcp:init",
    )


def init_openai_agents_instrumentor():
    return _init_generic_instrumentor(
        library_name="OpenAI Agents",
        instrumentor_package="opentelemetry-instrumentation-openai-agents",
        instrumentor_class_name="OpenAIAgentsInstrumentor",
        instrumentor_import_path="opentelemetry.instrumentation.openai_agents",
        library_packages=["openai-agents"],
        extra_name="openai-agents",
        instrumentor_kwargs={
            "exception_logger": lambda e: Telemetry().log_exception(e),
        },
        telemetry_event="instrumentation:openai_agents:init",
    )


def _init_generic_instrumentor(
    library_name: str,
    instrumentor_package: str,
    instrumentor_class_name: str,
    instrumentor_import_path: str,
    library_packages: list = None,
    extra_name: str = None,
    instrumentor_kwargs: dict = None,
    telemetry_event: str = None,
    excluded_urls: str = None,
    requires_instrumentation_package: bool = True,
    custom_check_function: callable = None,
    use_warning_log: bool = False,
):
    """
    Generic instrumentor initialization function
    
    Args:
        library_name: Display name of the library (e.g., "OpenAI", "Anthropic")
        instrumentor_package: Package name to check for instrumentation (e.g., "opentelemetry-instrumentation-openai")
        instrumentor_class_name: Name of the instrumentor class (e.g., "OpenAIInstrumentor")
        instrumentor_import_path: Import path for the instrumentor (e.g., "opentelemetry.instrumentation.openai")
        library_packages: List of library packages to check (e.g., ["openai"])
        extra_name: Name of the extra for installation instructions (e.g., "openai")
        instrumentor_kwargs: Additional kwargs to pass to the instrumentor
        telemetry_event: Custom telemetry event name
        excluded_urls: URLs to exclude for basic instrumentors
        requires_instrumentation_package: Whether to check for instrumentation package
        custom_check_function: Custom function to check if library is available
        use_warning_log: Whether to use warning log instead of error log for exceptions
    """
    try:
        # Default library packages to check
        if library_packages is None:
            library_packages = []
        
        # Check if library is installed
        library_installed = False
        if custom_check_function:
            library_installed = custom_check_function()
        elif library_packages:
            library_installed = any(is_package_installed(pkg) for pkg in library_packages)
        else:
            # For basic instrumentors, assume library is installed if no packages specified
            library_installed = True
        
        # Check if instrumentation package is installed
        instrumentation_installed = True
        if requires_instrumentation_package:
            instrumentation_installed = is_package_installed(instrumentor_package)
        
        if library_installed and instrumentation_installed:
            # Capture telemetry event
            if telemetry_event:
                Telemetry().capture(telemetry_event)
            
            # Import and initialize instrumentor
            module = importlib.import_module(instrumentor_import_path)
            instrumentor_class = getattr(module, instrumentor_class_name)
            
            # Prepare kwargs
            kwargs = instrumentor_kwargs or {}
            
            instrumentor = instrumentor_class(**kwargs)
            
            if not instrumentor.is_instrumented_by_opentelemetry:
                if excluded_urls:
                    instrumentor.instrument(excluded_urls=excluded_urls)
                else:
                    instrumentor.instrument()
                    
        elif library_installed and requires_instrumentation_package:
            # Library is installed but instrumentation package is not
            extra_instruction = f"Install traceloop-sdk with the '{extra_name}' extra: pip install 'traceloop-sdk[{extra_name}]'" if extra_name else "Install the appropriate instrumentation package."
            logging.info(
                f"{library_name} SDK is installed but {instrumentor_package} is not. "
                f"{extra_instruction}"
            )
        
        return True
    
    except Exception as e:
        log_func = logging.warning if use_warning_log else logging.error
        log_func(f"Error initializing {library_name} instrumentor: {e}")
        Telemetry().log_exception(e)
    
    return False


def metrics_common_attributes():
    common_attributes = {}
    workflow_name = get_value("workflow_name")
    if workflow_name is not None:
        common_attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    entity_name = get_value("entity_name")
    if entity_name is not None:
        common_attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name

    association_properties = get_value("association_properties")
    if association_properties is not None:
        for key, value in association_properties.items():
            common_attributes[
                f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.{key}"
            ] = value

    return common_attributes
