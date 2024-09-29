import os
import sys
import logging
from pathlib import Path

from typing import Optional, Set
from colorama import Fore
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.util.re import parse_env_headers

from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.metrics.metrics import MetricsWrapper
from traceloop.sdk.telemetry import Telemetry
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.config import (
    is_content_tracing_enabled,
    is_tracing_enabled,
    is_metrics_enabled,
)
from traceloop.sdk.fetcher import Fetcher
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_association_properties,
    set_external_prompt_tracing_context,
)
from typing import Dict


class Traceloop:
    AUTO_CREATED_KEY_PATH = str(
        Path.home() / ".cache" / "traceloop" / "auto_created_key"
    )
    AUTO_CREATED_URL = str(Path.home() / ".cache" / "traceloop" / "auto_created_url")

    __tracer_wrapper: TracerWrapper
    __fetcher: Fetcher = None

    @staticmethod
    def init(
        app_name: Optional[str] = sys.argv[0],
        api_endpoint: str = "https://api.traceloop.com",
        api_key: str = None,
        headers: Dict[str, str] = {},
        disable_batch=False,
        exporter: SpanExporter = None,
        metrics_exporter: MetricExporter = None,
        metrics_headers: Dict[str, str] = None,
        processor: SpanProcessor = None,
        propagator: TextMapPropagator = None,
        traceloop_sync_enabled: bool = False,
        should_enrich_metrics: bool = True,
        resource_attributes: dict = {},
        instruments: Optional[Set[Instruments]] = None,
        trace_content: bool = True # trace_content controls whether instrumentations are sending sensitive content
    ) -> None:
        Telemetry()

        api_endpoint = os.getenv("TRACELOOP_BASE_URL") or api_endpoint
        api_key = os.getenv("TRACELOOP_API_KEY") or api_key

        if (
            traceloop_sync_enabled
            and api_endpoint.find("traceloop.com") != -1
            and api_key
            and not exporter
            and not processor
        ):
            Traceloop.__fetcher = Fetcher(base_url=api_endpoint, api_key=api_key)
            Traceloop.__fetcher.run()
            print(
                Fore.GREEN + "Traceloop syncing configuration and prompts" + Fore.RESET
            )

        if not is_tracing_enabled():
            print(Fore.YELLOW + "Tracing is disabled" + Fore.RESET)
            return

        enable_content_tracing = trace_content
        
        # if trace content is false, disable sensitive content in instruments
        if enable_content_tracing is False:
            # pass configuration through all instruments
            if instruments is not None:
                for instrument in instruments:
                    if instrument == Instruments.OPENAI:
                        # set instrumentation
                        try:
                            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
                            
                            # set sensitive data off
                            instrumentor = OpenAIInstrumentor(
                                enrich_assistant = False,
                                enrich_token_usage = False,
                            )
                            
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                            
                        except Exception as e:
                            logging.error(f"Error initializing OpenAI instrumentor: {e}")
                            Telemetry().log_exception(e)
                            
                    elif instrument == Instruments.ANTHROPIC:
                        try:
                            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

                            instrumentor = AnthropicInstrumentor(
                                enrich_token_usage = False,
                            )
                            
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Anthropic instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.COHERE:
                        try:
                            from opentelemetry.instrumentation.cohere import CohereInstrumentor
                            
                            instrumentor = CohereInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Cohere instrumentor: {e}")
                            Telemetry().log_exception(e)
                            
                    elif instrument == Instruments.PINECONE:
                        try:
                            from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

                            instrumentor = PineconeInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Pinecone instrumentor: {e}")
                            Telemetry().log_exception(e)

                    elif instrument == Instruments.CHROMA:
                        try:
                            from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

                            instrumentor = ChromaInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Chroma instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.GOOGLE_GENERATIVEAI:
                        try:
                            from opentelemetry.instrumentation.google_generativeai import (
                                GoogleGenerativeAiInstrumentor,
                            )

                            instrumentor = GoogleGenerativeAiInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Google Generative AI instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.LANGCHAIN:
                        try:
                            from opentelemetry.instrumentation.langchain import LangchainInstrumentor

                            instrumentor = LangchainInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Langchain instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.MISTRAL:
                        try:
                            from opentelemetry.instrumentation.MISTRAL import MistralAiInstrumentor

                            instrumentor = MistralAiInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Mistral instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.OLLAMA:
                        try:
                            from opentelemetry.instrumentation.pinecone import OllamaInstrumentor

                            instrumentor = OllamaInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Ollama instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.LLAMA_INDEX:
                        try:
                            from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

                            instrumentor = LlamaIndexInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Llama Index instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.MILVUS:
                        try:
                            from opentelemetry.instrumentation.milvus import MilvusInstrumentor

                            instrumentor = MilvusInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Milvus instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.TRANSFORMERS:
                        try:
                            from opentelemetry.instrumentation.transformers import (
                                TransformersInstrumentor,
                            )

                            instrumentor = TransformersInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Transformers instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.TOGETHER:
                        try:
                            from opentelemetry.instrumentation.together import TogetherAiInstrumentor

                            instrumentor = TogetherAiInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Together instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.REQUESTS:
                        try:
                            from opentelemetry.instrumentation.requests import RequestsInstrumentor

                            instrumentor = RequestsInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Requests instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.URLLIB3:
                        try:
                            from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

                            instrumentor = URLLib3Instrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing urllib3 instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.PYMYSQL:
                        try:
                            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

                            instrumentor = SQLAlchemyInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing SQLAlchemy instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.BEDROCK:
                        try:
                            from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

                            instrumentor = BedrockInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Bedrock instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.REPLICATE:
                        try:
                            from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

                            instrumentor = ReplicateInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Replicate instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.VERTEXAI:
                        try:
                            from opentelemetry.instrumentation.vertexai import VertexAiInstrumentor

                            instrumentor = VertexAiInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Vertex AI instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.WATSONX:
                        try:
                            from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

                            instrumentor = WatsonxInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing WatsonX instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.WEAVIATE:
                        try:
                            from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

                            instrumentor = WeaviateInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Weaviate instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.ALEPHALPHA:
                        try:
                            from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

                            instrumentor = AlephAlphaInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing AlephAlpha instrumentor: {e}")
                            Telemetry().log_exception(e)
                    elif instrument == Instruments.MARQO:
                        try:
                            from opentelemetry.instrumentation.marqo import MarqoInstrumentor

                            instrumentor = MarqoInstrumentor(
                                
                            )
                            if not instrumentor.is_instrumented_by_opentelemetry:
                                instrumentor.instrument()
                        except Exception as e:
                            logging.error(f"Error initializing Marqo instrumentor: {e}")
                            Telemetry().log_exception(e)
                    else:
                        print(
                            Fore.RED
                            + "Warning: "
                            + instrument
                            + " instrumentation does not exist."
                        )
                        print(
                            # good error message
                            "Usage:\n"
                            + "from traceloop.sdk.instruments import Instruments\n"
                        )
                        print(Fore.RESET)

        if exporter or processor:
            print(Fore.GREEN + "Traceloop exporting traces to a custom exporter")

        headers = os.getenv("TRACELOOP_HEADERS") or headers

        if isinstance(headers, str):
            headers = parse_env_headers(headers)

        if (
            not exporter
            and not processor
            and api_endpoint == "https://api.traceloop.com"
            and not api_key
        ):
            print(
                Fore.RED
                + "Error: Missing Traceloop API key,"
                + " go to https://app.traceloop.com/settings/api-keys to create one"
            )
            print("Set the TRACELOOP_API_KEY environment variable to the key")
            print(Fore.RESET)
            return

        if not exporter and not processor and headers:
            print(
                Fore.GREEN
                + f"Traceloop exporting traces to {api_endpoint}, authenticating with custom headers"
            )

        if api_key and not exporter and not processor and not headers:
            print(
                Fore.GREEN
                + f"Traceloop exporting traces to {api_endpoint} authenticating with bearer token"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

        print(Fore.RESET)

        # Tracer init
        resource_attributes.update({SERVICE_NAME: app_name})
        TracerWrapper.set_static_params(
            resource_attributes, enable_content_tracing, api_endpoint, headers
        )
        Traceloop.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch,
            processor=processor,
            propagator=propagator,
            exporter=exporter,
            should_enrich_metrics=should_enrich_metrics,
            image_uploader=ImageUploader(api_endpoint, api_key),
            instruments=instruments,
        )

        if not metrics_exporter and exporter:
            return

        metrics_endpoint = os.getenv("TRACELOOP_METRICS_ENDPOINT") or api_endpoint
        metrics_headers = (
            os.getenv("TRACELOOP_METRICS_HEADERS") or metrics_headers or headers
        )

        if not is_metrics_enabled() or not metrics_exporter and exporter:
            print(Fore.YELLOW + "Metrics are disabled" + Fore.RESET)
            return

        MetricsWrapper.set_static_params(
            resource_attributes, metrics_endpoint, metrics_headers
        )

        Traceloop.__metrics_wrapper = MetricsWrapper(exporter=metrics_exporter)

    def set_association_properties(properties: dict) -> None:
        set_association_properties(properties)

    def set_prompt(template: str, variables: dict, version: int):
        set_external_prompt_tracing_context(template, variables, version)

    def report_score(
        association_property_name: str,
        association_property_id: str,
        score: float,
    ):
        if not Traceloop.__fetcher:
            print(
                Fore.RED
                + "Error: Cannot report score. Missing Traceloop API key,"
                + " go to https://app.traceloop.com/settings/api-keys to create one"
            )
            print("Set the TRACELOOP_API_KEY environment variable to the key")
            print(Fore.RESET)
            return

        Traceloop.__fetcher.post(
            "score",
            {
                "entity_name": f"traceloop.association.properties.{association_property_name}",
                "entity_id": association_property_id,
                "score": score,
            },
        )
