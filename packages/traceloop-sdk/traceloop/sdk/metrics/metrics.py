from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider

from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, MetricExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GRPCExporter
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HTTPExporter
)
from typing import Dict


class MetricsWrapper(object):
    resource_attributes: dict = {}
    endpoint: str = None
    # if it needs headers?
    headers: Dict[str, str] = {}

    def __new__(cls, exporter: MetricExporter = None) -> "MetricsWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(MetricsWrapper, cls).__new__(cls)
            if not MetricsWrapper.endpoint:
                return obj

            obj.__metrics_exporter: MetricExporter = (
                exporter if exporter
                else init_metrics_exporter(
                    MetricsWrapper.endpoint, MetricsWrapper.headers
                )
            )

            obj.__metrics_provider: MeterProvider = init_metrics_provider(obj.__metrics_exporter,
                                                                          MetricsWrapper.resource_attributes)

        return cls.instance

    @staticmethod
    def set_static_params(
            resource_attributes: dict,
            endpoint: str,
            headers: Dict[str, str],
    ) -> None:
        MetricsWrapper.resource_attributes = resource_attributes
        MetricsWrapper.endpoint = endpoint
        MetricsWrapper.headers = headers


def init_metrics_exporter(endpoint: str, headers: Dict[str, str]) -> MetricExporter:
    if "http" in endpoint.lower() or "https" in endpoint.lower():
        return HTTPExporter(endpoint=endpoint)
    else:
        return GRPCExporter(endpoint=endpoint)


def init_metrics_provider(exporter: MetricExporter,
                          resource_attributes: dict = None) -> MeterProvider:
    resource = Resource.create(resource_attributes) if resource_attributes else Resource.create()
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    return provider
