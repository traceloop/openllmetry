"""Regression coverage for OpenTelemetry zero-code dependency discovery.

The distro auto-loader examines the package's `instruments` extra *before*
loading the entry point. This must match the runtime dependency declared by
LangchainInstrumentor, even when the top-level langchain package is absent.
"""

from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from opentelemetry.instrumentation.langchain import LangchainInstrumentor


def test_instruments_extra_matches_langchain_core_runtime_gate():
    declared = requires("opentelemetry-instrumentation-langchain")
    assert declared is not None, "Instrumentation distribution metadata is missing"

    # Include requirements enabled specifically by the instruments extra.
    extra = [
        Requirement(item)
        for item in declared
        if Requirement(item).marker is not None
        and Requirement(item).marker.evaluate({"extra": "instruments"})
        and not Requirement(item).marker.evaluate({"extra": ""})
    ]
    runtime = [
        Requirement(item)
        for item in LangchainInstrumentor().instrumentation_dependencies()
    ]

    assert len(extra) == len(runtime) == 1
    assert canonicalize_name(extra[0].name) == "langchain-core"
    assert canonicalize_name(extra[0].name) == canonicalize_name(runtime[0].name)
    assert extra[0].specifier == runtime[0].specifier
