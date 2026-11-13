"""Protect Writer zero-code instrumentation metadata from wrong SDK gates.

The Writer Python SDK distribution is named writer-sdk, but imported as
`writerai`. OpenTelemetry distro checks the optional `instruments` extra
before importing or installing the instrumentor.
"""

from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from opentelemetry.instrumentation.writer import WriterInstrumentor


def test_instruments_extra_matches_writer_sdk_runtime_gate():
    declared = requires("opentelemetry-instrumentation-writer")
    assert declared is not None, "Instrumentation distribution metadata is missing"

    extra = [
        Requirement(item)
        for item in declared
        if Requirement(item).marker is not None
        and Requirement(item).marker.evaluate({"extra": "instruments"})
        and not Requirement(item).marker.evaluate({"extra": ""})
    ]
    runtime = [
        Requirement(item)
        for item in WriterInstrumentor().instrumentation_dependencies()
    ]

    assert len(extra) == len(runtime) == 1
    assert canonicalize_name(extra[0].name) == "writer-sdk"
    assert canonicalize_name(extra[0].name) == canonicalize_name(runtime[0].name)
    assert extra[0].specifier == runtime[0].specifier
