from __future__ import annotations

from opentelemetry.instrumentation.openai.v1.assistant_async_wrappers import (
    aassistants_create_wrapper,
    amessages_create_wrapper,
    amessages_list_wrapper,
    aruns_create_and_stream_wrapper,
    aruns_create_wrapper,
    aruns_retrieve_wrapper,
)
from opentelemetry.instrumentation.openai.v1.assistant_wrappers import messages_create_wrapper
from opentelemetry.instrumentation.utils import unwrap


def instrument_additional_beta_safety_surfaces(instrumentor, tracer):
    instrumentor._try_wrap(
        "openai.resources.beta.assistants",
        "AsyncAssistants.create",
        aassistants_create_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.create",
        aruns_create_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.retrieve",
        aruns_retrieve_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.create_and_stream",
        aruns_create_and_stream_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.messages",
        "Messages.create",
        messages_create_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.messages",
        "AsyncMessages.create",
        amessages_create_wrapper(tracer),
    )
    instrumentor._try_wrap(
        "openai.resources.beta.threads.messages",
        "AsyncMessages.list",
        amessages_list_wrapper(tracer),
    )


def uninstrument_additional_beta_safety_surfaces():
    unwrap("openai.resources.beta.assistants", "AsyncAssistants.create")
    unwrap("openai.resources.beta.threads.runs", "AsyncRuns.create")
    unwrap("openai.resources.beta.threads.runs", "AsyncRuns.retrieve")
    unwrap("openai.resources.beta.threads.runs", "AsyncRuns.create_and_stream")
    unwrap("openai.resources.beta.threads.messages", "Messages.create")
    unwrap("openai.resources.beta.threads.messages", "AsyncMessages.create")
    unwrap("openai.resources.beta.threads.messages", "AsyncMessages.list")
