import json

import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from traceloop.sdk.prompts import get_prompt
from traceloop.sdk.prompts.client import PromptRegistryClient

prompts_json = """
{
  "prompts": [
    {
      "id": "clpuabf3h0002kdmoz3k3hesu",
      "created_at": "2023-12-06T21:31:58.637Z",
      "updated_at": "2023-12-06T21:31:58.637Z",
      "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
      "key": "joke_generator",
      "versions": [
        {
          "id": "clpuabf780000106giwxkinpq",
          "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
          "created_at": "2023-12-06T21:31:58.772Z",
          "updated_at": "2024-01-07T08:25:17.194Z",
          "author": "tomer f",
          "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
          "hash": "8296b2fe",
          "version": 0,
          "name": "dfgdfg",
          "provider": "openai",
          "templating_engine": "jinja2",
          "messages": [
            {
              "role": "user",
              "index": 0,
              "template": "Tell me a joke about OpenTelemetry, {{style}} style",
              "variables": [
                "style"
              ]
            }
          ],
          "llm_config": {
            "mode": "chat",
            "stop": [],
            "model": "gpt-3.5-turbo",
            "top_p": 1,
            "temperature": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0
          },
          "publishable": true
        }
      ],
      "target": {
        "id": "clr38ainb0001mrby3q81lec6",
        "updated_at": "2024-01-07T08:24:55.271Z",
        "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
        "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
        "environment": "dev",
        "version": "clpuabf780000106giwxkinpq"
      }
    }
  ],
  "environment": "dev"
}
"""

prompts_with_tools_json = """
{
  "prompts": [
    {
      "id": "clpuabf3h0002kdmoz3k3hesu",
      "created_at": "2023-12-06T21:31:58.637Z",
      "updated_at": "2023-12-06T21:31:58.637Z",
      "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
      "key": "joke_generator",
      "versions": [
        {
          "id": "clpuabf780000106giwxkinpq",
          "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
          "created_at": "2023-12-06T21:31:58.772Z",
          "updated_at": "2024-01-07T08:25:17.194Z",
          "author": "tomer f",
          "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
          "hash": "8296b2fe",
          "version": 0,
          "name": "dfgdfg",
          "provider": "openai",
          "templating_engine": "jinja2",
          "messages": [
            {
              "role": "user",
              "index": 0,
              "template": "Tell me a joke about OpenTelemetry, {{style}} style",
              "variables": [
                "style"
              ]
            }
          ],
          "llm_config": {
            "mode": "chat",
            "stop": [],
            "model": "gpt-3.5-turbo",
            "top_p": 1,
            "temperature": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tool_choice": "required",
            "tools": [
              {
                "type": "function",
                "function": {
                  "name": "get_joke",
                  "description": "Get a joke about OpenTelemetry",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "style": {
                        "type": "string",
                        "description": "The style of the joke"
                      }
                    },
                    "required": ["style"]
                  }
                }
              }
            ]
          },
          "publishable": true
        }
      ],
      "target": {
        "id": "clr38ainb0001mrby3q81lec6",
        "updated_at": "2024-01-07T08:24:55.271Z",
        "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
        "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
        "environment": "dev",
        "version": "clpuabf780000106giwxkinpq"
      }
    }
  ],
  "environment": "dev"
}
"""

prompts_with_response_format_json = """

{
  "prompts": [
    {
      "id": "clpuabf3h0002kdmoz3k3hesu",
      "created_at": "2023-12-06T21:31:58.637Z",
      "updated_at": "2023-12-06T21:31:58.637Z",
      "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
      "key": "joke_generator",
      "versions": [
        {
          "id": "clpuabf780000106giwxkinpq",
          "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
          "created_at": "2023-12-06T21:31:58.772Z",
          "updated_at": "2024-01-07T08:25:17.194Z",
          "author": "tomer f",
          "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
          "hash": "8296b2fe",
          "version": 0,
          "name": "dfgdfg",
          "provider": "openai",
          "templating_engine": "jinja2",
          "messages": [
            {
              "role": "user",
              "index": 0,
              "template": "Tell me a joke about OpenTelemetry, {{style}} style",
              "variables": [
                "style"
              ]
            }
          ],
          "llm_config": {
            "mode": "chat",
            "stop": [],
            "model": "gpt-4o-mini",
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 1,
            "response_format": {
              "type": "json_schema",
              "json_schema": {
                "name": "joke",
                "schema": {
                  "type": "object",
                  "properties": {
                    "joke": {
                      "type": "string",
                      "description": "The joke"
                    }
                  }
                }
              }
            }
          },
          "publishable": true
        }
      ],
      "target": {
        "id": "clr38ainb0001mrby3q81lec6",
        "updated_at": "2024-01-07T08:24:55.271Z",
        "prompt_id": "clpuabf3h0002kdmoz3k3hesu",
        "org_id": "82930720-5041-4deb-b1a5-705cc46bda3c",
        "environment": "dev",
        "version": "clpuabf780000106giwxkinpq"
      }
    }
  ]
}
"""


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_prompt_management(exporter, openai_client):
    PromptRegistryClient()._registry.load(prompts_json=json.loads(prompts_json))
    prompt_args = get_prompt(key="joke_generator", variables={"style": "pirate"})
    openai_client.chat.completions.create(**prompt_args)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about OpenTelemetry, pirate style"
    )
    assert open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert open_ai_span.attributes.get("traceloop.prompt.key") == "joke_generator"


@pytest.mark.vcr
def test_prompt_management_with_tools(exporter, openai_client):
    PromptRegistryClient()._registry.load(
        prompts_json=json.loads(prompts_with_tools_json)
    )
    prompt_args = get_prompt(key="joke_generator", variables={"style": "pirate"})
    openai_client.chat.completions.create(**prompt_args)

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    completion = open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"
    )
    assert completion == "get_joke"


@pytest.mark.vcr
def test_prompt_management_with_response_format(exporter, openai_client):
    PromptRegistryClient()._registry.load(
        prompts_json=json.loads(prompts_with_response_format_json)
    )
    prompt_args = get_prompt(key="joke_generator", variables={"style": "pirate"})
    openai_client.chat.completions.create(**prompt_args)

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    completion = open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    )
    try:
        json.loads(completion)
    except json.JSONDecodeError:
        pytest.fail("Response is not valid JSON")
    assert True
