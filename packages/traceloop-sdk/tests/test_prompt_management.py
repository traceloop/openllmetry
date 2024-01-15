import json
import pytest
from openai import OpenAI
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


@pytest.fixture
def openai_client():
    return OpenAI()


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
        open_ai_span.attributes["llm.prompts.0.content"]
        == "Tell me a joke about OpenTelemetry, pirate style"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")
    assert open_ai_span.attributes.get("traceloop.prompt.key") == "joke_generator"
