import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai


def test_openai_streaming_sample_requests_a_supported_model(monkeypatch, capsys):
    request_bodies = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        request_bodies.append(json.loads(request.content))
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": "A story"}, "finish_reason": None}],
        }
        return httpx.Response(
            200,
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    # Sample startup should not register real telemetry or contact OpenAI.
    sdk = SimpleNamespace(Traceloop=SimpleNamespace(init=lambda **kwargs: None))
    decorators = SimpleNamespace(workflow=lambda **kwargs: lambda function: function)
    monkeypatch.setitem(sys.modules, "traceloop.sdk", sdk)
    monkeypatch.setitem(sys.modules, "traceloop.sdk.decorators", decorators)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = openai.OpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
        sample = Path(__file__).resolve().parents[1] / "sample_app" / "openai_streaming.py"
        runpy.run_path(str(sample))

    assert len(request_bodies) == 1
    body = request_bodies[0]
    assert body["model"] == "gpt-5.6-sol"
    assert body["reasoning_effort"] == "none"
    assert body["stream"] is True
    assert capsys.readouterr().out == "A story\n"
