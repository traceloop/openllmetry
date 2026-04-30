# sample-app

Sample applications for validating `traceloop-sdk` and the instrumentation packages locally.

## Environment

Copy `.env.example` to `.env` and fill in the provider you want to use.

### Atlas Cloud

Atlas Cloud exposes an OpenAI-compatible LLM endpoint at `https://api.atlascloud.ai/v1`, so it works with the standard OpenAI Python SDK.

Example `.env` values:

```bash
LLM_PROVIDER=atlascloud
ATLASCLOUD_API_KEY=apikey-your-atlascloud-api-key-here
ATLASCLOUD_BASE_URL=https://api.atlascloud.ai/v1
ATLASCLOUD_MODEL=deepseek-ai/DeepSeek-V3-0324
```

Run the example:

```bash
uv run python -m sample_app.atlascloud_openai
```

## Provider Helper

`sample_app/provider_factory.py` contains a lightweight OpenAI-compatible provider factory.

- `openai`: reads `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `OPENAI_MODEL`
- `atlascloud`: reads `ATLASCLOUD_API_KEY`, `ATLASCLOUD_BASE_URL`, and `ATLASCLOUD_MODEL`

This keeps sample scripts simple while making it easy to switch between upstream OpenAI and OpenAI-compatible providers.
