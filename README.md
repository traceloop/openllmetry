<p align="center">
<a href="https://www.traceloop.com/openllmetry#gh-light-mode-only">
<img width="600" src="https://raw.githubusercontent.com/traceloop/openllmetry/main/img/logo-light.png">
</a>
<a href="https://www.traceloop.com/openllmetry#gh-dark-mode-only">
<img width="600" src="https://raw.githubusercontent.com/traceloop/openllmetry/main/img/logo-dark.png">
</a>
</p>
<p align="center">
  <p align="center">Open-source observability for your LLM application</p>
</p>
<h4 align="center">
    <a href="https://traceloop.com/docs/openllmetry/getting-started-python"><strong>Get started »</strong></a>
    <br />
    <br />
  <a href="https://traceloop.com/slack">Slack</a> |
  <a href="https://traceloop.com/docs/openllmetry/introduction">Docs</a> |
  <a href="https://www.traceloop.com/openllmetry">Website</a>
</h4>

<h4 align="center">
  <a href="https://github.com/traceloop/openllmetry/releases">
    <img src="https://img.shields.io/github/release/traceloop/openllmetry">
  </a>
  <a href="https://pepy.tech/project/opentelemetry-instrumentation-openai">
  <img src="https://static.pepy.tech/badge/opentelemetry-instrumentation-openai/month">
  </a>
   <a href="https://github.com/traceloop/openllmetry/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache 2.0-blue.svg" alt="OpenLLMetry is released under the Apache-2.0 License">
  </a>
  <a href="https://github.com/traceloop/openllmetry/actions/workflows/ci.yml">
  <img src="https://github.com/traceloop/openllmetry/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://github.com/traceloop/openllmetry/issues">
    <img src="https://img.shields.io/github/commit-activity/m/traceloop/openllmetry" alt="git commit activity" />
  </a>
  <a href="https://www.ycombinator.com/companies/traceloop"><img src="https://img.shields.io/website?color=%23f26522&down_message=Y%20Combinator&label=Backed&logo=ycombinator&style=flat-square&up_message=Y%20Combinator&url=https%3A%2F%2Fwww.ycombinator.com"></a>
  <a href="https://github.com/traceloop/openllmetry/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs welcome!" />
  </a>
  <a href="https://traceloop.com/slack">
    <img src="https://img.shields.io/badge/chat-on%20Slack-blueviolet" alt="Slack community channel" />
  </a>
  <a href="https://twitter.com/traceloopdev">
    <img src="https://img.shields.io/badge/follow-%40traceloopdev-1DA1F2?logo=twitter&style=social" alt="Traceloop Twitter" />
  </a>
</h4>

**🎉 New**:
Our semantic conventions are now part of OpenTelemetry! Join the [discussion](https://github.com/open-telemetry/community/blob/1c71595874e5d125ca92ec3b0e948c4325161c8a/projects/llm-semconv.md) and help us shape the future of LLM observability.

Looking for the JS/TS version? Check out [OpenLLMetry-JS](https://github.com/traceloop/openllmetry-js).

OpenLLMetry is a set of extensions built on top of [OpenTelemetry](https://opentelemetry.io/) that gives you complete observability over your LLM application. Because it uses OpenTelemetry under the hood, [it can be connected to your existing observability solutions](https://www.traceloop.com/docs/openllmetry/integrations/introduction) - Datadog, Honeycomb, and others.

It's built and maintained by Traceloop under the Apache 2.0 license.

The repo contains standard OpenTelemetry instrumentations for LLM providers and Vector DBs, as well as a Traceloop SDK that makes it easy to get started with OpenLLMetry, while still outputting standard OpenTelemetry data that can be connected to your observability stack.
If you already have OpenTelemetry instrumented, you can just add any of our instrumentations directly.

## 🚀 Getting Started

The easiest way to get started is to use our SDK.
For a complete guide, go to our [docs](https://traceloop.com/docs/openllmetry/getting-started-python).

Install the SDK:

```bash
pip install traceloop-sdk
```

Then, to start instrumenting your code, just add this line to your code:

```python
from traceloop.sdk import Traceloop

Traceloop.init()
```

That's it. You're now tracing your code with OpenLLMetry!
If you're running this locally, you may want to disable batch sending, so you can see the traces immediately:

```python
Traceloop.init(disable_batch=True)
```

## ⏫ Supported (and tested) destinations

- ✅ [Traceloop](https://www.traceloop.com/docs/openllmetry/integrations/traceloop)
- ✅ [Axiom](https://www.traceloop.com/docs/openllmetry/integrations/axiom)
- ✅ [Azure Application Insights](https://www.traceloop.com/docs/openllmetry/integrations/azure)
- ✅ [Braintrust](https://www.traceloop.com/docs/openllmetry/integrations/braintrust)
- ✅ [Dash0](https://www.traceloop.com/docs/openllmetry/integrations/dash0)
- ✅ [Datadog](https://www.traceloop.com/docs/openllmetry/integrations/datadog)
- ✅ [Dynatrace](https://www.traceloop.com/docs/openllmetry/integrations/dynatrace)
- ✅ [Google Cloud](https://www.traceloop.com/docs/openllmetry/integrations/gcp)
- ✅ [Grafana](https://www.traceloop.com/docs/openllmetry/integrations/grafana)
- ✅ [Highlight](https://www.traceloop.com/docs/openllmetry/integrations/highlight)
- ✅ [Honeycomb](https://www.traceloop.com/docs/openllmetry/integrations/honeycomb)
- ✅ [HyperDX](https://www.traceloop.com/docs/openllmetry/integrations/hyperdx)
- ✅ [IBM Instana](https://www.traceloop.com/docs/openllmetry/integrations/instana)
- ✅ [KloudMate](https://www.traceloop.com/docs/openllmetry/integrations/kloudmate)
- ✅ [Laminar](https://www.traceloop.com/docs/openllmetry/integrations/laminar)
- ✅ [New Relic](https://www.traceloop.com/docs/openllmetry/integrations/newrelic)
- ✅ [OpenTelemetry Collector](https://www.traceloop.com/docs/openllmetry/integrations/otel-collector)
- ✅ [Oracle Cloud](https://www.traceloop.com/docs/openllmetry/integrations/oraclecloud)
- ✅ [Scorecard](https://www.traceloop.com/docs/openllmetry/integrations/scorecard)
- ✅ [Service Now Cloud Observability](https://www.traceloop.com/docs/openllmetry/integrations/service-now)
- ✅ [SigNoz](https://www.traceloop.com/docs/openllmetry/integrations/signoz)
- ✅ [Sentry](https://www.traceloop.com/docs/openllmetry/integrations/sentry)
- ✅ [Splunk](https://www.traceloop.com/docs/openllmetry/integrations/splunk)
- ✅ [Tencent Cloud](https://www.traceloop.com/docs/openllmetry/integrations/tencent)

See [our docs](https://traceloop.com/docs/openllmetry/integrations/exporting) for instructions on connecting to each one.

## 🪗 What do we instrument?

OpenLLMetry can instrument everything that [OpenTelemetry already instruments](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation) - so things like your DB, API calls, and more. On top of that, we built a set of custom extensions that instrument things like your calls to OpenAI or Anthropic, or your Vector DB like Chroma, Pinecone, Qdrant or Weaviate.

- ✅ [Aleph Alpha](https://www.aleph-alpha.com/)
- ✅ [Anthropic](https://www.anthropic.com/)
- ✅ [Bedrock (AWS)](https://aws.amazon.com/bedrock/)
- ✅ [Cohere](https://cohere.com/)
- ✅ [Google Generative AI (Gemini)](https://ai.google/)
- ✅ [Groq](https://groq.com/)
- ✅ [HuggingFace](https://huggingface.co/)
- ✅ [IBM Watsonx AI](https://www.ibm.com/watsonx)
- ✅ [Mistral AI](https://mistral.ai/)
- ✅ [Ollama](https://ollama.com/)
- ✅ [OpenAI / Azure OpenAI](https://openai.com/)
- ✅ [Replicate](https://replicate.com/)
- ✅ [SageMaker (AWS)](https://aws.amazon.com/sagemaker/)
- ✅ [Together AI](https://together.xyz/)
- ✅ [Vertex AI (GCP)](https://cloud.google.com/vertex-ai)
- ✅ [WRITER](https://writer.com/)

### Vector DBs

- ✅ [Chroma](https://www.trychroma.com/)
- ✅ [LanceDB](https://lancedb.com/)
- ✅ [Marqo](https://marqo.ai/)
- ✅ [Milvus](https://milvus.io/)
- ✅ [Pinecone](https://www.pinecone.io/)
- ✅ [Qdrant](https://qdrant.tech/)
- ✅ [Weaviate](https://weaviate.io/)

### Frameworks

- ✅ [Agno](https://github.com/agno-agi/agno)
- ✅ [AWS Strands](https://strandsagents.com/) (built-in OTEL support)
- ✅ [CrewAI](https://docs.crewai.com/introduction)
- ✅ [Haystack](https://haystack.deepset.ai/integrations/traceloop)
- ✅ [LangChain](https://python.langchain.com/docs/introduction/)
- ✅ [Langflow](https://docs.langflow.org/)
- ✅ [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- ✅ [LiteLLM](https://docs.litellm.ai/docs/observability/opentelemetry_integration)
- ✅ [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html#openllmetry)
- ✅ [OpenAI Agents](https://openai.github.io/openai-agents-python/)

### Protocol

- ✅ [MCP](https://modelcontextprotocol.io/)

## 🔎 Telemetry

We no longer log or collect any telemetry in the SDK or in the instrumentations. Make sure to bump to v0.49.2 and above.

### Why we collect telemetry

- The primary purpose is to detect exceptions within instrumentations. Since LLM providers frequently update their APIs, this helps us quickly identify and fix any breaking changes.
- We only collect anonymous data, with no personally identifiable information. You can view exactly what data we collect in our [Privacy documentation](https://www.traceloop.com/docs/openllmetry/privacy/telemetry).
- Telemetry is only collected in the SDK. If you use the instrumentations directly without the SDK, no telemetry is collected.

## 🌱 Contributing

Whether big or small, we love contributions ❤️ Check out our guide to see how to [get started](https://traceloop.com/docs/openllmetry/contributing/overview).

Not sure where to get started? You can:

- [Book a free pairing session with one of our teammates](mailto:nir@traceloop.com?subject=Pairing%20session&body=I'd%20like%20to%20do%20a%20pairing%20session!)!
- Join our <a href="https://traceloop.com/slack">Slack</a>, and ask us any questions there.

## 💚 Community & Support

- [Slack](https://traceloop.com/slack) (For live discussion with the community and the Traceloop team)
- [GitHub Discussions](https://github.com/traceloop/openllmetry/discussions) (For help with building and deeper conversations about features)
- [GitHub Issues](https://github.com/traceloop/openllmetry/issues) (For any bugs and errors you encounter using OpenLLMetry)
- [Twitter](https://twitter.com/traceloopdev) (Get news fast)

## 🙏 Special Thanks

To @patrickdebois, who [suggested the great name](https://x.com/patrickdebois/status/1695518950715473991?s=46&t=zn2SOuJcSVq-Pe2Ysevzkg) we're now using for this repo!

## 💫 Contributors

<a href="https://github.com/traceloop/openllmetry/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=traceloop/openllmetry"/>
</a>
