<p align="center">
<a href="https://www.traceloop.com/">
<img width="300" src="https://raw.githubusercontent.com/traceloop/openllmetry/main/img/logo.png">
</a>
</p>
<h1 align="center">Open LLMetry</h1>
<p align="center">
  <p align="center">Open-source observability for your LLM application</p>
</p>
<h4 align="center">
    <a href="https://traceloop.com/docs/python-sdk/getting-started"><strong>Get started »</strong></a>
    <br />
    <br />
  <a href="https://join.slack.com/t/traceloopcommunity/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g">Slack</a> |
  <a href="https://traceloop.com/docs/python-sdk/introduction">Docs</a> |
  <a href="https://www.traceloop.com">Website</a>
</h4>

<h4 align="center">
   <a href="https://github.com/traceloop/openllmetry/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache 2.0-blue.svg" alt="OpenLLMetry is released under the Apache-2.0 License">
  </a>
  <a href="https://www.ycombinator.com/companies/traceloop"><img src="https://img.shields.io/website?color=%23f26522&down_message=Y%20Combinator&label=Backed&logo=ycombinator&style=flat-square&up_message=Y%20Combinator&url=https%3A%2F%2Fwww.ycombinator.com"></a>
  <a href="https://github.com/traceloop/openllmetry/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs welcome!" />
  </a>
  <a href="https://github.com/traceloop/openllmetry/issues">
    <img src="https://img.shields.io/github/commit-activity/m/traceloop/openllmetry" alt="git commit activity" />
  </a>
  <a href="https://join.slack.com/t/traceloopcommunity/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g">
    <img src="https://img.shields.io/badge/chat-on%20Slack-blueviolet" alt="Slack community channel" />
  </a>
  <a href="https://twitter.com/traceloopdev">
    <img src="https://img.shields.io/badge/follow-%40traceloopdev-1DA1F2?logo=twitter&style=social" alt="Traceloop Twitter" />
  </a>
</h4>

OpenLLMetry is a set of extensions built on top of [OpenTelemetry](https://opentelemetry.io/) that gives you complete observability over your LLM application. Because it uses OpenTelemetry under the hood, it can be connected to your existing observability solutions - Datadog, Honeycomb, and others.

It's built and maintained by Traceloop under the Apache 2.0 license.

The repo contains standard OpenTelemetry instrumentations for LLM providers and Vector DBs, as well as a Traceloop SDK that makes it easy to get started with OpenLLMetry, while still outputting standard OpenTelemetry data that can be connected to your observability stack.
If you already have OpenTelemetry instrumented, you can just add any of our instrumentations directly.

## 🚀 Getting Started

The easiest to get started is to use our SDK.
For a complete guide, go to our [docs](https://traceloop.com/docs/python-sdk/getting-started).

Install the SDK into your project:

```python
pip install traceloop-sdk
```

To start instrumenting your code, just add this line to your code:

```python
Traceloop.init(app_name="your_app_name")
```

Next, you need to decide where to export the traces to. You can choose between the following providers:
- [x] [Traceloop](#traceloop)
- [x] [Datadog](#datadog)
- [x] [Honeycomb](#honeycomb)
- [x] [SigNoz](#signoz)

### Traceloop

Open an account at [app.traceloop.com](https://app.traceloop.com), and then set an env var with your API key:

```bash
TRACELOOP_API_KEY=your_api_key
```

### Datadog

With datadog, there are 2 options - you can either export directly to a Datadog Agent in your cluster, or through an OpenTelemetry Collector (which requires that you deploy one in your cluster).
See also [Datadog's documentation](https://docs.datadoghq.com/opentelemetry/).

Exporting directly to an agent is easiest. To do that, first enable the OTLP GRPC collector in your agent configuration. This depends on how you deployed your Datadog agent. For example, if you've used a Helm chart, you can add the following to your `values.yaml` (see [this](https://docs.datadoghq.com/opentelemetry/otlp_ingest_in_the_agent/?tab=kuberneteshelmvaluesyaml#enabling-otlp-ingestion-on-the-datadog-agent) for other options):

```yaml
otlp:
  receiver:
    protocols:
      grpc:
        enabled: true
```

Then, set this env var, and you're done!

```bash
TRACELOOP_API_ENDPOINT=http://<datadog-agent-hostname>:4317
```

### Honeycomb

Since Honeycomb natively supports OpenTelemetry, you just need to route the traces to Honeycomb's endpoint and set the
API key:

```bash
TRACELOOP_API_ENDPOINT=https://api.honeycomb.io
TRACELOOP_HEADERS="x-honeycomb-team=<YOUR_API_KEY>"
```

### SigNoz

Since SigNoz natively supports OpenTelemetry, you just need to route the traces to SigNoz's endpoint and set the
API key:

```bash
TRACELOOP_API_ENDPOINT=https://ingest.{region}.signoz.cloud
TRACELOOP_HEADERS="signoz-access-token=<SIGNOZ_INGESTION_KEY>"
```

Make sure to set the region to the region you're using (see [table](https://signoz.io/docs/instrumentation/python/#send-traces-to-signoz-cloud)).

## 🪗 What do we instrument?

OpenLLMetry can instrument everything that [OpenTelemetry already instruments](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation) - so things like your DB, API calls, and more. On top of that, we built a set of custom extensions that instrument things like your calls to OpenAI or Anthropic, or your Vector DB like Pinecone, Chroma, or Weaviate.

### LLM Providers

- [x] OpenAI / Azure OpenAI
- [ ] Anthropic
- [ ] Cohere
- [ ] Replicate
- [ ] Vertex AI (GCP)
- [ ] Bedrock (AWS)

### Vector DBs

- [ ] Pinecone
- [ ] Chroma
- [ ] Weaviate

## 🌱 Contributing

Whether it's big or small, we love contributions ❤️ Check out our guide to see how to [get started](https://traceloop.com/docs/contributing/overview).

Not sure where to get started? You can:

- [Book a free pairing session with one of our teammates](mailto:nir@traceloop.com?subject=Pairing%20session&body=I'd%20like%20to%20do%20a%20pairing%20session!)!
- Join our <a href="https://join.slack.com/t/traceloopcommunity/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g">Slack</a>, and ask us any questions there.

## 💚 Community & Support

- [Slack](https://join.slack.com/t/traceloopcommunity/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g) (For live discussion with the community and the Traceloop team)
- [GitHub Discussions](https://github.com/traceloop/openllmetry/discussions) (For help with building and deeper conversations about features)
- [GitHub Issues](https://github.com/traceloop/openllmetry/issues) (For any bugs and errors you encounter using OpenLLMetry)
- [Twitter](https://twitter.com/traceloopdev) (Get news fast)

## 🙏 Special Thanks

To @patrickdebois, who [suggested the great name](https://x.com/patrickdebois/status/1695518950715473991?s=46&t=zn2SOuJcSVq-Pe2Ysevzkg) we're now using for this repo!

```

```
