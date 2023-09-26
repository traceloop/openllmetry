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

Next, you need to decide where to export the traces to.

## ⏫ Supported destinations

- [x] [Traceloop](https://traceloop.com/docs/python-sdk/exporting#traceloop)
- [x] [Datadog](https://traceloop.com/docs/python-sdk/exporting#datadog)
- [x] [New Relic](https://traceloop.com/docs/python-sdk/exporting#new-relic)
- [x] [Honeycomb](https://traceloop.com/docs/python-sdk/exporting#honeycomb)
- [x] [SigNoz](https://traceloop.com/docs/python-sdk/exporting#signoz)

## 🪗 What do we instrument?

OpenLLMetry can instrument everything that [OpenTelemetry already instruments](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation) - so things like your DB, API calls, and more. On top of that, we built a set of custom extensions that instrument things like your calls to OpenAI or Anthropic, or your Vector DB like Pinecone, Chroma, or Weaviate.

### LLM Providers

- [x] OpenAI / Azure OpenAI
- [x] Anthropic
- [ ] Cohere
- [ ] Replicate
- [ ] HuggingFace
- [ ] Vertex AI (GCP)
- [ ] Bedrock (AWS)

### Vector DBs

- [x] Pinecone
- [ ] Chroma
- [ ] Weaviate

### Frameworks

- [x] Haystack
- [ ] LangChain
- [ ] LlamaIndex

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
