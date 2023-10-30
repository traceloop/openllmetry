## v0.1.0 (2023-10-30)

### Feat

- **ci-cd**: add release workflow (#132)

### Fix

- release workflow credentials

## v0.0.70 (2023-10-29)

### Feat

- disable content tracing for privacy reasons (#118)
- add prompt version hash (#119)
- propagate prompt management attributes to llm spans (#109)
- support association IDs as objects (#111)
- hugging-face transformers pipeline instrumentation (#104)
- add chromadb instrumentation + fix langchain instrumentation (#103)
- export to Grafana tempo (#95)
- langchain instrumentation (#88)
- **cohere**: support for chat and rerank (#84)
- cohere instrumentation (#82)
- Anthropic instrumentation (#71)
- basic prompt management  (#69)
- Pinecone Instrumentation (#3)
- basic testing framework (#70)
- haystack instrumentations (#55)
- auto-create link to traceloop dashboard
- setting headers for exporting traces
- sdk code + openai instrumentation (#4)

### Fix

- **sdk**: disable sync when using external exporter
- disable content tracing when not overridden (#121)
- **langchain**: add retrieval_qa workflow span (#112)
- **traceloop-sdk**: logging of service name in traces (#99)
- do not trigger dashboard auto-creation if exporter is set (#96)
- **docs**: clarification on getting API key
- **chore**: spaces and nits on README
- **docs**: bad link for python SDK
- **docs**: updated TRACELOOP_BASE_URL (#81)
- add openai function call data to telemetry (#80)
- **sdk**: disabled prompt registry by default (#78)
- support pinecone non-grpc (#76)
- support python 3.12
- **docs**: upgrades; docs about prompt mgmt (#74)
- **traceloop-sdk**: missing lockfile (#72)
- **traceloop-sdk**: flushing in notebooks (#66)
- py security issue
- **docs**: update exporting.mdx to include nr instrumentation (#12)
- **sdk**: async decorators not awaited
- **sdk**: missing dependency
- warn if Traceloop wasn't initialized properly (#11)
- match new dashboard API
- **traceloop-sdk**: duplicate spans reporting (#10)
- moved api key to /tmp
- /v1/traces is always appended to endpoint
- parse headers correctly
- **traceloop-sdk**:  replace context variables with otel context + refactor (#8)
- traceloop sdk initialization and initial versions release for instrumentations (#7)
- wrong imports and missing code components (#6)
- gitignore
- README
