## v0.13.1 (2024-03-01)

### Fix

- **watsonx**: Watsonx stream generate support (#552)
- **watsonx instrumentation**: Init OTEL_EXPORTER_OTLP_INSECURE before import watsonx models (#549)
- link back to repo in pyproject.toml (#548)

## v0.13.0 (2024-02-28)

### Feat

- basic Support for OpenTelemetry Metrics and Token Usage Metrics in OpenAI V1 (#369)
- **weaviate**: implement weaviate instrumentation (#394)

### Fix

- **watsonx**: exclude http request, adding span for model initialization (#543)

## v0.12.5 (2024-02-27)

### Fix

- **llamaindex**: instrument agents & tools (#533)
- **openai**: Fix `with_raw_response` redirect crashing span (#536)
- **openai**: track client attributes for v1 SDK of OpenAI (#522)
- **sdk**: replaced MySQL instrumentor with SQLAlchemy (#531)

## v0.12.4 (2024-02-26)

### Fix

- **sdk**: fail gracefully if input/output is not json serializable (#525)

## v0.12.3 (2024-02-26)

### Fix

- new PR template (#524)

## v0.12.2 (2024-02-23)

### Fix

- **cohere**: enrich rerank attributes (#476)

## v0.12.1 (2024-02-23)

### Fix

- **llamaindex**: support query pipeline (#475)

## v0.12.0 (2024-02-22)

### Feat

- Qdrant instrumentation (#364)

### Fix

- **langchain**: support LCEL (#473)
- **sdk**: fail gracefully in case input/output serialization failure (#472)

## v0.11.3 (2024-02-19)

### Fix

- **llamaindex**: support both new and legacy llama_index versions (#422)

## v0.11.2 (2024-02-16)

### Fix

- **sdk**: url for getting API key (#424)

## v0.11.1 (2024-02-14)

### Fix

- **openai**: handle async streaming responses for openai v1 client (#421)

## v0.11.0 (2024-02-13)

### Feat

- support both new and legacy llama_index versions (#420)

### Fix

- **sdk**: support input/output of tasks & workflows (#419)

## v0.10.5 (2024-02-13)

### Fix

- **langchain**: backport to 0.0.346 (#418)

## v0.10.4 (2024-02-08)

### Fix

- **openai**: handle OpenAI async completion streaming responses (#409)

## v0.10.3 (2024-01-30)

### Fix

- README

## v0.10.2 (2024-01-25)

### Fix

- re-enabled haystack instrumentation (#77)

## v0.10.1 (2024-01-24)

### Fix

- `resource_attributes` always being None (#359)

## v0.10.0 (2024-01-22)

### Feat

- watsonx support for traceloop (#341)

### Fix

- **sdk**: support arbitrary resources (#338)

## v0.9.4 (2024-01-15)

### Fix

- bug in managed prompts (#337)

## v0.9.3 (2024-01-15)

### Fix

- support langchain v0.1 (#320)

## v0.9.2 (2024-01-12)

### Fix

- otel deps (#336)

## v0.9.1 (2024-01-12)

### Fix

- **openai**: instrument embeddings APIs (#335)

## v0.9.0 (2024-01-11)

### Feat

- google-vertexai-instrumentation (#289)

## v0.8.2 (2024-01-10)

### Fix

- version bump error with replicate (#318)
- version bump error with replicate (#318)

## v0.8.1 (2024-01-10)

### Fix

- replicate release (#316)

## v0.8.0 (2024-01-04)

### Feat

- **semconv**: added top-k (#291)

### Fix

- support anthropic v0.8.1 (#301)
- **ci**: fix replicate release (#285)

## v0.7.0 (2023-12-21)

### Feat

- replicate support (#248)

### Fix

- support pydantic v1 (#282)
- broken tests (#281)

## v0.6.0 (2023-12-16)

### Feat

- **sdk**: user feedback scores (#247)

## v0.5.3 (2023-12-12)

### Fix

- **openai**: async streaming instrumentation (#245)

## v0.5.2 (2023-12-09)

### Fix

- send SDK version on fetch requests (#239)

## v0.5.1 (2023-12-08)

### Fix

- support async workflows in llama-index and openai (#233)

## v0.5.0 (2023-12-07)

### Feat

- **sdk**: support vision api for prompt management (#234)

## v0.4.2 (2023-12-01)

### Fix

- **openai**: langchain streaming bug (#225)

## v0.4.1 (2023-11-30)

### Fix

- **traceloop-sdk**: support explicit prompt versioning in prompt management (#221)

## v0.4.0 (2023-11-29)

### Feat

- bedrock support (#218)

### Fix

- lint issues

## v0.3.6 (2023-11-27)

### Fix

- **openai**: attributes for functions in request (#211)

## v0.3.5 (2023-11-23)

### Fix

- **llama-index**: support ollama completion (#212)

## v0.3.4 (2023-11-22)

### Fix

- **sdk**: flag for dashboard auto-creation (#210)

## v0.3.3 (2023-11-22)

### Fix

- new logo

## v0.3.2 (2023-11-16)

### Fix

- python 3.8 compatibility (#198)
- **cohere**: cohere chat token usage (#196)

## v0.3.1 (2023-11-14)

### Fix

- disable telemetry in tests (#171)

## v0.3.0 (2023-11-10)

### Feat

- sdk telemetry data (#168)

### Fix

- make auto-create path persisted (#170)

## v0.2.1 (2023-11-09)

### Fix

- **openai**: yield chunks for streaming (#166)

## v0.2.0 (2023-11-08)

### Feat

- llamaindex auto instrumentation (#157)

## v0.1.12 (2023-11-07)

### Fix

- **openai**: new OpenAI API v1 (#154)

## v0.1.11 (2023-11-06)

### Fix

- **sdk**: max_tokens are now optional from the backend (#153)

## v0.1.10 (2023-11-03)

### Fix

- errors on logging openai streaming completion calls (#144)

## v0.1.9 (2023-11-03)

### Fix

- **langchain**: improved support for agents and tools with Langchain (#143)
- support streaming API for OpenAI (#142)

## v0.1.8 (2023-11-02)

### Fix

- **prompt-registry**: remove redundant variables print

## v0.1.7 (2023-11-01)

### Fix

- **tracing**: add missing prompt manager template variables to span attributes (#140)

## v0.1.6 (2023-11-01)

### Fix

- **sdk**: allow overriding processor & propagator (#139)
- proper propagation of api key to fetcher (#138)

## v0.1.5 (2023-10-31)

### Fix

- **ci-cd**: release workflow fetches the outdated commit on release package jobs

## v0.1.4 (2023-10-31)

### Fix

- disable syncing when no API key is defined (#135)
- **ci-cd**: finalize release flow (#133)

## v0.1.3 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

## v0.1.2 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

## v0.1.1 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

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
