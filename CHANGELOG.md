## v0.26.0 (2024-07-26)

### Feat

- **openai**: add tool call id (#1664)

### Fix

- **pinecone**: support v5 (#1665)

## v0.25.6 (2024-07-23)

### Fix

- **sdk**: aworkflow wasn't propagating workflow_name attribute (#1648)
- **langchain**: agent executor weren't producing traces (#1616)

## v0.25.5 (2024-07-17)

### Fix

- **openai**: pydantic tool calls in prompt weren't serialized correctly (#1572)

## v0.25.4 (2024-07-15)

### Fix

- **sdk**: manual reporting of llm spans (#1555)

## v0.25.3 (2024-07-11)

### Fix

- **langchain**: input/output values weren't respecting user config (#1540)

## v0.25.2 (2024-07-11)

### Fix

- **llamaindex**: report entity name (#1525)
- **langchain**: remove leftover print
- **langchain**: cleanups, and fix streaming issue (#1522)
- **langchain**: report llm spans (instead of normal instrumentations) (#1452)

## v0.25.1 (2024-07-09)

### Fix

- association properties and workflow / task on metrics (#1494)
- **llamaindex**: report inputs+outputs on entities (#1495)

## v0.25.0 (2024-07-08)

### Feat

- suppress LLM instrumentations through context (#1453)
- **langchain**: improve callbacks (#1426)

### Fix

- **sdk**: llamaindex instrumentation was never initialized (#1490)

## v0.24.0 (2024-07-03)

### Feat

- **sdk**: prompt versions and workflow versions (#1425)
- **openai**: add support for parallel function calls (#1424)
- **marqo**: Add marqo instrumentation (#1373)

### Fix

- **sdk**: context detach issues on fastapi (#1432)
- **openai**: Handle `tool_calls` assistant messages (#1429)
- **sdk**: speedup SDK initialization (#1374)
- **gemini**: relax version requirements (#1367)

### Refactor

- **openai**: rename `function_call` to `tool_calls` (#1431)

## v0.23.0 (2024-06-17)

### Feat

- **langchain**: use callbacks (#1170)

### Fix

- input/output serialization issue for langchain (#1341)
- **sdk**: remove auto-create dashboard option (#1315)

## v0.22.1 (2024-06-13)

### Fix

- **sdk**: backpropagate association property to nearest workflow/task (#1300)
- **sdk**: clear context when @workflow or @task is ending (#1301)
- **bedrock**: utilize invocation metrics from response body for AI21, Anthropic, Meta models when available to record usage on spans (#1286)

## v0.22.0 (2024-06-10)

### Feat

- **gemini**: basic support in generate_content API (#1293)
- **alephalpha**: Add AlephAlpha instrumentation (#1285)
- **instrumentation**: add streamed OpenAI function tracing (#1284)
- **togetherai**: Add together ai instrumentation (#1264)

### Fix

- **anthropic**: duplicate creation of metrics (#1294)
- **haystack**: add input and output (#1202)
- **openai**: calculate token usage for azure (#1274)
- use constants (#1131)
- **instrumentation**: Handle OpenAI run polling (#1256)

## v0.21.5 (2024-06-05)

### Fix

- **openai**: handle empty finish_reason (#1236)
- removed debug prints from instrumentations
- **vertexai**: change the span names to match method calls (#1234)

## v0.21.4 (2024-06-03)

### Fix

- **openai+anthropic+watsonx**: align duration and token.usage metrics attributes with conventions (#1182)

## v0.21.3 (2024-06-03)

### Fix

- **openai**: async streaming responses (#1229)
- **sdk**: temporarily (?) remove sentry (#1228)

## v0.21.2 (2024-05-31)

### Fix

- **all packages**: Bump opentelemetry-api to 1.25.0 and opentelemetry-instrumentation to 0.46b0 (#1189)

## v0.21.1 (2024-05-30)

### Fix

- log tracing errors on debug level (#1180)
- **bedrock**: support streaming API (#1179)
- **weaviate**: support v4.6.3 (#1134)
- **sdk**: wrong package check for mistral instrumentations (#1168)

## v0.21.0 (2024-05-27)

### Feat

- **vertexai**: `vertexai.generative_models` / `llm_model` detection (#1141)

### Fix

- **bedrock**: support simple string in prompts (#1167)
- **langchain**: stringification fails for lists of LangChain `Documents` (#1140)

## v0.20.0 (2024-05-26)

### Feat

- **mistral**: implement instrumentation (#1139)
- **ollama**: implement instrumentation (#1138)

### Fix

- **anthropic**: don't fail if can't count anthropic tokens (#1142)
- **ollama**: proper unwrapping; limit instrumentations to versions <1
- **bedrock**: instrument bedrock calls for Langchain (with session) (#1135)

## v0.19.0 (2024-05-22)

### Feat

- **milvus**: add Milvus instrumentation (#1068)

### Fix

- add explicit buckets to pinecone histograms (#1129)
- **pinecone**: backport to v2.2.2 (#1122)
- llm metrics naming + views (#1121)
- **langchain**: better serialization of inputs and outputs (#1120)
- **sdk**: failsafe against instrumentation initialization errors (#1117)
- **sdk**: instrument milvus (#1116)

## v0.18.2 (2024-05-17)

### Fix

- **openai**: old streaming handling for backward compatibility with OpenAI v0 (#1064)
- **openai**: report fingerprint from response (#1066)
- **sdk**: special handling for metrics with custom traces exporter (#1065)

## v0.18.1 (2024-05-17)

### Fix

- **openai**: fallback to response model if request model is not set when calculating token usage (#1054)
- **openai**: add default value of stream as false in token usage metric (#1055)

## v0.18.0 (2024-05-14)

### Feat

- **pinecone**: metrics support (#1041)

### Fix

- **sdk**: handle workflow & tasks generators (#1045)
- **cohere**: use billed units for token usage (#1040)

## v0.17.7 (2024-05-13)

### Fix

- remove all un-needed tiktoken deps (#1039)

## v0.17.6 (2024-05-13)

### Fix

- **sdk**: removed unneeded tiktoken dependency (#1038)

## v0.17.5 (2024-05-13)

### Fix

- **openai**: relax tiktoken requirements (#1035)

## v0.17.4 (2024-05-13)

### Fix

- **sdk**: loosen SDK requirements for Sentry + Posthog (#1027)

## v0.17.3 (2024-05-08)

### Fix

- **sdk**: separate sentry SDK (#1004)

## v0.17.2 (2024-05-07)

### Fix

- **langchain**: support model-specific packages (#985)
- **pinecone**: filter argument may be dict (#984)

## v0.17.1 (2024-05-01)

### Fix

- **instrumentation**: correct the module declaration to match package filepath name (#940)

## v0.17.0 (2024-04-29)

### Feat

- **sdk**: otel metrics with traceloop (#883)
- Updated semantic conventions based on otel community (#884)

### Fix

- **sdk**: do not instrument sentry requests (used internally by SDK) (#939)

## v0.16.9 (2024-04-26)

### Fix

- **openai**: missing await for Embedding.acreate (#900)
- **cohere**: support v5 (#899)
- **pinecone**: support v3 (#895)
- **instrumentation**: the build problem for watsonx auto instrumentation (#885)

## v0.16.8 (2024-04-25)

### Fix

- **langchain**: input/output reporting (#894)
- **sdk**: reset the color of messages in the custom metrics exporter (#893)

## v0.16.7 (2024-04-25)

### Fix

- **openai**: azure filtering masked all completions (#886)
- **chromadb**: exception thrown when metadata isn't set (#882)

## v0.16.6 (2024-04-19)

### Fix

- properly handle and report exceptions (#748)
- **langchain**: bug when retrieving messages as kwargs from model invoke (#856)
- **openai**: handle filtered content (#854)
- **bedrock**: loosen version requirement of anthropic (#830)
- **haystack**: V2 Support (#710)

## v0.16.5 (2024-04-17)

### Fix

- **sdk**: warn for reporting score when not using Traceloop (#829)
- **openai**: fix aembeddings init error (#828)
- **openai**: missing aembedding metrics

## v0.16.4 (2024-04-15)

### Fix

- **anthropic**: fix issue with disabled metrics (#820)

## v0.16.3 (2024-04-15)

### Fix

- **openai**: missing metrics for OpenAI v0 instrumentation (#818)

## v0.16.2 (2024-04-14)

### Fix

- **bedrock**: enrich token usage for anthropic calls (#805)
- **langchain**: use chain names if exist (#804)

## v0.16.1 (2024-04-11)

### Fix

- **llamaindex**: proper support for custom LLMs (#776)
- **anthropic**: prompt attribute name (#775)
- **langchain**: BedrockChat model name should be model_id (#763)

## v0.16.0 (2024-04-10)

### Feat

- **instrumentation-anthropic**: Support for OpenTelemetry metrics for Anthropic (#764)

### Fix

- **bedrock**: support anthropic v3 (#770)

## v0.15.13 (2024-04-08)

### Fix

- **sdk**: custom instruments missing parameters (#769)
- **sdk**: import of removed method
- **sdk**: removed deprecated set_context

## v0.15.12 (2024-04-08)

### Fix

- **anthropic**: do not fail for missing methods
- **anthropic**: Async and streaming Anthropic (#750)

## v0.15.11 (2024-04-04)

### Fix

- **openai**: async streaming metrics (#749)

## v0.15.10 (2024-04-04)

### Fix

- **anthropic**: token usage (#747)

## v0.15.9 (2024-04-03)

### Fix

- **openai**: switch to init flag for token usage enrichment (#745)
- **anthropic**: support multi-modal (#746)
- **langchain**: instrument chat models (#741)

## v0.15.8 (2024-04-03)

### Fix

- bump otel -> 0.45.0 (#740)

## v0.15.7 (2024-04-03)

### Fix

- enrich spans with related entity name + support entities nesting (#713)

## v0.15.6 (2024-04-02)

### Fix

- **sdk**: stricter dependencies for instrumentations

## v0.15.5 (2024-04-02)

### Fix

- **openai**: missing metric for v0 instrumentation (#735)

## v0.15.4 (2024-03-31)

### Fix

- **traceloop-sdk**: default value for metrics endpoint (#711)

## v0.15.3 (2024-03-28)

### Fix

- instrumentation deps without the SDK (#707)
- **langchain**: support custom models (#706)

## v0.15.2 (2024-03-27)

### Fix

- **openai**: enrich assistant data if not available (#705)

## v0.15.1 (2024-03-27)

### Fix

- **openai**: support pre-created assistants (#701)

## v0.15.0 (2024-03-26)

### Feat

- **openai**: assistants API (#673)
- **pinecone**: instrument pinecone query embeddings (#368)

### Fix

- **traceloop-sdk**: custom span processor's on_start is honored (#695)
- **openai**: do not import tiktoken if not used
- **sdk**: exclude api.traceloop.com from requests
- **openai**: Support report token usage in stream mode (#661)

## v0.14.5 (2024-03-21)

### Fix

- **anthropic**: support messages API (#671)

## v0.14.4 (2024-03-21)

### Fix

- auto-instrumentation support (#662)
- **sample**: poetry issues; litellm sample
- **sdk**: better logging for otel metrics
- **sdk**: error for manually providing instrumentation list

## v0.14.3 (2024-03-17)

### Fix

- support python 3.12 (#639)
- **traceloop-sdk**: Log error message when providing wrong API key. (#638)

## v0.14.2 (2024-03-15)

### Fix

- **openai**: support tool syntax (#630)

## v0.14.1 (2024-03-12)

### Fix

- **sdk**: protect against unserializable inputs/outputs (#626)

## 0.14.0 (2024-03-12)

### Feat

- **watsonx instrumentation**: Watsonx metric support (#593)

### Fix

- **instrumentations**: add entry points to support auto-instrumentation (#592)

## 0.13.3 (2024-03-07)

### Fix

- **llamaindex**: backport to support v0.9.x (#590)
- **openai**: is_streaming attribute (#589)

## 0.13.2 (2024-03-06)

### Fix

- **openai**: span events on completion chunks in streaming (#586)
- **openai**: streaming metrics (#585)

## 0.13.1 (2024-03-01)

### Fix

- **watsonx**: Watsonx stream generate support (#552)
- **watsonx instrumentation**: Init OTEL_EXPORTER_OTLP_INSECURE before import watsonx models (#549)
- link back to repo in pyproject.toml (#548)

## 0.13.0 (2024-02-28)

### Feat

- basic Support for OpenTelemetry Metrics and Token Usage Metrics in OpenAI V1 (#369)
- **weaviate**: implement weaviate instrumentation (#394)

### Fix

- **watsonx**: exclude http request, adding span for model initialization (#543)

## 0.12.5 (2024-02-27)

### Fix

- **llamaindex**: instrument agents & tools (#533)
- **openai**: Fix `with_raw_response` redirect crashing span (#536)
- **openai**: track client attributes for v1 SDK of OpenAI (#522)
- **sdk**: replaced MySQL instrumentor with SQLAlchemy (#531)

## 0.12.4 (2024-02-26)

### Fix

- **sdk**: fail gracefully if input/output is not json serializable (#525)

## 0.12.3 (2024-02-26)

### Fix

- new PR template (#524)

## 0.12.2 (2024-02-23)

### Fix

- **cohere**: enrich rerank attributes (#476)

## 0.12.1 (2024-02-23)

### Fix

- **llamaindex**: support query pipeline (#475)

## 0.12.0 (2024-02-22)

### Feat

- Qdrant instrumentation (#364)

### Fix

- **langchain**: support LCEL (#473)
- **sdk**: fail gracefully in case input/output serialization failure (#472)

## 0.11.3 (2024-02-19)

### Fix

- **llamaindex**: support both new and legacy llama_index versions (#422)

## 0.11.2 (2024-02-16)

### Fix

- **sdk**: url for getting API key (#424)

## 0.11.1 (2024-02-14)

### Fix

- **openai**: handle async streaming responses for openai v1 client (#421)

## 0.11.0 (2024-02-13)

### Feat

- support both new and legacy llama_index versions (#420)

### Fix

- **sdk**: support input/output of tasks & workflows (#419)

## 0.10.5 (2024-02-13)

### Fix

- **langchain**: backport to 0.0.346 (#418)

## 0.10.4 (2024-02-08)

### Fix

- **openai**: handle OpenAI async completion streaming responses (#409)

## 0.10.3 (2024-01-30)

### Fix

- README

## 0.10.2 (2024-01-25)

### Fix

- re-enabled haystack instrumentation (#77)

## 0.10.1 (2024-01-24)

### Fix

- `resource_attributes` always being None (#359)

## 0.10.0 (2024-01-22)

### Feat

- watsonx support for traceloop (#341)

### Fix

- **sdk**: support arbitrary resources (#338)

## 0.9.4 (2024-01-15)

### Fix

- bug in managed prompts (#337)

## 0.9.3 (2024-01-15)

### Fix

- support langchain v0.1 (#320)

## 0.9.2 (2024-01-12)

### Fix

- otel deps (#336)

## 0.9.1 (2024-01-12)

### Fix

- **openai**: instrument embeddings APIs (#335)

## 0.9.0 (2024-01-11)

### Feat

- google-vertexai-instrumentation (#289)

## 0.8.2 (2024-01-10)

### Fix

- version bump error with replicate (#318)
- version bump error with replicate (#318)

## 0.8.1 (2024-01-10)

### Fix

- replicate release (#316)

## 0.8.0 (2024-01-04)

### Feat

- **semconv**: added top-k (#291)

### Fix

- support anthropic v0.8.1 (#301)
- **ci**: fix replicate release (#285)

## 0.7.0 (2023-12-21)

### Feat

- replicate support (#248)

### Fix

- support pydantic v1 (#282)
- broken tests (#281)

## 0.6.0 (2023-12-16)

### Feat

- **sdk**: user feedback scores (#247)

## 0.5.3 (2023-12-12)

### Fix

- **openai**: async streaming instrumentation (#245)

## 0.5.2 (2023-12-09)

### Fix

- send SDK version on fetch requests (#239)

## 0.5.1 (2023-12-08)

### Fix

- support async workflows in llama-index and openai (#233)

## 0.5.0 (2023-12-07)

### Feat

- **sdk**: support vision api for prompt management (#234)

## 0.4.2 (2023-12-01)

### Fix

- **openai**: langchain streaming bug (#225)

## 0.4.1 (2023-11-30)

### Fix

- **traceloop-sdk**: support explicit prompt versioning in prompt management (#221)

## 0.4.0 (2023-11-29)

### Feat

- bedrock support (#218)

### Fix

- lint issues

## 0.3.6 (2023-11-27)

### Fix

- **openai**: attributes for functions in request (#211)

## 0.3.5 (2023-11-23)

### Fix

- **llama-index**: support ollama completion (#212)

## 0.3.4 (2023-11-22)

### Fix

- **sdk**: flag for dashboard auto-creation (#210)

## 0.3.3 (2023-11-22)

### Fix

- new logo

## 0.3.2 (2023-11-16)

### Fix

- python 3.8 compatibility (#198)
- **cohere**: cohere chat token usage (#196)

## 0.3.1 (2023-11-14)

### Fix

- disable telemetry in tests (#171)

## 0.3.0 (2023-11-10)

### Feat

- sdk telemetry data (#168)

### Fix

- make auto-create path persisted (#170)

## 0.2.1 (2023-11-09)

### Fix

- **openai**: yield chunks for streaming (#166)

## 0.2.0 (2023-11-08)

### Feat

- llamaindex auto instrumentation (#157)

## 0.1.12 (2023-11-07)

### Fix

- **openai**: new OpenAI API v1 (#154)

## 0.1.11 (2023-11-06)

### Fix

- **sdk**: max_tokens are now optional from the backend (#153)

## 0.1.10 (2023-11-03)

### Fix

- errors on logging openai streaming completion calls (#144)

## 0.1.9 (2023-11-03)

### Fix

- **langchain**: improved support for agents and tools with Langchain (#143)
- support streaming API for OpenAI (#142)

## 0.1.8 (2023-11-02)

### Fix

- **prompt-registry**: remove redundant variables print

## 0.1.7 (2023-11-01)

### Fix

- **tracing**: add missing prompt manager template variables to span attributes (#140)

## 0.1.6 (2023-11-01)

### Fix

- **sdk**: allow overriding processor & propagator (#139)
- proper propagation of api key to fetcher (#138)

## 0.1.5 (2023-10-31)

### Fix

- **ci-cd**: release workflow fetches the outdated commit on release package jobs

## 0.1.4 (2023-10-31)

### Fix

- disable syncing when no API key is defined (#135)
- **ci-cd**: finalize release flow (#133)

## 0.1.3 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

## 0.1.2 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

## 0.1.1 (2023-10-30)

### Fix

- **ci-cd**: fix release workflow publish step

## 0.1.0 (2023-10-30)

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
