## v0.27.0 (2024-08-15)

### Feat

- **llama-index**: Use callbacks (#1546)
- LanceDB Integration (#1749)
- **sdk**: chained entity path on nested tasks (#1782)

### Fix

- workflow_name and entity_path support for langchain + fix entity_name (#1844)
- **sdk**: disable traceloop sync by default (#1835)

## v0.26.5 (2024-08-06)

### Fix

- **langchain**: export metadata as association properties (#1805)
- **bedrock**: add model name for amazon bedrock response (#1757)

## v0.26.4 (2024-08-03)

### Fix

- **bedrock**: token count for titan (#1748)

## v0.26.3 (2024-08-02)

### Fix

- **langchain**: various cases where not all parameters were logged properly (#1725)

## v0.26.2 (2024-07-31)

### Fix

- separate semconv-ai module to avoid conflicts (#1716)

## v0.26.1 (2024-07-30)

### Fix

- bump to otel 0.47b0 (#1695)
- **openai**: log content filter results in proper attributes (#1539)

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
- **openai**: Handle tool_calls assistant messages (#1429)
- **sdk**: speedup SDK initialization (#1374)
- **gemini**: relax version requirements (#1367)

### Refactor

- **openai**: rename function_call to tool_calls (#1431)

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

- **vertexai**: vertexai.generative_models / llm_model detection (#1141)

### Fix

- **bedrock**: support simple string in prompts (#1167)
- **langchain**: stringification fails for lists of LangChain Documents (#1140)

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

- **sdk**: loosen SDK requirements for Sentry (#1034)

## v0.17.3 (2024-05-10)

### Fix

- **openai**: add default model support for token usage (#1033)

## v0.17.2 (2024-05-10)

### Fix

- **sdk**: pin sentry version to <2.0 (#1032)

## v0.17.1 (2024-05-10)

### Fix

- **sdk**: allow user to opt out of child spans (#1031)

## v0.17.0 (2024-05-09)

### Feat

- **openai**: support Azure OpenAI (#1029)
- **cohere**: support cohere models (#1028)

### Fix

- **openai**: fix internal bug on completion token counting (#1026)

## v0.16.0 (2024-05-01)

### Feat

- **openai**: add `max_tokens` and `stop` support (#1020)

### Fix

- **sdk**: add default tracing for `Llama` and `GPT` models (#1019)

## v0.15.0 (2024-04-20)

### Feat

- **openai**: use `content_filter` for additional metrics (#1006)

### Fix

- **openai**: fix missing spans (#1005)

## v0.14.0 (2024-04-15)

### Feat

- **togetherai**: initial togetherai metrics (#994)

### Fix

- **openai**: additional serialization of token usages (#993)

## v0.13.0 (2024-04-12)

### Feat

- **sklearn**: added sklearn package (#989)

### Fix

- **sdk**: ensure metric values are persisted (#988)

## v0.12.0 (2024-04-11)

### Feat

- **langchain**: initial langchain support (#984)

### Fix

- **sdk**: add child tracing for SDK (#983)

## v0.11.0 (2024-04-05)

### Feat

- **sdk**: initial SDK instrumentations (#981)

### Fix

- **openai**: relax versions of token counts (#980)

## v0.10.0 (2024-04-01)

### Feat

- **openai**: token count instrumentations (#978)

### Fix

- **openai**: relaxed model versions (#979)

## v0.9.0 (2024-03-31)

### Feat

- **openai**: instrumenting models for request/response (#976)

### Fix

- **openai**: fallback for token counts (#975)

## v0.8.0 (2024-03-30)

### Feat

- **sdk**: baseline SDK tracing (#973)

### Fix

- **sdk**: added generic span creation (#972)
- **openai**: updated instrumentations (#971)

## v0.7.0 (2024-03-28)

### Feat

- **langchain**: langchain integration (#970)

### Fix

- **openai**: fixed missing metrics (#969)

## v0.6.0 (2024-03-27)

### Feat

- **openai**: added token count metrics (#968)

### Fix

- **openai**: added error handling (#967)

## v0.5.0 (2024-03-25)

### Feat

- **openai**: token usage improvements (#966)

### Fix

- **openai**: fixed bugs with inputs (#965)

## v0.4.0 (2024-03-22)

### Feat

- **openai**: added support for model instrumentation (#964)

### Fix

- **openai**: improved metrics handling (#963)

## v0.3.0 (2024-03-20)

### Feat

- **openai**: initial metrics implementation (#962)

### Fix

- **openai**: updated documentation (#961)

## v0.2.0 (2024-03-19)

### Feat

- **openai**: improved token usage tracking (#960)

### Fix

- **openai**: fix bugs in tracking (#959)

## v0.1.0 (2024-03-15)

### Feat

- Initial release of the package.
