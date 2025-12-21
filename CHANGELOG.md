## v0.50.1 (2025-12-16)

### Fix

- **sample-app**: lint fix (#3522)

## v0.50.0 (2025-12-15)

### Feat

- **guardrail**: Add guardrail decorator (#3521)

## v0.49.8 (2025-12-11)

### Fix

- **openai**: add support for realtime api (websockets) (#3511)
- **ollama**: support Older Version Ollama (#3501)

## v0.49.7 (2025-12-08)

### Fix

- **exp**: Add a real agent example (#3507)
- **evals**: Add agent evaluators to made by traceloop (#3505)
- **exp**: Add made by traceloop evaluators (#3503)
- **traceloop-sdk**: Fixes gRPC exporter initialisation with insecure OTLP (#3481)

## v0.49.6 (2025-12-01)

### Fix

- **agno**: add streaming support for Agent.run() and Agent.arun() (#3483)

## v0.49.5 (2025-11-27)

### Fix

- **openai**: responses instrumentation broken traces for async streaming (#3475)
- **mcp**: remove faulty logic of trying to deduce HTTP errors (#3477)

## v0.49.4 (2025-11-27)

### Fix

- **exp**: Add run in github experiment  (#3459)

## v0.49.3 (2025-11-26)

### Fix

- **openai**: recognize NOT_GIVEN and Omit (#3473)
- **dataset**: add support for file cells in datasets with upload and external URL linking capabilities (#3462)
- **openai**: report request attributes in responses API instrumentation (#3471)
- **sdk**: crewai tracing provider conflict (#3470)
- **sdk**: watsonx warning on initialization (#3469)
- **traceloop-sdk**: add type-checking support with mypy (#3463)

## v0.49.2 (2025-11-25)

### Fix

- **sdk**: remove posthog (#3466)

## v0.49.1 (2025-11-24)

### Fix

- **langchain**: allow configuration of metadata key prefix (#3367)
- **openai**: record service_tier attribute (#3458)

## v0.49.0 (2025-11-23)

### Feat

- **agno**: add instrumentation for agno framework (#3452)

## v0.48.2 (2025-11-23)

### Fix

- add structured outputs schema logging for Anthropic and Gemini (#3454)
- **openai**: use SpanAttributes instead of GenAIAttributes for cache token attributes (#3442)
- migrate from events api to log records for otel 1.37.0+ compatibility (#3453)

## v0.48.1 (2025-11-17)

### Fix

- **openai**: safe handle None tools value in responses api (#3447)
- **mcp**: move exporter dependency to dev and test environment (#3445)

## v0.48.0 (2025-11-11)

### Feat

- **instrumentation**: updated GenAI attributes to use OTel's (#3138)

### Fix

- **openai**: add streaming support for responses.create() api (#3437)
- **bedrock**: handle non-text contentBlockDelta events in converse_stream (#3404)
- **openai-agents**: span attribute handling for tool calls and results (#3422)
- **watson**: collect prompt content and set as span attribute (#3417)

## v0.47.5 (2025-10-24)

### Fix

- **google-genai**: make streaming responses work (again) (#3421)
- **langchain**: changed dictionary access from spans[run_id] to spans.get(run_id) (#3403)

## v0.47.4 (2025-10-22)

### Fix

- **fastmcp**: Remote MCP instrumentation (#3419)

## v0.47.3 (2025-09-21)

### Fix

- **openai-agents**: propagate gen_ai.agent.name through an agent flow + set workflow name to fast mcp (#3388)

## v0.47.2 (2025-09-17)

### Fix

- **mcp**: add mcp.server parent span wrapper for FastMCP tool calls (#3382)

## v0.47.1 (2025-09-14)

### Fix

- **mcp**: better instrumentation for FastMCP (#3372)
- **anthropic**: preserve streaming helper methods in instrumentation (#3377)
- **cohere**: add v2 api instrumentation (#3378)
- **mistralai**: instrumentation for version 1.9+ compatibility (#3376)
- **sdk**: dual bearer send via httpx (#3373)

## v0.47.0 (2025-09-10)

### Feat

- **writer**: initial implementation (#3209)

### Fix

- **crewai**: Update CrewAI instrumentation name (#3363)
- **sample-app**: Update google genai package (#3358)
- **traceloop-sdk**: include telemetry SDK attributes in tracing (#3359)
- **sdk**: get default span processor don't work without a base URL (#3360)

## v0.46.2 (2025-08-29)

### Fix

- **vertexai**: add missing role attributes when handling images (#3347)
- **sdk**: manual logging example + fix span ended error (#3352)
- **sdk**: support disabling all instrumentations (#3353)
- **openai-agents**: support json inputs (#3354)
- **openai**: reasoning jsons weren't stored
- **crewai**: fix unpack error when metrics are disabled (#3345)
- **milvus**: Set default values when metrics are disabled (#3344)

## v0.46.1 (2025-08-24)

### Fix

- **google-generativeai,vertexai**: image support for Gemini models (#3340)

## v0.46.0 (2025-08-24)

### Feat

- **openai**: add reasoning attributes (#3336)
- **semantic-conventions-ai**: Add reasoning attributes (#3330)
- **experiment**: Add run experiment capabilities (#3331)

### Fix

- **traceloop-sdk**: bump logging instrumentation to support newer otel versions (#3339)
- **traceloop-sdk**: add @staticmethod decorator to set_association_properties (#3341)
- **google-genai**: update logic for deciding whether to use awrap or wrap in the Google Generative AI Instrumentation (#3329)
- **ollama**: missing response model attr in operation duration metric (#3328)
- **bedrock**: add guardrail on span attributes (#3326)

## v0.45.6 (2025-08-18)

### Fix

- **anthropic**: fix with_raw_response wrapper consistency and re-enable beta API instrumentation (#3297)
- **langchain**: include content attribute when assistant messages have tool calls (#3287)
- **google-genai**: migrate Google Generative AI instrumentation to googleapis/python-genai (#3282)

## v0.45.5 (2025-08-15)

### Fix

- **openai-agents**: switch to hook-based instrumentation (#3283)

## v0.45.4 (2025-08-14)

### Fix

- relax opentelemetry-semantic-conventions-ai deps (#3259)

## v0.45.3 (2025-08-14)

### Fix

- **anthropic**: temp disable beta apis instrumentation (#3258)

## v0.45.2 (2025-08-14)

### Fix

- **langchain**: langgraph application crash due to context detach (#3256)

## v0.45.1 (2025-08-13)

### Fix

- **langchain**: context detach exception (#3255)
- **mcp**: MCP Instrumentation: streamablehttp_client Parameter Corruption (#3199)

## v0.45.0 (2025-08-12)

### Feat

- **datasets**: add dataset and datasets functionality (#3247)

### Fix

- **anthropic**: support with_raw_response wrapper for span generation (#3250)
- **langchain**: fix nesting of langgraph spans (#3206)
- **langchain**: Add "dont_throw" to "on_llm_end" and remove blank file (#3232)

## v0.44.3 (2025-08-12)

### Fix

- **sdk**: avoid initializing metrics exporter on custom tracing config (#3249)
- **openai**: propagate span IDs properly to events (#3243)

## v0.44.2 (2025-08-11)

### Fix

- **openai**: dynamically import types for 1.99 (#3244)
- **langchain**: Added new method for fetching model name from association metadata (#3237)

## v0.44.1 (2025-08-04)

### Fix

- **mcp**: do not override meta pydantic types (#3179)

## v0.44.0 (2025-08-03)

### Feat

- **sdk**: support multiple span processors (#3207)
- **semantic-conentions-ai**: add LLMVendor enum to semantic conventions (#3170)

### Fix

- **langchain**: spans dictionary memory leak (#3216)
- **openai-agents**: use framework's context to infer trace (#3215)
- **sdk**: respect truncation otel environment variable (#3212)
- **anthropic**: async stream manager (#3220)
- **langchain**: populate metadata as span attributes in batch operations (#3218)
- **anthropic**: various fixes around tools parsing (#3204)
- **qdrant**: fix qdrant-client auto instrumentation condition (#3208)
- **instrumentation**: remove param `enrich_token_usage` and simplify token calculation (#3205)
- **langchain**: ensure llm spans are created for sync cases (#3201)
- **openai**: support for openai non-consumed streams (#3155)

## v0.43.1 (2025-07-23)

### Fix

- **langchain**: added vendors to llm calls (#3165)

## v0.43.0 (2025-07-22)

### Feat

- **prompts**: add tool function support (#3153)

### Fix

- **llamaindex**: structured llm model and temperature parsing (#3159)
- **langchain**: report token usage histogram (#3059)
- **openai**: prioritize api-provided token over tiktoken calculation (#3142)
- **milvus**: Add metrics support (#3013)

## v0.42.0 (2025-07-17)

### Feat

- **llamaindex**: support llamaparse instrumentation (#3103)
- **milvus**: add semantic convention for Milvus DB metrics (#3015)

### Fix

- **openai-agents**: fix broken traces with agents handoff on run_stream (#3143)
- **traceloop-sdk**: redefine histogram bucket boundaries (#3129)

## v0.41.0 (2025-07-13)

### Feat

- **openai-agents**: initial instrumentation; collect OpenAI agent traces and metrics (#2966)
- **google-generativeai**: implement emitting events in addition to current behavior (#2887)
- **vertexai**: implement emitting events in addition to current behavior (#2942)
- **langchain**: implement emitting events in addition to current behavior (#2889)
- **anthropic**: implement emitting events in addition to current behavior (#2884)
- **bedrock**: implement emitting events in addition to current behavior (#2885)
- **llamaindex**: implement emitting events in addition to current behavior (#2941)
- **watsonx**: implement emitting events in addition to current behavior (#2896)
- **cohere**: implement emitting events in addition to current behavior (#2886)
- **groq**: implement emitting events in addition to current behavior (#2888)
- **sagemaker**: implement emitting events in addition to current behavior  (#2894)
- **together**: implement emitting events in addition to current behavior  (#2895)
- **replicate**: implement emitting events in addition to current behavior (#2893)
- **ollama**: implement emitting events in addition to current behavior (#2891)
- **mistralai**: implement emitting events in addition to current behavior  (#2890)
- vendor matching (#3062)
- **semconv**: add an attribute for output schema (#3064)
- **transformers**: implement the support to emitting events in addition to current behavior (#2940)
- **alephalpha**: implement emitting events in addition to current behavior (#2880)
- **mcp**: Add support for mcp streamable http transport type (#3049)
- **milvus**: Add error.type attribute from OpenTelemetry Semantic Conventions  (#3009)
- **openai**: OpenAI responses minimal instrumentation (#3052)
- **ollama**: add meter STTG to ollama instrumentation (#3053)
- **openai**: implement emitting events in addition to current behavior (#2892)

### Fix

- align semconv deps (#3106)
- **sagemaker**: add should_send_prompts checks (#3072)
- **watsonx**: add should_send_prompts check to model response (#3071)
- **groq**: add should_send_prompts checks (#3074)
- **ollama**: add should_send_prompts check (#3073)
- **transformers**: add should_send_prompts checks (#3070)
- **groq**: wrong system attribute was given (#3069)
- **openai**: record exception as span events as well (#3067)
- **openai**: add request schema attribute (#3065)
- **mcp**: add support for error_type in mcp instrumentation (#3050)
- google gemini insturmentation (#3055)
- **openai**: completions.parse out of beta, azure remove double-slash (#3051)

## v0.40.14 (2025-06-24)

### Fix

- instrumentation dependencies issue for google, ollama and redis (#3044)

## v0.40.13 (2025-06-24)

### Fix

- **sdk**: manual report of usage data (#3045)
- **sagemaker**: Improve _handle_call to safely parse JSON, CSV, and byte inputs (#2963)
- **anthropic**: serialize assistant message pydantic models (#3041)
- **sdk**: Ensure instrumentors don’t report successful init if package isn’t installed (#3043)

## v0.40.12 (2025-06-20)

### Fix

- **langchain**: add tool call ids to tool message in history (#3033)

## v0.40.11 (2025-06-17)

### Fix

- **sdk**: sampling support (#3027)

## v0.40.10 (2025-06-17)

### Fix

- **google-genai**: Add support for generate_content method in google genai models (#3014)

## v0.40.9 (2025-06-10)

### Fix

- **langchain**: Fix missing langchain dependency for LangGraph tracing (#2988)

## v0.40.8 (2025-06-09)

### Fix

- **openai**: dump pydantic input message (#2979)
- **langchain**: trace langchain tool definitions (#2978)

## v0.40.7 (2025-05-20)

### Fix

- **mcp**: Added support for newer version of MCP (#2956)
- **gemini**: proper chat support (#2948)
- **milvus**: Add instrumentation for pymilvus MilvusClient hybrid search operation (#2945)

## v0.40.6 (2025-05-16)

### Fix

- **sdk**: support overriding the span processor on_end hook (#2947)
- **milvus**: Added New Semantic Conventions for pymilvus MilvusClient Hybrid Search (#2944)

## v0.40.5 (2025-05-13)

### Fix

- **langchain**: tools in message history (#2939)
- **sdk**: Place MCP in its lexical order (#2943)

## v0.40.4 (2025-05-10)

### Fix

- **milvus**: Enhanced Milvus VectorDB Instrumentation for Improved search Monitoring (#2815)
- **milvus**: Added New Semantic Conventions for Milvus Search (Request for Version Update 0.4.5 -> 0.4.6) (#2883)
- **MCP**: Added error status to traces in MCP server for tool calls (#2914)
- **ollama**: pre-imported funcs instrumentation failure (#2871)

## v0.40.3 (2025-05-07)

### Fix

- **langchain**: report token counts when trace content is enabled (#2899)
- **mcp+anthropic**: vanilla mcp crashed due to argument manipulation (#2881)

## v0.40.2 (2025-04-30)

### Fix

- **ci**: align mcp instrumentation version (#2876)

## v0.40.1 (2025-04-30)

### Fix

- **ci-cd**: add mcp to commitizen (#2875)

## v0.40.0 (2025-04-30)

### Feat

- **instrumentation**: Adding MCP opentelemetry-instrumentation into traceloop (#2829)

## v0.39.4 (2025-04-28)

### Fix

- **sdk**: improve type safety in decorators (#2867)

## v0.39.3 (2025-04-24)

### Fix

- **langchain**: support cached tokens attributes logging (#2830)

## v0.39.2 (2025-04-18)

### Fix

- **openai**: add cache read tokens from returned usage block (#2820)
- **ollama**: type error in dict combination of ollama instrumentation (#2814)
- **llama-index**: use the correct instrumentation point (#2807)

## v0.39.1 (2025-04-15)

### Fix

- **sdk**: Loosen tenacity dependency constraint to allow versions up to 10.0 (#2816)

## v0.39.0 (2025-03-25)

### Feat

- **instrumentation**: add metric for Bedrock prompt caching (#2788)
- **bedrock**: add support for ARN and cross region endpoint (#2785)
- **instrumentation**: Support Converse APIs and guardrail metrics (#2725)

### Fix

- **bedrock**: add span attr for Bedrock prompt caching (#2789)
- **anthropic**: add thinking as a separate completion message (#2780)
- **langchain**: support for date/time in langchain serializations (#2792)
- **openai**: set user messages as prompts, not completions (#2781)
- **groq**: exception when metrics are turned off (#2778)
- **ollama**: Implemented meter in the instrumentation (#2741)

## v0.38.12 (2025-03-07)

### Fix

- **sdk**: client shouldn't be initialized if destination is not traceloop (#2754)

## v0.38.11 (2025-03-06)

### Fix

- **sdk**: When tracing task with no `name` provided , use qualified name instaed of name (#2743)

## v0.38.10 (2025-03-05)

### Fix

- **sdk**: record exceptions (#2733)

## v0.38.9 (2025-03-04)

### Fix

- **milvus**: updated the instrumentation to collect get() and create_collection() span attributes  (#2687)
- **semconv**: added new semantic conventions for milvus db (#2727)

## v0.38.8 (2025-02-27)

### Fix

- **vertexai**: support generative model chat session tracing (#2689)
- **groq**: Updated the instrumentation to collect the token histogram (#2685)

## v0.38.7 (2025-02-19)

### Fix

- **langchain**: warning with mixed metadata value types (#2665)
- **langchain**: handle errors (#2664)
- **groq**: streaming support (#2663)

## v0.38.6 (2025-02-17)

### Fix

- **sdk**: async generator wrapping (#2635)
- **instrumentation**: watsonx initialize parameters (#2633)

## v0.38.5 (2025-02-10)

### Fix

- **sdk**: Fix async decorator input & output json encoder (#2629)

## v0.38.4 (2025-02-06)

### Fix

- **sdk**: improve package name detection with type hints and null safety (#2618)

## v0.38.3 (2025-02-05)

### Fix

- **langchain**: sanitize metadata (#2608)
- **sdk**: improve package name detection for metadata (#2607)
- **openai**: support context propagation for OpenAI v0 and v1 (#2606)
- **openai**: report exceptions on spans (#2604)

## v0.38.2 (2025-02-04)

### Fix

- dependency issue for python 3.13 (#2603)
- **openai**: compatibility issue with openai attribute not available (#2602)

## v0.38.1 (2025-02-04)

### Fix

- **openai**: don't eagerly import types from openai as they may not be available (#2601)
- **sdk**: watsonx package name was wrong preventing instrumentation (#2600)

## v0.38.0 (2025-02-04)

### Feat

- **crewai**: Implemented histogram for crewai instrumentation (#2576)

### Fix

- **mistral**: add response id attribute (#2549)
- **openai**: add response id attribute (#2550)
- **together**: add response id attribute (#2551)
- **langchain**: add response id (anthropic only) (#2548)

## v0.37.1 (2025-01-25)

### Fix

- **crewai**: instrumentation package version (#2570)

## v0.37.0 (2025-01-25)

### Feat

- **crewai**: initial instrumentation; collect agent and task traces (#2489)

### Fix

- add crewai to supported frameworks in readme (#2559)
- **cohere**: add response id attribute (#2545)
- **bedrock**: add response id attribute (#2544)
- **anthropic**: add response id attribute (#2543)
- **groq**: add response id attribute (#2546)
- **langchain**: address warning for NoneType attibute; update some test deps (#2539)

## v0.36.1 (2025-01-20)

### Fix

- **sdk**: remove use of `__all__` (#2536)
- **gemini**: set prompt attributes for role and content (#2511)

## v0.36.0 (2025-01-13)

### Feat

- **sdk**: client, annotations (#2452)

## v0.35.0 (2025-01-03)

### Feat

- **sdk**: combine async/sync decorators & annotations (#2442)

### Fix

- **sdk**: option to disable SDK (#2454)

## v0.34.1 (2024-12-22)

### Fix

- **llamaindex**: workflow context detach exception and span names (#2421)
- **langchain**: azure openai missing request model  (#2416)

## v0.34.0 (2024-12-12)

### Feat

- **docs**: Update README.md to add link to recently-added GCP integration document. (#2384)

### Fix

- bump otel >0.50b0 (#2386)
- update google generative AI import (#2382)
- **sdk**: Update JSONEncoder to allow class instance methods to be serializable (#2383)
- **openai**: Add token count and system to assistants (#2323)
- **openai**: Add trace context to client requests (#2321)

## v0.33.12 (2024-11-13)

### Fix

- **sdk**: aworkflow decorator for async generators (#2292)
- **cohere**: rerank exception on saving response when return_documents=True (#2289)
- **sdk**: gemini instrumentation was never installed due to package name error (#2288)

## v0.33.11 (2024-11-09)

### Fix

- **sdk**: remove print (#2285)

## v0.33.10 (2024-11-08)

### Fix

- general bump of otel dependencies (#2274)

## v0.33.9 (2024-11-05)

### Fix

- **openai**: exception throw with pydantic v1 (#2262)

## v0.33.8 (2024-11-05)

### Fix

- **vertex**: async / streaming was missing output fields (#2253)

## v0.33.7 (2024-11-04)

### Fix

- **sdk**: don't serialize large jsons as inputs/outputs (#2252)

## v0.33.6 (2024-11-04)

### Fix

- **anthropic**: instrument anthropics system message as gen_ai.prompt.0 (#2238)
- **llamaindex**: streaming LLMs caused detached spans (#2237)
- **sdk**: missing sagemaker initialization (#2235)
- **sdk**: support a "block-list" of things to not instrument (#1958)

## v0.33.5 (2024-10-29)

### Fix

- **openai+anthropic**: async call crashing the app when already in a running asyncio loop (#2226)

## v0.33.4 (2024-10-28)

### Fix

- **langchain**: structured output response parsing (#2214)
- **anthropic**: add instrumentation for Anthropic prompt caching (#2175)
- **bedrock**: cohere models failed to report prompts (#2204)

## v0.33.3 (2024-10-22)

### Fix

- **sdk**: capture posthog events as anonymous; roll ingestion key (#2194)

## v0.33.2 (2024-10-17)

### Fix

- **langchain**: various bugs and edge cases in metric exporting (#2167)
- **sdk**: add header for logging exporter (#2164)

## v0.33.1 (2024-10-16)

### Fix

- **langchain**: metrics support (#2154)
- **langchain**: Add trace context to client requests (#2152)
- **anthropic**: add instrumentation for Anthropic tool calling (alternative to #1372) (#2150)
- **openai**: add structured output instrumentation (#2111)

## v0.33.0 (2024-10-15)

### Feat

- **sdk**: add OpenTelemetry logging support (#2112)
- **ollama**: tool calling (#2059)

### Fix

- **llama-index**: add attribute to span for llm request type in dispatcher wrapper (#2141)

## v0.32.2 (2024-10-04)

### Fix

- **traceloop-sdk**: add aiohttp as dependency (#2094)

## v0.32.1 (2024-10-03)

### Fix

- **anthropic**: Replace count_tokens with usage for newer models (#2086)

## v0.32.0 (2024-10-03)

### Feat

- **bedrock**: support metrics for bedrock (#1957)
- **SageMaker**: Add SageMaker instrumentation (#2028)

### Fix

- **langchain**: token usage reporting (#2074)

## v0.31.4 (2024-10-01)

### Fix

- **langchain**: serialize inputs and outputs with pydantic (#2065)
- **sdk**: custom image uploader (#2064)

## v0.31.3 (2024-09-28)

### Fix

- support async image upload flows (#2051)

## v0.31.2 (2024-09-27)

### Fix

- **anthropic**: add support for base64 images upload for anthropic (#2029)

## v0.31.1 (2024-09-26)

### Fix

- **anthropic**: token counting exception when prompt contains images (#2030)

## v0.31.0 (2024-09-25)

### Feat

- **sdk+openai**: support base64 images upload (#2000)

### Fix

- **sdk**: wrong package check for Vertex AI (#2015)

## v0.30.1 (2024-09-16)

### Fix

- **langchain**: support v0.3.0 (#1985)

## v0.30.0 (2024-09-04)

### Feat

- add groq instrumentation (#1928)

## v0.29.2 (2024-08-31)

### Fix

- **langchain**: allow external and langchain metadata (#1922)

## v0.29.1 (2024-08-29)

### Fix

- **bedrock**: llama3 completion wasnt logged (#1914)

## v0.29.0 (2024-08-29)

### Feat

- **instrumentation**: Import redis from OpenTelemetry, add redis sample rag application (#1837)

### Fix

- **langchain**: add missing kind property (#1901)

## v0.28.2 (2024-08-26)

### Fix

- **openai**: calculating streaming usage didnt work on azure models

## v0.28.1 (2024-08-24)

### Fix

- **langchain**: langgraph traces were broken (#1895)

## v0.28.0 (2024-08-24)

### Feat

- **llama-index**: callback improvements (#1859)

### Fix

- **openai**: re-enabled token count for azure instances (#1877)
- **openai**: not given values thrown errors (#1876)
- **sdk**: `aentity_class` was missing a positional argument (#1816)
- **sdk**: instrument threading for propagating otel context (#1868)
- **openai**: TypeError: '<' not supported between instances of 'NoneType' and 'int' in embeddings_wrappers.py (#1836)

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
