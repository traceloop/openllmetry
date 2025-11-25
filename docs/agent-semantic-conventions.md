# OpenTelemetry Semantic Conventions for AI Agents

**Status**: Experimental
**Version**: 0.1.0
**Last Updated**: 2025-01-23

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Span Definitions](#span-definitions)
4. [Attribute Registry](#attribute-registry)
5. [Event Definitions](#event-definitions)
6. [Metrics Definitions](#metrics-definitions)
7. [Span Hierarchies](#span-hierarchies)
8. [Framework Mappings](#framework-mappings)
9. [Use Case Scenarios](#use-case-scenarios)
10. [Integration Guide](#integration-guide)

---

## Introduction

This document defines semantic conventions for observability of AI agent systems built with frameworks like LangGraph, CrewAI, Autogen, Google ADK, LlamaIndex, OpenAI Agents SDK, Agno, MastraAI, Smolagents, Haystack, and AWS Bedrock AgentCore.

These conventions extend the existing [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) by adding specialized concepts for:

- **Agent lifecycle and orchestration** - Creation, execution, handoffs, termination
- **Multi-agent coordination** - Teams, crews, agent-to-agent communication
- **Task management** - Task creation, execution, delegation, hierarchies
- **Memory systems** - Short-term, long-term, vector-based memory operations
- **Tool execution** - Function calling, MCP integrations, error handling
- **Workflow orchestration** - Graph-based execution, state transitions, checkpointing
- **Quality assurance** - Guardrails, runtime evaluations, human-in-the-loop
- **Session management** - Conversations, context, checkpointing

### Design Principles

1. **Framework Agnostic** - Conventions work across all major agent frameworks
2. **Extend, Don't Replace** - Build on existing `gen_ai.*` conventions
3. **Practical Cardinality** - Balance detail with performance
4. **Observable by Default** - Enable monitoring, debugging, and optimization
5. **Privacy Aware** - Guidance on PII and sensitive data handling

### Namespace Convention

All agent-specific attributes use the `gen_ai.*` prefix to align with existing OpenTelemetry GenAI semantic conventions:

- `gen_ai.agent.*` - Agent-specific attributes
- `gen_ai.team.*` - Multi-agent team attributes
- `gen_ai.task.*` - Task-specific attributes
- `gen_ai.tool.*` - Tool execution attributes (extends existing)
- `gen_ai.memory.*` - Memory system attributes
- `gen_ai.workflow.*` - Workflow orchestration attributes
- `gen_ai.session.*` - Session management attributes
- And more...

---

## Overview

### Agent System Architecture

Modern AI agent systems typically consist of:

```
Session (Conversation)
├── Agent(s)
│   ├── Workflow/Graph (Orchestration)
│   │   ├── Task(s) (Work Units)
│   │   │   ├── LLM Calls (Reasoning)
│   │   │   ├── Tool Executions (Actions)
│   │   │   ├── Memory Operations (Context)
│   │   │   └── Guardrails (Quality)
│   │   └── State Transitions
│   ├── Handoffs (Agent-to-Agent)
│   └── Human Reviews (HITL)
└── Artifacts (Outputs)
```

### Span Types Summary

This specification defines **20 primary span types** organized into categories:

- **Lifecycle** (4): session, agent.create, agent.invoke, agent.terminate
- **Orchestration** (6): team.create, team.execute, team.coordinate, workflow.execute, workflow.transition, workflow.branch
- **Task Execution** (4): task.create, task.execute, task.delegate, agent.handoff
- **Memory** (5): memory.store, memory.retrieve, memory.search, memory.update, memory.delete
- **Tools & Integration** (3): tool.execute, mcp.connect, mcp.execute
- **Context & State** (2): context.checkpoint, context.compress
- **Quality & Control** (3): guardrail.check, eval.execute, human.review

Plus existing **gen_ai.client.*** spans for LLM operations.

---

## Span Definitions

### Naming Convention

All span names follow the pattern: `gen_ai.<component>.<operation>`

Examples:
- `gen_ai.agent.invoke`
- `gen_ai.memory.retrieve`
- `gen_ai.workflow.execute`

---

## 1. Lifecycle Spans

### 1.1 `gen_ai.session`

**Description**: Represents a complete agent session, conversation, or autonomous run. This is the top-level span encompassing all agent activities.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.session.id` | string | Unique session identifier (stable across resumptions) | `"sess_abc123"`, `"conv_456def"` |
| `gen_ai.session.start_time` | timestamp | Session start timestamp | `2025-01-23T10:30:00Z` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.session.type` | string | Session category | `"chat"`, `"autonomous_run"`, `"multi_agent_session"`, `"batch"` |
| `gen_ai.session.thread_id` | string | Thread identifier (multi-tenant scenarios) | `"thread_789"` |
| `gen_ai.session.user_id` | string | End-user identifier (hashed/PII-safe) | `"user_hash_xyz"` |
| `gen_ai.session.persistent` | boolean | Whether session state persists | `true`, `false` |
| `gen_ai.session.message_count` | int | Total messages in session | `15` |
| `gen_ai.session.turn_count` | int | Total conversation turns | `7` |
| `gen_ai.session.start_reason` | string | What triggered this session | `"user_message"`, `"scheduled_task"`, `"api_call"`, `"webhook"` |
| `gen_ai.agent.framework` | string | Primary agent framework used | `"langgraph"`, `"crewai"`, `"autogen"`, `"openai-agents"` |
| `gen_ai.agent.framework.version` | string | Framework version | `"0.2.0"`, `"1.5.3"` |
| `gen_ai.environment` | string | Deployment environment | `"dev"`, `"staging"`, `"prod"` |

**Framework Examples**:

- **LangGraph**: Session maps to a thread with persistent checkpoint history
- **CrewAI**: Session encompasses full crew execution
- **OpenAI Agents SDK**: Session corresponds to SQLiteSession or custom session
- **Google ADK**: Session managed by SessionService (in-memory, SQL, or Vertex AI)

---

### 1.2 `gen_ai.agent.create`

**Description**: Agent initialization and configuration. Captures the creation of an agent instance with its configuration.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.id` | string | Unique agent instance identifier | `"agent_123"`, `"researcher_01"` |
| `gen_ai.agent.name` | string | Human-readable agent name | `"TravelAssistant"`, `"CodeReviewer"`, `"ResearchAgent"` |
| `gen_ai.agent.type` | string | Agent implementation type | `"react"`, `"function_calling"`, `"conversational"`, `"task_executor"` |
| `gen_ai.agent.framework` | string | Framework used to build agent | `"langgraph"`, `"crewai"`, `"llamaindex"`, `"smolagents"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.role` | string | Agent's role or persona | `"Senior Python Engineer"`, `"Support Specialist"`, `"Researcher"` |
| `gen_ai.agent.goal` | string | Agent's high-level objective | `"Research AI trends"`, `"Fix bugs in codebase"` |
| `gen_ai.agent.backstory` | string | Agent's personality/context description | `"You're a seasoned researcher..."` |
| `gen_ai.agent.mode` | string | Architectural pattern | `"react"`, `"plan_and_solve"`, `"autonomous"`, `"supervisor"`, `"code_interpreter"` |
| `gen_ai.agent.version` | string | Agent version | `"1.0.0"`, `"v2"` |
| `gen_ai.agent.capabilities` | string[] | List of agent capabilities | `["web_search", "code_execution", "file_access"]` |
| `gen_ai.agent.tools` | string[] | List of tool names available to agent | `["calculator", "search_web", "read_file"]` |
| `gen_ai.agent.memory_enabled` | boolean | Whether agent has memory | `true`, `false` |
| `gen_ai.agent.delegation_enabled` | boolean | Whether agent can delegate to others | `true`, `false` |
| `gen_ai.agent.max_iterations` | int | Maximum execution iterations | `10`, `50` |
| `gen_ai.agent.timeout_ms` | int | Execution timeout in milliseconds | `30000`, `60000` |

**Framework Examples**:

- **CrewAI**: `@agent` decorator with role, goal, backstory, tools
- **Autogen**: `AssistantAgent` with name, system_message, tools, max_tool_iterations
- **OpenAI SDK**: `Agent` with name, instructions, tools, handoffs, output_type
- **Agno**: `Agent(name, model, instructions, tools, add_history_to_context)`

---

### 1.3 `gen_ai.agent.invoke`

**Description**: A single agent invocation/execution. This is the primary span for agent activity.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.id` | string | Agent instance identifier | `"agent_123"` |
| `gen_ai.agent.name` | string | Agent name | `"TravelAssistant"` |
| `gen_ai.operation.name` | string | Operation being performed | `"execute"`, `"run"`, `"process"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.session.id` | string | Associated session ID | `"sess_abc123"` |
| `gen_ai.session.thread_id` | string | Thread identifier | `"thread_789"` |
| `gen_ai.request.model` | string | Primary LLM model used | `"gpt-4"`, `"claude-3-5-sonnet"` |
| `gen_ai.response.model` | string | Actual model that responded | `"gpt-4-0613"` |
| `gen_ai.usage.total_tokens` | int | Total tokens consumed | `1523` |
| `gen_ai.runtime.llm_calls_count` | int | Number of LLM calls made | `3` |
| `gen_ai.runtime.tool_calls_count` | int | Number of tool invocations | `5` |
| `gen_ai.runtime.duration_ms` | int | Total execution duration | `4500` |
| `gen_ai.runtime.iterations` | int | Number of agent loop iterations | `3` |
| `error.type` | string | Error type if failed | `"timeout"`, `"tool_error"`, `"model_error"` |

**Framework Examples**:

- **LlamaIndex**: `agent.run()` or `agent.chat()` invocation
- **LangGraph**: `graph.invoke()` or `graph.stream()` execution
- **OpenAI SDK**: `Runner.run()` or `Runner.run_sync()` call
- **Smolagents**: `agent.run()` with ReAct loop

---

### 1.4 `gen_ai.agent.terminate`

**Description**: Agent cleanup and termination. Captures the end of an agent's lifecycle.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.id` | string | Agent instance identifier | `"agent_123"` |
| `gen_ai.agent.name` | string | Agent name | `"TravelAssistant"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.termination_reason` | string | Why agent terminated | `"completed"`, `"error"`, `"timeout"`, `"user_cancelled"` |
| `gen_ai.runtime.total_invocations` | int | Total times agent was invoked | `15` |
| `gen_ai.runtime.total_duration_ms` | int | Cumulative execution time | `125000` |

---

## 2. Orchestration Spans

### 2.1 `gen_ai.team.create`

**Description**: Multi-agent team or crew initialization.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.team.id` | string | Unique team identifier | `"team_research"`, `"crew_123"` |
| `gen_ai.team.name` | string | Team name | `"Research Team"`, `"Support Crew"` |
| `gen_ai.team.size` | int | Number of agents in team | `3`, `5` |
| `gen_ai.team.orchestration_pattern` | string | How team coordinates | `"sequential"`, `"hierarchical"`, `"round_robin"`, `"selector"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.team.manager_agent_id` | string | Manager/coordinator agent ID | `"agent_manager_1"` |
| `gen_ai.agent.framework` | string | Framework used | `"crewai"`, `"autogen"` |
| `gen_ai.team.agents` | string[] | List of agent IDs in team | `["agent_1", "agent_2", "agent_3"]` |

**Framework Examples**:

- **CrewAI**: `Crew(agents=[...], process=Process.SEQUENTIAL)`
- **Autogen**: `GroupChat(participants=[...], max_rounds=10)`
- **Agno**: Multi-agent teams with shared state

---

### 2.2 `gen_ai.team.execute`

**Description**: Execution of a multi-agent team workflow.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.team.id` | string | Team identifier | `"team_research"` |
| `gen_ai.team.name` | string | Team name | `"Research Team"` |
| `gen_ai.workflow.type` | string | Type of workflow | `"sequential"`, `"hierarchical"`, `"parallel"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.id` | string | Workflow instance ID | `"workflow_run_456"` |
| `gen_ai.workflow.status` | string | Execution status | `"running"`, `"completed"`, `"failed"`, `"paused"` |
| `gen_ai.runtime.total_duration_ms` | int | Total execution time | `45000` |
| `gen_ai.runtime.total_tokens` | int | Total tokens consumed | `5000` |
| `gen_ai.team.rounds_completed` | int | Conversation rounds completed | `3` |
| `error.type` | string | Error type if failed | `"coordination_error"`, `"timeout"` |

---

### 2.3 `gen_ai.team.coordinate`

**Description**: Coordination action between team members (e.g., turn selection, task routing).

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.team.id` | string | Team identifier | `"team_research"` |
| `gen_ai.team.coordination_type` | string | Type of coordination | `"turn_selection"`, `"task_routing"`, `"conflict_resolution"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.team.current_speaker` | string | Agent currently active | `"agent_researcher"` |
| `gen_ai.team.next_speaker` | string | Next agent to activate | `"agent_writer"` |
| `gen_ai.team.selection_method` | string | How next speaker selected | `"round_robin"`, `"llm_selected"`, `"manual"` |

**Framework Examples**:

- **Autogen**: GroupChatManager selecting next speaker
- **CrewAI**: Hierarchical process with manager delegation

---

### 2.4 `gen_ai.workflow.execute`

**Description**: Execution of a workflow, graph, or orchestration pattern.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.id` | string | Workflow identifier | `"workflow_123"`, `"graph_main"` |
| `gen_ai.workflow.name` | string | Workflow name | `"Research Pipeline"`, `"RAG Workflow"` |
| `gen_ai.workflow.type` | string | Workflow pattern | `"graph"`, `"sequential"`, `"parallel"`, `"loop"`, `"conditional"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.status` | string | Execution status | `"running"`, `"completed"`, `"failed"`, `"interrupted"` |
| `gen_ai.workflow.total_nodes` | int | Number of nodes/steps | `10` |
| `gen_ai.workflow.execution_path` | string[] | Nodes executed in order | `["start", "retrieve", "grade", "generate", "end"]` |
| `gen_ai.workflow.depth` | int | Nesting depth (for nested workflows) | `1`, `2` |
| `gen_ai.team.id` | string | Associated team if applicable | `"team_research"` |
| `gen_ai.runtime.total_duration_ms` | int | Total execution time | `12000` |

**Framework Examples**:

- **LangGraph**: `StateGraph.invoke()` execution
- **Mastra**: `Workflow.execute()` with .then()/.branch()/.parallel()
- **CrewAI**: Flow execution with @start, @listen, @router
- **Google ADK**: SequentialAgent, ParallelAgent, LoopAgent execution

---

### 2.5 `gen_ai.workflow.transition`

**Description**: State transition between workflow nodes/steps.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.id` | string | Workflow identifier | `"workflow_123"` |
| `gen_ai.state.transition_from` | string | Source node/state | `"retrieve_docs"` |
| `gen_ai.state.transition_to` | string | Destination node/state | `"grade_docs"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.current_node` | string | Current node name | `"grade_docs"` |
| `gen_ai.state.current` | string (JSON) | Current state snapshot (truncated) | `"{\"messages\": [...], \"docs\": [...]}"` |
| `gen_ai.state.keys_changed` | string[] | State keys modified | `["documents", "relevance_scores"]` |
| `gen_ai.agent.id` | string | Agent executing this transition | `"agent_grader"` |

**Framework Examples**:

- **LangGraph**: Edge traversal in StateGraph with state updates
- **Mastra**: .then() or .branch() transitions

---

### 2.6 `gen_ai.workflow.branch`

**Description**: Conditional branching decision in workflow.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.id` | string | Workflow identifier | `"workflow_123"` |
| `gen_ai.workflow.branch_node` | string | Node where branching occurs | `"route_question"` |
| `gen_ai.workflow.branch_condition` | string | Condition evaluated | `"is_relevant"`, `"needs_retrieval"` |
| `gen_ai.workflow.branch_taken` | string | Which branch was taken | `"relevant_path"`, `"irrelevant_path"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.workflow.branch_options` | string[] | Available branches | `["relevant", "irrelevant", "uncertain"]` |
| `gen_ai.workflow.branch_reason` | string | Why this branch chosen | `"relevance_score > 0.8"` |

**Framework Examples**:

- **LangGraph**: Conditional edges with routing functions
- **Mastra**: `.branch()` with conditional logic
- **CrewAI**: `@router` decorator in Flows

---

## 3. Task Execution Spans

### 3.1 `gen_ai.task.create`

**Description**: Creation and definition of a task.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.id` | string | Unique task identifier | `"task_123"`, `"research_task_1"` |
| `gen_ai.task.name` | string | Task name | `"Research AI trends"`, `"Write summary"` |
| `gen_ai.task.type` | string | Task category | `"research"`, `"analysis"`, `"generation"`, `"review"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.description` | string | Detailed task description | `"Research latest developments in AI and summarize"` |
| `gen_ai.task.assigned_agent` | string | Agent assigned to task | `"agent_researcher"` |
| `gen_ai.task.parent_task_id` | string | Parent task (for hierarchies) | `"task_parent_1"` |
| `gen_ai.task.priority` | int | Task priority (higher = more urgent) | `1`, `5`, `10` |
| `gen_ai.task.deadline` | timestamp | Task deadline | `2025-01-24T10:00:00Z` |
| `gen_ai.task.expected_output` | string | Expected output specification | `"A markdown report with 3 sections"` |

**Framework Examples**:

- **CrewAI**: `@task` decorator with description, expected_output, agent
- **Google ADK**: Task submitted to agent with instructions

---

### 3.2 `gen_ai.task.execute`

**Description**: Execution of a task by an agent.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.id` | string | Task identifier | `"task_123"` |
| `gen_ai.task.name` | string | Task name | `"Research AI trends"` |
| `gen_ai.task.status` | string | Execution status | `"running"`, `"completed"`, `"failed"`, `"pending"` |
| `gen_ai.agent.id` | string | Executing agent | `"agent_researcher"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.type` | string | Task category | `"research"`, `"analysis"` |
| `gen_ai.runtime.duration_ms` | int | Task execution time | `15000` |
| `gen_ai.runtime.tool_calls_count` | int | Tools invoked during task | `7` |
| `gen_ai.runtime.iterations` | int | Agent loop iterations | `3` |
| `gen_ai.artifact.id` | string | Produced artifact ID | `"artifact_report_1"` |
| `gen_ai.artifact.type` | string | Type of artifact produced | `"text/markdown"`, `"application/json"` |
| `error.type` | string | Error type if failed | `"tool_error"`, `"timeout"` |

---

### 3.3 `gen_ai.task.delegate`

**Description**: Delegation of a task from one agent to another.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.id` | string | Task being delegated | `"task_123"` |
| `gen_ai.task.name` | string | Task name | `"Review code"` |
| `gen_ai.handoff.source_agent` | string | Agent delegating | `"agent_manager"` |
| `gen_ai.handoff.target_agent` | string | Agent receiving task | `"agent_reviewer"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.task.parent_task_id` | string | Parent task | `"task_parent_1"` |
| `gen_ai.handoff.reason` | string | Why delegated | `"expertise_required"`, `"workload_balancing"` |

---

### 3.4 `gen_ai.agent.handoff`

**Description**: Agent-to-agent handoff or delegation.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.handoff.source_agent` | string | Source agent ID/name | `"agent_triage"` |
| `gen_ai.handoff.target_agent` | string | Target agent ID/name | `"agent_specialist"` |
| `gen_ai.handoff.timestamp` | timestamp | When handoff occurred | `2025-01-23T10:35:00Z` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.handoff.reason` | string | Semantic reason | `"expertise_required"`, `"critique_requested"`, `"escalation"` |
| `gen_ai.handoff.intent` | string | Short description of intent | `"summarize"`, `"classify"`, `"review"` |
| `gen_ai.handoff.type` | string | Collaboration pattern | `"delegation"`, `"transfer"`, `"escalation"`, `"broadcast"` |
| `gen_ai.handoff.context_transferred` | boolean | Whether context was passed | `true`, `false` |
| `gen_ai.handoff.arguments_json` | string (JSON) | Arguments passed (redacted) | `"{\"context\": \"...\", \"task\": \"...\"}"` |
| `gen_ai.handoff.response_summary` | string | Short summary of response | `"Analysis completed successfully"` |
| `gen_ai.session.id` | string | Associated session | `"sess_abc123"` |
| `gen_ai.task.id` | string | Associated task | `"task_456"` |

**Framework Examples**:

- **OpenAI SDK**: Handoff mechanism as core primitive
- **Google ADK**: Sub-agent delegation or AgentTool invocation
- **Autogen**: Message passing between agents in GroupChat

---

## 4. Memory Spans

### 4.1 `gen_ai.memory.store`

**Description**: Storing data to memory system.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.operation` | string | Always `"store"` | `"store"` |
| `gen_ai.memory.type` | string | Memory type | `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"`, `"procedural"` |
| `gen_ai.memory.store` | string | Memory backend | `"chromadb"`, `"sqlite"`, `"redis"`, `"in_memory"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.session_id` | string | Associated session | `"sess_abc123"` |
| `gen_ai.memory.actor_id` | string | Agent storing memory | `"agent_123"` |
| `gen_ai.memory.items_stored` | int | Number of items stored | `3` |
| `gen_ai.memory.size_bytes` | int | Approximate size | `4096` |
| `gen_ai.memory.ttl_seconds` | int | Time-to-live | `3600`, `86400` |
| `gen_ai.memory.embedding_model` | string | Embedding model if vector | `"text-embedding-3-small"` |
| `gen_ai.memory.namespace` | string | Memory namespace | `"user_123_memories"` |

**Framework Examples**:

- **CrewAI**: Memory writes to ChromaDB (short/long/entity/user memory)
- **Google ADK**: Memory service store operations
- **Agno**: Semantic/episodic/procedural memory storage

---

### 4.2 `gen_ai.memory.retrieve`

**Description**: Retrieving data from memory system.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.operation` | string | Always `"retrieve"` | `"retrieve"` |
| `gen_ai.memory.type` | string | Memory type | `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"` |
| `gen_ai.memory.store` | string | Memory backend | `"chromadb"`, `"sqlite"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.session_id` | string | Associated session | `"sess_abc123"` |
| `gen_ai.memory.actor_id` | string | Agent retrieving memory | `"agent_123"` |
| `gen_ai.memory.items_retrieved` | int | Number of items retrieved | `5` |
| `gen_ai.memory.relevance_score` | float | Minimum relevance score | `0.75` |
| `gen_ai.memory.hit` | boolean | Whether memory hit occurred | `true`, `false` |

---

### 4.3 `gen_ai.memory.search`

**Description**: Searching memory with semantic query.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.operation` | string | Always `"search"` | `"search"` |
| `gen_ai.memory.type` | string | Memory type | `"semantic"`, `"episodic"`, `"vector"` |
| `gen_ai.memory.search.query` | string | Search query text | `"Previous conversations about pricing"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.store` | string | Memory backend | `"chromadb"`, `"pinecone"`, `"weaviate"` |
| `gen_ai.memory.search.top_k` | int | Number of results requested | `5`, `10` |
| `gen_ai.memory.search.min_score` | float | Minimum relevance threshold | `0.7` |
| `gen_ai.memory.search.filters` | string (JSON) | Search filters | `"{\"user_id\": \"123\", \"date_after\": \"2025-01-01\"}"` |
| `gen_ai.memory.items_retrieved` | int | Actual results returned | `3` |
| `gen_ai.memory.vector_dimensions` | int | Embedding dimensions | `1536`, `768` |

**Framework Examples**:

- **Agno**: Semantic memory search with similarity matching
- **LlamaIndex**: Vector store retrieval with top_k and filters
- **Google ADK**: VertexAiMemoryBankService semantic search

---

### 4.4 `gen_ai.memory.update`

**Description**: Updating existing memory entries.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.operation` | string | Always `"update"` | `"update"` |
| `gen_ai.memory.type` | string | Memory type | `"long_term"`, `"semantic"` |
| `gen_ai.memory.store` | string | Memory backend | `"sqlite"`, `"redis"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.items_updated` | int | Number of items updated | `2` |
| `gen_ai.memory.keys` | string[] | Memory keys updated | `["pref_timezone", "pref_language"]` |

---

### 4.5 `gen_ai.memory.delete`

**Description**: Deleting memory entries.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.operation` | string | Always `"delete"` | `"delete"` |
| `gen_ai.memory.type` | string | Memory type | `"short_term"`, `"episodic"` |
| `gen_ai.memory.store` | string | Memory backend | `"chromadb"`, `"redis"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.memory.items_deleted` | int | Number of items deleted | `10` |
| `gen_ai.memory.keys` | string[] | Memory keys deleted | `["session_123_messages"]` |

---

## 5. Tool & Integration Spans

### 5.1 `gen_ai.tool.execute`

**Description**: Execution of a tool, function, or capability by an agent.

**Span Kind**: `CLIENT`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.tool.name` | string | Tool/function name | `"web_search"`, `"calculator"`, `"read_file"` |
| `gen_ai.tool.type` | string | Tool category | `"api"`, `"function"`, `"code"`, `"mcp"`, `"native"`, `"browser"` |
| `gen_ai.operation.name` | string | Operation performed | `"execute"`, `"invoke"`, `"call"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.tool.id` | string | Tool instance identifier | `"tool_search_123"` |
| `gen_ai.tool.category` | string | Functional category | `"search"`, `"computation"`, `"io"`, `"communication"` |
| `gen_ai.tool.provider` | string | Tool provider | `"serper"`, `"tavily"`, `"google"` |
| `gen_ai.tool.version` | string | Tool version | `"1.0.0"`, `"v2"` |
| `gen_ai.tool.invocation_id` | string | Unique invocation ID | `"inv_abc123"` |
| `gen_ai.tool.parameters` | string (JSON) | Input parameters (redacted) | `"{\"query\": \"...\", \"max_results\": 5}"` |
| `gen_ai.tool.result` | string (JSON) | Tool result (truncated) | `"{\"results\": [...]}"` |
| `gen_ai.tool.duration_ms` | int | Execution duration | `1500` |
| `gen_ai.tool.selection_method` | string | How tool was selected | `"model_generated"`, `"forced"`, `"heuristic"`, `"human_in_the_loop"` |
| `gen_ai.tool.error_strategy` | string | Error handling approach | `"retry"`, `"fallback"`, `"ignore"`, `"terminate"` |
| `gen_ai.tool.retry_count` | int | Current retry attempt | `0`, `1`, `2` |
| `gen_ai.agent.id` | string | Agent using tool | `"agent_researcher"` |
| `error.type` | string | Error type if failed | `"timeout"`, `"rate_limit"`, `"auth_error"` |

**Framework Examples**:

- **All frameworks**: Function calling, custom tools, built-in tools
- **Agno**: 100+ toolkits with thousands of tools
- **Haystack**: ComponentTool wrapping pipeline components
- **Smolagents**: CodeAgent generates Python code to invoke tools

---

### 5.2 `gen_ai.mcp.connect`

**Description**: Connection to Model Context Protocol (MCP) server.

**Span Kind**: `CLIENT`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.mcp.server_name` | string | MCP server name | `"filesystem-server"`, `"github-mcp"` |
| `gen_ai.mcp.transport` | string | Transport protocol | `"stdio"`, `"http"`, `"websocket"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.mcp.protocol_version` | string | MCP protocol version | `"1.0"`, `"2024-11-05"` |
| `gen_ai.mcp.capabilities` | string[] | Server capabilities | `["tools", "resources", "prompts"]` |
| `server.address` | string | Server host | `"localhost"`, `"mcp.example.com"` |
| `server.port` | int | Server port | `3000`, `8080` |

**Framework Examples**:

- **Agno**: First-class MCP support via MCPTools
- **Autogen**: McpWorkbench integration
- **Mastra**: MCP tool integration
- **Google ADK**: MCP tool support

---

### 5.3 `gen_ai.mcp.execute`

**Description**: Execution of MCP command/tool.

**Span Kind**: `CLIENT`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.mcp.server_name` | string | MCP server name | `"filesystem-server"` |
| `gen_ai.tool.name` | string | Tool/command name | `"read_file"`, `"list_directory"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.tool.parameters` | string (JSON) | Command parameters | `"{\"path\": \"/data/file.txt\"}"` |
| `gen_ai.tool.result` | string (JSON) | Command result | `"{\"content\": \"...\"}"` |
| `gen_ai.tool.duration_ms` | int | Execution time | `250` |
| `error.type` | string | Error type if failed | `"file_not_found"`, `"permission_denied"` |

---

## 6. Context & State Spans

### 6.1 `gen_ai.context.checkpoint`

**Description**: Checkpointing of conversation/workflow state (especially relevant for LangGraph).

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.context.checkpoint_id` | string | Unique checkpoint identifier | `"checkpoint_789"`, `"ckpt_abc123"` |
| `gen_ai.session.id` | string | Associated session | `"sess_abc123"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.context.state_size_bytes` | int | Size of checkpointed state | `8192` |
| `gen_ai.state.checkpoint_saved` | boolean | Whether save succeeded | `true`, `false` |
| `gen_ai.workflow.id` | string | Associated workflow | `"workflow_123"` |
| `gen_ai.context.checkpoint_backend` | string | Storage backend | `"sqlite"`, `"postgres"`, `"memory"` |

**Framework Examples**:

- **LangGraph**: Automatic checkpointing after each super-step
- **Google ADK**: Session state persistence
- **AgentCore**: Session persistence across invocations

---

### 6.2 `gen_ai.context.compress`

**Description**: Compression of context to manage token limits.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.context.compression_enabled` | boolean | Whether compression active | `true`, `false` |
| `gen_ai.context.compression_ratio` | float | Compression ratio achieved | `0.5`, `0.75` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.context.window_size` | int | Context window size | `8192`, `128000` |
| `gen_ai.context.tokens_before` | int | Tokens before compression | `16000` |
| `gen_ai.context.tokens_after` | int | Tokens after compression | `8000` |
| `gen_ai.context.compression_method` | string | Compression technique | `"summarization"`, `"truncation"`, `"rolling_window"` |
| `gen_ai.session.id` | string | Associated session | `"sess_abc123"` |

**Framework Examples**:

- **Google ADK**: Automatic context compression
- **LlamaIndex**: Context window management
- **Most frameworks**: Token limit handling

---

## 7. Quality & Control Spans

### 7.1 `gen_ai.guardrail.check`

**Description**: Execution of a guardrail or safety check.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.guardrail.name` | string | Guardrail identifier | `"pii_detector"`, `"toxicity_filter"`, `"prompt_injection_check"` |
| `gen_ai.guardrail.type` | string | Guardrail category | `"input_validation"`, `"output_validation"`, `"content_moderation"`, `"safety"` |
| `gen_ai.guardrail.triggered` | boolean | Whether guardrail activated | `true`, `false` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.guardrail.action` | string | Action taken when triggered | `"block"`, `"warn"`, `"modify"`, `"log"` |
| `gen_ai.guardrail.confidence` | float | Confidence score | `0.95` |
| `gen_ai.guardrail.policy_id` | string | Associated policy | `"policy_content_safety"` |
| `gen_ai.guardrail.violation_type` | string | Type of violation detected | `"pii_present"`, `"toxic_content"`, `"jailbreak_attempt"` |
| `gen_ai.agent.id` | string | Agent being guarded | `"agent_123"` |

**Framework Examples**:

- **Agno**: Built-in guardrails (PII detection, prompt injection, content moderation)
- **CrewAI**: Input/output validation guardrails
- **AgentCore**: Safety checks in managed runtime

---

### 7.2 `gen_ai.eval.execute`

**Description**: Runtime evaluation or quality assessment.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.eval.criteria` | string | What's being evaluated | `"faithfulness"`, `"relevance"`, `"toxicity"`, `"code_correctness"`, `"answer_quality"` |
| `gen_ai.eval.method` | string | Evaluation method | `"llm_judge"`, `"heuristic"`, `"human_feedback"`, `"rule_based"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.eval.score` | float | Quantitative score (0.0-1.0) | `0.85`, `0.92` |
| `gen_ai.eval.passed` | boolean | Whether threshold met | `true`, `false` |
| `gen_ai.eval.threshold` | float | Pass/fail threshold | `0.7` |
| `gen_ai.eval.feedback` | string | Textual feedback | `"Response is accurate but lacks detail"` |
| `gen_ai.eval.model` | string | Judge model if LLM-based | `"gpt-4"`, `"claude-3-opus"` |
| `gen_ai.agent.id` | string | Agent being evaluated | `"agent_writer"` |
| `gen_ai.task.id` | string | Task being evaluated | `"task_123"` |

**Framework Examples**:

- **CrewAI**: Built-in evals via CrewAI AMP Suite
- **Google ADK**: Quality signal tracking with Maxim AI integration
- **Agno**: Accuracy, performance, reliability metrics

---

### 7.3 `gen_ai.human.review`

**Description**: Human-in-the-loop review or approval.

**Span Kind**: `INTERNAL`

**Required Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.human.approval_required` | boolean | Whether approval needed | `true`, `false` |
| `gen_ai.human.intervention_type` | string | Type of human intervention | `"approval"`, `"feedback"`, `"correction"`, `"guidance"` |

**Optional Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.human.approval_granted` | boolean | Whether approved | `true`, `false` |
| `gen_ai.human.feedback` | string | Human feedback text | `"Looks good, proceed"` |
| `gen_ai.human.response_time_ms` | int | How long human took | `45000` |
| `gen_ai.human.reviewer_id` | string | Reviewer identifier (hashed) | `"reviewer_hash_xyz"` |
| `gen_ai.agent.id` | string | Agent awaiting approval | `"agent_executor"` |
| `gen_ai.task.id` | string | Task being reviewed | `"task_456"` |
| `gen_ai.tool.name` | string | Tool awaiting approval | `"send_email"` |

**Framework Examples**:

- **Google ADK**: Tool confirmation flow (HITL)
- **Agno**: Human approval for sensitive operations
- **CrewAI**: HITL workflows
- **LangGraph**: Breakpoints for human review

---

## 8. Existing GenAI Spans (Extended)

These spans are defined in the existing [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) and are used within agent systems:

### 8.1 `gen_ai.client.{operation}`

**Description**: LLM client operations (chat, completion, embedding, etc.)

**Span Kind**: `CLIENT`

**Key Attributes** (from existing GenAI semconv):

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.system` | string | Provider: `"openai"`, `"anthropic"`, `"google"`, etc. |
| `gen_ai.request.model` | string | Requested model name |
| `gen_ai.response.model` | string | Actual model used |
| `gen_ai.request.temperature` | float | Sampling temperature |
| `gen_ai.request.top_p` | float | Nucleus sampling parameter |
| `gen_ai.request.max_tokens` | int | Maximum tokens to generate |
| `gen_ai.usage.input_tokens` | int | Prompt tokens consumed |
| `gen_ai.usage.output_tokens` | int | Completion tokens generated |
| `gen_ai.usage.total_tokens` | int | Total tokens |

**Agent-Specific Extensions**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `gen_ai.agent.id` | string | Agent making this LLM call | `"agent_123"` |
| `gen_ai.task.id` | string | Task context if applicable | `"task_456"` |
| `gen_ai.llm.is_tool_call` | boolean | Whether this call involves tool use | `true`, `false` |

---

## Attribute Registry

### Complete Attribute Reference

Below is the complete registry of all `gen_ai.*` attributes introduced for agent observability, organized by namespace.

#### gen_ai.agent.* (Agent Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.agent.id` | string | Required | Unique agent instance identifier | `"agent_123"`, `"researcher_01"` |
| `gen_ai.agent.name` | string | Required | Human-readable agent name | `"TravelAssistant"`, `"CodeReviewer"` |
| `gen_ai.agent.type` | string | Required | Agent implementation type | `"react"`, `"function_calling"`, `"conversational"` |
| `gen_ai.agent.framework` | string | Required | Framework used | `"langgraph"`, `"crewai"`, `"llamaindex"` |
| `gen_ai.agent.framework.version` | string | Optional | Framework version | `"0.2.0"`, `"1.5.3"` |
| `gen_ai.agent.role` | string | Optional | Agent's role/persona | `"Senior Python Engineer"`, `"Support Specialist"` |
| `gen_ai.agent.goal` | string | Optional | Agent's objective | `"Research AI trends"` |
| `gen_ai.agent.backstory` | string | Optional | Agent personality/context | `"You're a seasoned researcher..."` |
| `gen_ai.agent.mode` | string | Optional | Architectural pattern | `"react"`, `"plan_and_solve"`, `"autonomous"`, `"supervisor"`, `"code_interpreter"` |
| `gen_ai.agent.version` | string | Optional | Agent version | `"1.0.0"`, `"v2"` |
| `gen_ai.agent.capabilities` | string[] | Optional | Agent capabilities | `["web_search", "code_execution"]` |
| `gen_ai.agent.tools` | string[] | Optional | Available tool names | `["calculator", "search_web"]` |
| `gen_ai.agent.memory_enabled` | boolean | Optional | Memory enabled | `true`, `false` |
| `gen_ai.agent.delegation_enabled` | boolean | Optional | Can delegate | `true`, `false` |
| `gen_ai.agent.max_iterations` | int | Optional | Max iterations | `10`, `50` |
| `gen_ai.agent.timeout_ms` | int | Optional | Timeout in ms | `30000`, `60000` |
| `gen_ai.agent.termination_reason` | string | Optional | Why terminated | `"completed"`, `"error"`, `"timeout"` |

#### gen_ai.team.* (Multi-Agent Team Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.team.id` | string | Required | Unique team identifier | `"team_research"`, `"crew_123"` |
| `gen_ai.team.name` | string | Required | Team name | `"Research Team"` |
| `gen_ai.team.size` | int | Required | Number of agents | `3`, `5` |
| `gen_ai.team.orchestration_pattern` | string | Required | Coordination pattern | `"sequential"`, `"hierarchical"`, `"round_robin"` |
| `gen_ai.team.manager_agent_id` | string | Optional | Manager agent | `"agent_manager_1"` |
| `gen_ai.team.agents` | string[] | Optional | List of agent IDs | `["agent_1", "agent_2"]` |
| `gen_ai.team.coordination_type` | string | Optional | Type of coordination | `"turn_selection"`, `"task_routing"` |
| `gen_ai.team.current_speaker` | string | Optional | Currently active agent | `"agent_researcher"` |
| `gen_ai.team.next_speaker` | string | Optional | Next agent to activate | `"agent_writer"` |
| `gen_ai.team.selection_method` | string | Optional | Speaker selection method | `"round_robin"`, `"llm_selected"` |
| `gen_ai.team.rounds_completed` | int | Optional | Conversation rounds | `3` |

#### gen_ai.task.* (Task Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.task.id` | string | Required | Unique task identifier | `"task_123"` |
| `gen_ai.task.name` | string | Required | Task name | `"Research AI trends"` |
| `gen_ai.task.type` | string | Required | Task category | `"research"`, `"analysis"`, `"generation"` |
| `gen_ai.task.status` | string | Required (on execute) | Execution status | `"running"`, `"completed"`, `"failed"` |
| `gen_ai.task.description` | string | Optional | Detailed description | `"Research latest AI developments"` |
| `gen_ai.task.assigned_agent` | string | Optional | Assigned agent | `"agent_researcher"` |
| `gen_ai.task.parent_task_id` | string | Optional | Parent task | `"task_parent_1"` |
| `gen_ai.task.priority` | int | Optional | Task priority | `1`, `5`, `10` |
| `gen_ai.task.deadline` | timestamp | Optional | Task deadline | `2025-01-24T10:00:00Z` |
| `gen_ai.task.expected_output` | string | Optional | Expected output spec | `"A markdown report"` |

#### gen_ai.tool.* (Tool Attributes - Extends Existing)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.tool.name` | string | Required | Tool/function name | `"web_search"`, `"calculator"` |
| `gen_ai.tool.type` | string | Required | Tool category | `"api"`, `"function"`, `"code"`, `"mcp"` |
| `gen_ai.tool.id` | string | Optional | Tool instance ID | `"tool_search_123"` |
| `gen_ai.tool.category` | string | Optional | Functional category | `"search"`, `"computation"`, `"io"` |
| `gen_ai.tool.provider` | string | Optional | Tool provider | `"serper"`, `"tavily"`, `"google"` |
| `gen_ai.tool.version` | string | Optional | Tool version | `"1.0.0"` |
| `gen_ai.tool.invocation_id` | string | Optional | Invocation ID | `"inv_abc123"` |
| `gen_ai.tool.parameters` | string (JSON) | Optional | Input parameters | `"{\"query\": \"...\"}"` |
| `gen_ai.tool.result` | string (JSON) | Optional | Tool result | `"{\"results\": [...]}"` |
| `gen_ai.tool.duration_ms` | int | Optional | Execution time | `1500` |
| `gen_ai.tool.selection_method` | string | Optional | How selected | `"model_generated"`, `"forced"`, `"heuristic"` |
| `gen_ai.tool.error_strategy` | string | Optional | Error handling | `"retry"`, `"fallback"`, `"ignore"` |
| `gen_ai.tool.retry_count` | int | Optional | Retry attempt | `0`, `1`, `2` |

#### gen_ai.mcp.* (MCP Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.mcp.server_name` | string | Required | MCP server name | `"filesystem-server"`, `"github-mcp"` |
| `gen_ai.mcp.transport` | string | Required | Transport protocol | `"stdio"`, `"http"`, `"websocket"` |
| `gen_ai.mcp.protocol_version` | string | Optional | Protocol version | `"1.0"`, `"2024-11-05"` |
| `gen_ai.mcp.capabilities` | string[] | Optional | Server capabilities | `["tools", "resources", "prompts"]` |
| `gen_ai.mcp.server.version` | string | Optional | Server version | `"1.2.0"` |

#### gen_ai.memory.* (Memory Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.memory.operation` | string | Required | Operation type | `"store"`, `"retrieve"`, `"search"`, `"update"`, `"delete"` |
| `gen_ai.memory.type` | string | Required | Memory type | `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"`, `"procedural"` |
| `gen_ai.memory.store` | string | Required | Storage backend | `"chromadb"`, `"sqlite"`, `"redis"`, `"in_memory"` |
| `gen_ai.memory.session_id` | string | Optional | Associated session | `"sess_abc123"` |
| `gen_ai.memory.actor_id` | string | Optional | Agent accessing memory | `"agent_123"` |
| `gen_ai.memory.items_stored` | int | Optional | Items stored | `3` |
| `gen_ai.memory.items_retrieved` | int | Optional | Items retrieved | `5` |
| `gen_ai.memory.items_updated` | int | Optional | Items updated | `2` |
| `gen_ai.memory.items_deleted` | int | Optional | Items deleted | `10` |
| `gen_ai.memory.size_bytes` | int | Optional | Size in bytes | `4096` |
| `gen_ai.memory.ttl_seconds` | int | Optional | Time-to-live | `3600` |
| `gen_ai.memory.embedding_model` | string | Optional | Embedding model | `"text-embedding-3-small"` |
| `gen_ai.memory.vector_dimensions` | int | Optional | Vector dimensions | `1536`, `768` |
| `gen_ai.memory.namespace` | string | Optional | Memory namespace | `"user_123_memories"` |
| `gen_ai.memory.relevance_score` | float | Optional | Min relevance | `0.75` |
| `gen_ai.memory.hit` | boolean | Optional | Cache hit | `true`, `false` |
| `gen_ai.memory.search.query` | string | Required (search) | Search query | `"Previous pricing conversations"` |
| `gen_ai.memory.search.top_k` | int | Optional | Results requested | `5`, `10` |
| `gen_ai.memory.search.min_score` | float | Optional | Min score threshold | `0.7` |
| `gen_ai.memory.search.filters` | string (JSON) | Optional | Search filters | `"{\"user_id\": \"123\"}"` |
| `gen_ai.memory.keys` | string[] | Optional | Memory keys accessed | `["pref_timezone"]` |

#### gen_ai.session.* (Session Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.session.id` | string | Required | Unique session ID | `"sess_abc123"` |
| `gen_ai.session.start_time` | timestamp | Required | Session start | `2025-01-23T10:30:00Z` |
| `gen_ai.session.type` | string | Optional | Session category | `"chat"`, `"autonomous_run"`, `"batch"` |
| `gen_ai.session.thread_id` | string | Optional | Thread ID | `"thread_789"` |
| `gen_ai.session.user_id` | string | Optional | User ID (hashed) | `"user_hash_xyz"` |
| `gen_ai.session.persistent` | boolean | Optional | State persists | `true`, `false` |
| `gen_ai.session.message_count` | int | Optional | Total messages | `15` |
| `gen_ai.session.turn_count` | int | Optional | Conversation turns | `7` |
| `gen_ai.session.start_reason` | string | Optional | Trigger reason | `"user_message"`, `"scheduled_task"` |

#### gen_ai.context.* (Context Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.context.checkpoint_id` | string | Required (checkpoint) | Checkpoint ID | `"checkpoint_789"` |
| `gen_ai.context.state_size_bytes` | int | Optional | State size | `8192` |
| `gen_ai.context.checkpoint_backend` | string | Optional | Storage backend | `"sqlite"`, `"postgres"` |
| `gen_ai.context.window_size` | int | Optional | Context window | `8192`, `128000` |
| `gen_ai.context.tokens_used` | int | Optional | Tokens in use | `4000` |
| `gen_ai.context.tokens_before` | int | Optional | Before compression | `16000` |
| `gen_ai.context.tokens_after` | int | Optional | After compression | `8000` |
| `gen_ai.context.compression_enabled` | boolean | Required (compress) | Compression active | `true`, `false` |
| `gen_ai.context.compression_ratio` | float | Required (compress) | Compression ratio | `0.5`, `0.75` |
| `gen_ai.context.compression_method` | string | Optional | Compression technique | `"summarization"`, `"truncation"` |
| `gen_ai.context.window_usage_pct` | float | Optional | Window utilization % | `0.65`, `0.95` |

#### gen_ai.workflow.* (Workflow Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.workflow.id` | string | Required | Workflow ID | `"workflow_123"` |
| `gen_ai.workflow.name` | string | Required | Workflow name | `"Research Pipeline"` |
| `gen_ai.workflow.type` | string | Required | Workflow pattern | `"graph"`, `"sequential"`, `"parallel"` |
| `gen_ai.workflow.status` | string | Optional | Execution status | `"running"`, `"completed"`, `"failed"` |
| `gen_ai.workflow.total_nodes` | int | Optional | Number of nodes | `10` |
| `gen_ai.workflow.execution_path` | string[] | Optional | Nodes executed | `["start", "retrieve", "generate"]` |
| `gen_ai.workflow.current_node` | string | Optional | Current node | `"grade_docs"` |
| `gen_ai.workflow.depth` | int | Optional | Nesting depth | `1`, `2` |
| `gen_ai.workflow.branch_node` | string | Required (branch) | Branch node name | `"route_question"` |
| `gen_ai.workflow.branch_condition` | string | Required (branch) | Condition evaluated | `"is_relevant"` |
| `gen_ai.workflow.branch_taken` | string | Required (branch) | Branch selected | `"relevant_path"` |
| `gen_ai.workflow.branch_options` | string[] | Optional | Available branches | `["relevant", "irrelevant"]` |
| `gen_ai.workflow.branch_reason` | string | Optional | Branch reason | `"relevance_score > 0.8"` |

#### gen_ai.state.* (State Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.state.current` | string (JSON) | Optional | Current state (truncated) | `"{\"messages\": [...], \"docs\": []}"` |
| `gen_ai.state.keys_changed` | string[] | Optional | Modified keys | `["documents", "relevance_scores"]` |
| `gen_ai.state.transition_from` | string | Required (transition) | Source node/state | `"retrieve_docs"` |
| `gen_ai.state.transition_to` | string | Required (transition) | Target node/state | `"grade_docs"` |
| `gen_ai.state.checkpoint_saved` | boolean | Optional | Checkpoint success | `true`, `false` |

#### gen_ai.handoff.* (Agent Handoff Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.handoff.source_agent` | string | Required | Source agent | `"agent_triage"` |
| `gen_ai.handoff.target_agent` | string | Required | Target agent | `"agent_specialist"` |
| `gen_ai.handoff.timestamp` | timestamp | Required | Handoff time | `2025-01-23T10:35:00Z` |
| `gen_ai.handoff.reason` | string | Optional | Semantic reason | `"expertise_required"`, `"escalation"` |
| `gen_ai.handoff.intent` | string | Optional | Intent description | `"summarize"`, `"classify"` |
| `gen_ai.handoff.type` | string | Optional | Collaboration pattern | `"delegation"`, `"transfer"`, `"broadcast"` |
| `gen_ai.handoff.context_transferred` | boolean | Optional | Context passed | `true`, `false` |
| `gen_ai.handoff.arguments_json` | string (JSON) | Optional | Arguments passed | `"{\"context\": \"...\"}"` |
| `gen_ai.handoff.response_summary` | string | Optional | Response summary | `"Analysis completed"` |

#### gen_ai.artifact.* (Artifact Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.artifact.id` | string | Optional | Artifact ID | `"artifact_report_1"` |
| `gen_ai.artifact.type` | string | Optional | MIME type | `"text/markdown"`, `"application/json"`, `"image/png"` |
| `gen_ai.artifact.size_bytes` | int | Optional | Artifact size | `8192` |
| `gen_ai.artifact.uri` | string | Optional | Storage location | `"s3://bucket/file.md"`, `"/tmp/artifact.json"` |
| `gen_ai.artifact.description` | string | Optional | Description | `"Research report on AI trends"` |

#### gen_ai.guardrail.* (Guardrail Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.guardrail.name` | string | Required | Guardrail name | `"pii_detector"`, `"toxicity_filter"` |
| `gen_ai.guardrail.type` | string | Required | Guardrail category | `"input_validation"`, `"output_validation"`, `"safety"` |
| `gen_ai.guardrail.triggered` | boolean | Required | Activation status | `true`, `false` |
| `gen_ai.guardrail.action` | string | Optional | Action taken | `"block"`, `"warn"`, `"modify"` |
| `gen_ai.guardrail.confidence` | float | Optional | Confidence score | `0.95` |
| `gen_ai.guardrail.policy_id` | string | Optional | Policy ID | `"policy_content_safety"` |
| `gen_ai.guardrail.violation_type` | string | Optional | Violation type | `"pii_present"`, `"toxic_content"` |

#### gen_ai.eval.* (Evaluation Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.eval.criteria` | string | Required | Evaluation dimension | `"faithfulness"`, `"relevance"`, `"toxicity"` |
| `gen_ai.eval.method` | string | Required | Evaluation method | `"llm_judge"`, `"heuristic"`, `"human_feedback"` |
| `gen_ai.eval.score` | float | Optional | Quantitative score (0.0-1.0) | `0.85`, `0.92` |
| `gen_ai.eval.passed` | boolean | Optional | Threshold met | `true`, `false` |
| `gen_ai.eval.threshold` | float | Optional | Pass threshold | `0.7` |
| `gen_ai.eval.feedback` | string | Optional | Textual feedback | `"Response is accurate"` |
| `gen_ai.eval.model` | string | Optional | Judge model | `"gpt-4"`, `"claude-3-opus"` |

#### gen_ai.human.* (Human-in-the-Loop Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.human.approval_required` | boolean | Required | Approval needed | `true`, `false` |
| `gen_ai.human.intervention_type` | string | Required | Intervention type | `"approval"`, `"feedback"`, `"correction"` |
| `gen_ai.human.approval_granted` | boolean | Optional | Approval status | `true`, `false` |
| `gen_ai.human.feedback` | string | Optional | Human feedback | `"Looks good, proceed"` |
| `gen_ai.human.response_time_ms` | int | Optional | Response time | `45000` |
| `gen_ai.human.reviewer_id` | string | Optional | Reviewer ID (hashed) | `"reviewer_hash_xyz"` |

#### gen_ai.runtime.* (Runtime Attributes)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.runtime.llm_calls_count` | int | Optional | LLM calls made | `3` |
| `gen_ai.runtime.tool_calls_count` | int | Optional | Tool invocations | `5` |
| `gen_ai.runtime.duration_ms` | int | Optional | Duration in ms | `4500` |
| `gen_ai.runtime.total_duration_ms` | int | Optional | Total duration | `125000` |
| `gen_ai.runtime.iterations` | int | Optional | Loop iterations | `3` |
| `gen_ai.runtime.total_invocations` | int | Optional | Total invocations | `15` |
| `gen_ai.runtime.total_tokens` | int | Optional | Total tokens | `5000` |

#### gen_ai.operation.name (Operation Name)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.operation.name` | string | Required (some spans) | Operation name | `"execute"`, `"run"`, `"process"`, `"invoke"` |

#### gen_ai.environment (Environment)

| Attribute | Type | Requirement | Description | Examples |
|-----------|------|-------------|-------------|----------|
| `gen_ai.environment` | string | Optional | Deployment environment | `"dev"`, `"staging"`, `"prod"` |

---

## Event Definitions

Events provide additional detail within spans, capturing specific moments or data points during execution.

### Cross-Cutting Events (Any Span)

These events can be emitted on any span type:

#### 1. `agent.thought`

**Description**: Captures internal reasoning or chain-of-thought.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `content` | string | Reasoning text | `"I need to use the search tool because..."` |
| `redacted` | boolean | Whether PII was redacted | `true`, `false` |

**Framework Examples**:
- Smolagents ReAct "thought" steps
- LlamaIndex ReActAgent reasoning
- Any agent with observable CoT

---

#### 2. `agent.plan`

**Description**: Captures agent's plan for solving a problem.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `steps_json` | string (JSON) | Planned steps | `"[{\"step\": 1, \"action\": \"search\"}, ...]"` |
| `plan_type` | string | Planning approach | `"sequential"`, `"decomposition"`, `"reactive"` |

**Framework Examples**:
- Google ADK plan-and-solve agents
- CrewAI planning phase

---

#### 3. `agent.observation`

**Description**: Captures agent's observation of environment/tool results.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `content` | string | Observation text | `"The search returned 5 results..."` |
| `source` | string | Observation source | `"tool_result"`, `"environment"`, `"human_feedback"` |

**Framework Examples**:
- Smolagents/LlamaIndex ReAct "observation" phase
- All frameworks with tool execution feedback

---

#### 4. `artifact.produced`

**Description**: Captures creation of an artifact.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `artifact_type` | string | Artifact MIME type | `"text/markdown"`, `"image/png"` |
| `size_bytes` | int | Artifact size | `8192` |
| `uri` | string | Storage location | `"s3://bucket/artifact.md"` |
| `description` | string | Artifact description | `"Generated research report"` |

---

#### 5. `exception` (Standard OTel Event)

**Description**: Standard OpenTelemetry exception event.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `exception.type` | string | Exception class/type |
| `exception.message` | string | Error message |
| `exception.stacktrace` | string | Full stack trace |

---

### LLM-Specific Events (on gen_ai.client.* spans)

#### 6. `llm.prompt`

**Description**: Captures prompt sent to LLM.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `content` | string | Prompt text (may be truncated/masked) | `"You are a helpful assistant..."` |
| `messages_json` | string (JSON) | Structured messages (optional) | `"[{\"role\": \"user\", \"content\": \"...\"}]"` |

---

#### 7. `llm.completion`

**Description**: Captures LLM completion/response.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `content` | string | Completion text (may be truncated/masked) | `"Based on my analysis..."` |
| `messages_json` | string (JSON) | Structured assistant messages (optional) | `"[{\"role\": \"assistant\", \"content\": \"...\"}]"` |

---

#### 8. `llm.token`

**Description**: Individual token generated (streaming).

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `token` | string | Token text | `"Hello"`, `" world"` |
| `token_index` | int | Token position | `0`, `1`, `2` |

---

#### 9. `llm.function_call`

**Description**: LLM requests function/tool call.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `function_name` | string | Function requested | `"web_search"`, `"calculator"` |
| `arguments_json` | string (JSON) | Function arguments | `"{\"query\": \"AI trends\"}"` |

---

### Tool-Specific Events (on gen_ai.tool.execute spans)

#### 10. `tool.request`

**Description**: Tool invocation request.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `body` | string | Request body (truncated/redacted) | `"{\"query\": \"...\"}"` |
| `headers_json` | string (JSON) | Request headers (optional) | `"{\"Content-Type\": \"application/json\"}"` |
| `method` | string | HTTP method if API | `"GET"`, `"POST"` |

---

#### 11. `tool.response`

**Description**: Tool execution response.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `body` | string | Response body (truncated) | `"{\"results\": [...]}"` |
| `status_code` | int | HTTP status if API | `200`, `404` |

---

#### 12. `tool.error`

**Description**: Tool execution error.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `error_type` | string | Error category | `"timeout"`, `"auth_error"`, `"rate_limit"` |
| `error_message` | string | Error message | `"Request timed out after 30s"` |

---

### Memory Events (on gen_ai.memory.* spans)

#### 13. `memory.stored`

**Description**: Memory successfully stored.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `memory_ids` | string[] | IDs of stored items | `["mem_123", "mem_456"]` |

---

#### 14. `memory.retrieved`

**Description**: Memory successfully retrieved.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `memory_ids` | string[] | IDs of retrieved items | `["mem_789"]` |
| `relevance_scores` | float[] | Relevance scores (if applicable) | `[0.95, 0.82]` |

---

### Retrieval Events (on retrieval/RAG spans)

#### 15. `retrieval.document`

**Description**: One retrieved document (emit per result).

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `doc_id` | string | Document identifier | `"doc_abc123"` |
| `score` | float | Relevance/similarity score | `0.92` |
| `chunk_preview` | string | Content preview (truncated) | `"This document discusses..."` |
| `metadata_json` | string (JSON) | Document metadata (optional) | `"{\"source\": \"web\", \"date\": \"2025-01-20\"}"` |

---

### Workflow Events (on gen_ai.workflow.* spans)

#### 16. `workflow.step_started`

**Description**: Workflow step begins.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `step_name` | string | Step/node name | `"retrieve_documents"` |
| `step_index` | int | Step sequence number | `0`, `1`, `2` |

---

#### 17. `workflow.step_completed`

**Description**: Workflow step completes.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `step_name` | string | Step/node name | `"retrieve_documents"` |
| `step_index` | int | Step sequence number | `1` |
| `finish_reason` | string | Completion reason | `"success"`, `"skipped"`, `"conditional"` |

---

#### 18. `workflow.routed`

**Description**: Routing decision made.

**Attributes**:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `from_node` | string | Source node | `"route_question"` |
| `to_node` | string | Destination node | `"generate_answer"` |
| `routing_reason` | string | Why this route | `"Question is answerable"` |

---

## Metrics Definitions

Metrics enable quantitative monitoring and alerting on agent system performance.

### Counter Metrics

#### 1. `gen_ai.agent.invocations`

**Description**: Total number of agent invocations.

**Unit**: `{invocation}`

**Type**: Counter

**Dimensions**:
- `gen_ai.agent.name` - Agent name
- `gen_ai.agent.type` - Agent type
- `gen_ai.agent.framework` - Framework name
- `status` - `"success"`, `"failure"`

---

#### 2. `gen_ai.agent.handoffs`

**Description**: Total number of agent-to-agent handoffs.

**Unit**: `{handoff}`

**Type**: Counter

**Dimensions**:
- `gen_ai.handoff.source_agent` - Source agent
- `gen_ai.handoff.target_agent` - Target agent
- `gen_ai.handoff.type` - Handoff type

---

#### 3. `gen_ai.task.completions`

**Description**: Total completed tasks.

**Unit**: `{task}`

**Type**: Counter

**Dimensions**:
- `gen_ai.task.type` - Task type
- `gen_ai.task.status` - Completion status
- `gen_ai.agent.name` - Executing agent

---

#### 4. `gen_ai.tool.calls`

**Description**: Total tool invocations.

**Unit**: `{call}`

**Type**: Counter

**Dimensions**:
- `gen_ai.tool.name` - Tool name
- `gen_ai.tool.type` - Tool type
- `gen_ai.tool.category` - Tool category
- `status` - `"success"`, `"failure"`

---

#### 5. `gen_ai.memory.operations`

**Description**: Total memory operations.

**Unit**: `{operation}`

**Type**: Counter

**Dimensions**:
- `gen_ai.memory.operation` - Operation type
- `gen_ai.memory.type` - Memory type
- `gen_ai.memory.store` - Storage backend

---

#### 6. `gen_ai.guardrail.triggers`

**Description**: Total guardrail activations.

**Unit**: `{trigger}`

**Type**: Counter

**Dimensions**:
- `gen_ai.guardrail.name` - Guardrail name
- `gen_ai.guardrail.type` - Guardrail type
- `gen_ai.guardrail.action` - Action taken

---

#### 7. `gen_ai.eval.executions`

**Description**: Total evaluations executed.

**Unit**: `{evaluation}`

**Type**: Counter

**Dimensions**:
- `gen_ai.eval.criteria` - Evaluation criteria
- `gen_ai.eval.method` - Evaluation method
- `gen_ai.eval.passed` - Pass/fail status

---

#### 8. `gen_ai.llm.tokens.input`

**Description**: Total input/prompt tokens consumed.

**Unit**: `{token}`

**Type**: Counter

**Dimensions**:
- `gen_ai.request.model` - Model name
- `gen_ai.system` - Provider

---

#### 9. `gen_ai.llm.tokens.output`

**Description**: Total output/completion tokens generated.

**Unit**: `{token}`

**Type**: Counter

**Dimensions**:
- `gen_ai.response.model` - Model name
- `gen_ai.system` - Provider

---

#### 10. `gen_ai.llm.tokens.total`

**Description**: Total tokens (input + output).

**Unit**: `{token}`

**Type**: Counter

**Dimensions**:
- `gen_ai.request.model` - Model name
- `gen_ai.system` - Provider

---

#### 11. `gen_ai.errors`

**Description**: Total error occurrences.

**Unit**: `{error}`

**Type**: Counter

**Dimensions**:
- `error.type` - Error type
- `gen_ai.agent.name` - Agent name
- `gen_ai.operation.name` - Operation

---

#### 12. `gen_ai.cost.total`

**Description**: Total estimated cost (LLM + tools).

**Unit**: `USD`

**Type**: Counter

**Dimensions**:
- `gen_ai.agent.name` - Agent name
- `gen_ai.request.model` - Model name
- `cost_type` - `"llm"`, `"tool"`, `"total"`

---

### Histogram Metrics

#### 13. `gen_ai.agent.duration`

**Description**: Agent execution duration.

**Unit**: `ms` (milliseconds)

**Type**: Histogram

**Dimensions**:
- `gen_ai.agent.name` - Agent name
- `gen_ai.agent.type` - Agent type
- `gen_ai.agent.framework` - Framework

**Recommended Buckets**: `[10, 50, 100, 500, 1000, 5000, 10000, 30000]`

---

#### 14. `gen_ai.task.duration`

**Description**: Task execution duration.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.task.type` - Task type
- `gen_ai.agent.name` - Executing agent

**Recommended Buckets**: `[100, 500, 1000, 5000, 10000, 30000, 60000]`

---

#### 15. `gen_ai.tool.duration`

**Description**: Tool execution duration.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.tool.name` - Tool name
- `gen_ai.tool.type` - Tool type

**Recommended Buckets**: `[10, 50, 100, 500, 1000, 5000, 10000]`

---

#### 16. `gen_ai.memory.retrieval.duration`

**Description**: Memory retrieval latency.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.memory.type` - Memory type
- `gen_ai.memory.store` - Storage backend

**Recommended Buckets**: `[1, 5, 10, 25, 50, 100, 250, 500]`

---

#### 17. `gen_ai.llm.duration`

**Description**: LLM call latency.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.request.model` - Model name
- `gen_ai.system` - Provider

**Recommended Buckets**: `[50, 100, 250, 500, 1000, 2000, 5000, 10000]`

---

#### 18. `gen_ai.context.tokens`

**Description**: Context token usage distribution.

**Unit**: `{token}`

**Type**: Histogram

**Dimensions**:
- `gen_ai.agent.name` - Agent name
- `gen_ai.request.model` - Model

**Recommended Buckets**: `[10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000]`

---

#### 19. `gen_ai.human.response_time`

**Description**: Human review response time.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.human.intervention_type` - Intervention type

**Recommended Buckets**: `[1000, 5000, 10000, 30000, 60000, 300000]`

---

#### 20. `gen_ai.session.duration`

**Description**: Session lifetime.

**Unit**: `ms`

**Type**: Histogram

**Dimensions**:
- `gen_ai.session.type` - Session type
- `gen_ai.agent.framework` - Framework

**Recommended Buckets**: `[1000, 5000, 10000, 30000, 60000, 300000, 600000]`

---

#### 21. `gen_ai.workflow.steps`

**Description**: Number of steps per workflow execution.

**Unit**: `{step}`

**Type**: Histogram

**Dimensions**:
- `gen_ai.workflow.type` - Workflow type
- `gen_ai.agent.framework` - Framework

**Recommended Buckets**: `[1, 2, 3, 5, 10, 20, 50, 100]`

---

### Gauge Metrics

#### 22. `gen_ai.agents.active`

**Description**: Currently active agents.

**Unit**: `{agent}`

**Type**: Gauge

**Dimensions**:
- `gen_ai.agent.framework` - Framework
- `gen_ai.agent.type` - Agent type

---

#### 23. `gen_ai.tasks.pending`

**Description**: Pending tasks in queue.

**Unit**: `{task}`

**Type**: Gauge

**Dimensions**:
- `gen_ai.task.type` - Task type
- `gen_ai.team.name` - Team name

---

#### 24. `gen_ai.memory.items`

**Description**: Items stored in memory.

**Unit**: `{item}`

**Type**: Gauge

**Dimensions**:
- `gen_ai.memory.type` - Memory type
- `gen_ai.memory.store` - Storage backend

---

#### 25. `gen_ai.sessions.active`

**Description**: Active sessions.

**Unit**: `{session}`

**Type**: Gauge

**Dimensions**:
- `gen_ai.agent.framework` - Framework
- `gen_ai.environment` - Environment

---

#### 26. `gen_ai.workflow.depth`

**Description**: Current workflow execution depth (nested workflows).

**Unit**: `{level}`

**Type**: Gauge

**Dimensions**:
- `gen_ai.workflow.type` - Workflow type

---

#### 27. `gen_ai.context.window_usage`

**Description**: Context window utilization percentage.

**Unit**: `%` (0.0 to 1.0)

**Type**: Gauge

**Dimensions**:
- `gen_ai.agent.name` - Agent name
- `gen_ai.session.id` - Session ID

---

## Span Hierarchies

### Typical Span Structures

Below are common span hierarchies for different agent patterns:

#### 1. Simple Agent with Tools

```
gen_ai.session
└── gen_ai.agent.invoke
    ├── gen_ai.client.chat (LLM call #1)
    ├── gen_ai.tool.execute (Tool #1)
    ├── gen_ai.client.chat (LLM call #2)
    ├── gen_ai.tool.execute (Tool #2)
    └── gen_ai.client.chat (LLM call #3 - final answer)
```

**Frameworks**: All frameworks with ReAct-style agents

---

#### 2. Multi-Agent Team (Sequential)

```
gen_ai.session
└── gen_ai.team.execute
    ├── gen_ai.agent.invoke (Agent #1 - Researcher)
    │   ├── gen_ai.client.chat
    │   └── gen_ai.tool.execute (search)
    ├── gen_ai.agent.handoff (Researcher → Writer)
    ├── gen_ai.agent.invoke (Agent #2 - Writer)
    │   ├── gen_ai.client.chat
    │   └── gen_ai.tool.execute (write_file)
    ├── gen_ai.agent.handoff (Writer → Reviewer)
    └── gen_ai.agent.invoke (Agent #3 - Reviewer)
        └── gen_ai.client.chat
```

**Frameworks**: CrewAI (sequential process), Autogen (round-robin), OpenAI SDK (handoffs)

---

#### 3. Graph-Based Workflow with Checkpointing

```
gen_ai.session
└── gen_ai.workflow.execute (LangGraph)
    ├── gen_ai.context.checkpoint (Initial checkpoint)
    ├── gen_ai.workflow.transition (START → retrieve)
    ├── gen_ai.agent.invoke (Retriever node)
    │   ├── gen_ai.memory.search
    │   └── gen_ai.client.chat
    ├── gen_ai.context.checkpoint (After retrieve)
    ├── gen_ai.workflow.transition (retrieve → grade)
    ├── gen_ai.agent.invoke (Grader node)
    │   └── gen_ai.client.chat
    ├── gen_ai.context.checkpoint (After grade)
    ├── gen_ai.workflow.branch (Grade decision)
    ├── gen_ai.workflow.transition (grade → generate)
    ├── gen_ai.agent.invoke (Generator node)
    │   └── gen_ai.client.chat
    ├── gen_ai.context.checkpoint (After generate)
    └── gen_ai.workflow.transition (generate → END)
```

**Frameworks**: LangGraph, Mastra workflows

---

#### 4. Task-Based Execution (CrewAI)

```
gen_ai.session
└── gen_ai.team.execute (Crew)
    ├── gen_ai.task.create (Research task)
    ├── gen_ai.task.execute (Research task)
    │   └── gen_ai.agent.invoke (Researcher agent)
    │       ├── gen_ai.tool.execute (web_search)
    │       └── gen_ai.client.chat
    ├── gen_ai.task.create (Writing task)
    └── gen_ai.task.execute (Writing task)
        └── gen_ai.agent.invoke (Writer agent)
            ├── gen_ai.memory.retrieve (get research results)
            └── gen_ai.client.chat
```

**Frameworks**: CrewAI (task-centric), Google ADK (tasks)

---

#### 5. Hierarchical Multi-Agent (Supervisor Pattern)

```
gen_ai.session
└── gen_ai.team.execute
    ├── gen_ai.agent.invoke (Manager/Supervisor)
    │   └── gen_ai.client.chat (decides delegation)
    ├── gen_ai.team.coordinate (Select specialist #1)
    ├── gen_ai.agent.handoff (Manager → Specialist #1)
    ├── gen_ai.agent.invoke (Specialist #1)
    │   ├── gen_ai.client.chat
    │   └── gen_ai.tool.execute
    ├── gen_ai.team.coordinate (Select specialist #2)
    ├── gen_ai.agent.handoff (Manager → Specialist #2)
    ├── gen_ai.agent.invoke (Specialist #2)
    │   ├── gen_ai.client.chat
    │   └── gen_ai.tool.execute
    ├── gen_ai.agent.handoff (Specialist #2 → Manager)
    └── gen_ai.agent.invoke (Manager - final synthesis)
        └── gen_ai.client.chat
```

**Frameworks**: CrewAI (hierarchical process), Autogen (Selector), Google ADK (coordinator agents)

---

#### 6. Agent with Memory and Guardrails

```
gen_ai.session
└── gen_ai.agent.invoke
    ├── gen_ai.memory.retrieve (Load context)
    ├── gen_ai.guardrail.check (Input validation)
    ├── gen_ai.client.chat (LLM call)
    ├── gen_ai.guardrail.check (Output validation)
    ├── gen_ai.tool.execute (If needed)
    ├── gen_ai.memory.store (Save interaction)
    └── gen_ai.eval.execute (Quality check)
```

**Frameworks**: Agno, CrewAI, AgentCore

---

#### 7. Human-in-the-Loop Workflow

```
gen_ai.session
└── gen_ai.workflow.execute
    ├── gen_ai.agent.invoke (Planning agent)
    │   └── gen_ai.client.chat
    ├── gen_ai.human.review (Human approval)
    ├── gen_ai.agent.invoke (Execution agent)
    │   ├── gen_ai.tool.execute (high_risk_operation)
    │   └── gen_ai.human.review (Confirm tool use)
    └── gen_ai.agent.invoke (Completion agent)
        └── gen_ai.client.chat
```

**Frameworks**: LangGraph (breakpoints), Google ADK (tool confirmation), CrewAI (HITL)

---

### Span vs. Event Decision Guidelines

**Use Spans When**:
- The operation has measurable duration
- The operation may have child operations
- You need to track errors and status separately
- The operation is a significant architectural boundary

**Use Events When**:
- Capturing a point-in-time occurrence
- Logging data associated with a span (prompts, completions, observations)
- Recording state snapshots
- Annotating spans with additional context

**Examples**:
- ✅ Span: `gen_ai.tool.execute` (has duration, can have errors)
- ✅ Event: `tool.request`, `tool.response` (point-in-time data within tool execution)
- ✅ Span: `gen_ai.memory.search` (searchable operation with duration)
- ✅ Event: `retrieval.document` (each retrieved document as annotation)

---

## Framework Mappings

This section shows how each of the 11 major agent frameworks maps to the semantic conventions defined in this specification.

### Framework Coverage Matrix

| Feature | Agno | Autogen | CrewAI | ADK | LangGraph | LlamaIndex | OpenAI SDK | Mastra | Smolagents | Haystack | AWS AgentCore |
|---------|------|---------|--------|-----|-----------|------------|-----------|--------|------------|----------|---------------|
| **Agent Lifecycle** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-Agent Teams** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| **Task Management** | ⚠️ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| **Agent Handoffs** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| **Memory Systems** | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | ✅ |
| **Tool Execution** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MCP Support** | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | ❌ | ⚠️ |
| **Workflow/Graph** | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ | ✅ | ⚠️ |
| **Checkpointing** | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ | ✅ |
| **Guardrails** | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| **Runtime Evals** | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| **Human-in-the-Loop** | ✅ | ⚠️ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | ⚠️ | ⚠️ |
| **Session Management** | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ |

**Legend**:
- ✅ **Native Support** - Framework has first-class support
- ⚠️ **Partial/Indirect** - Supported through extensions, manual implementation, or related features
- ❌ **Not Supported** - No direct support

---

### 1. Agno Framework Mapping

**Framework Concepts → Semantic Conventions**

| Agno Concept | Maps To | Notes |
|--------------|---------|-------|
| `Agent` | `gen_ai.agent.invoke` | Agent execution |
| `Team` | `gen_ai.team.execute` | Multi-agent teams with shared state |
| `Workflow` | `gen_ai.workflow.execute` | Step-based workflows |
| `Tool` / `Toolkit` | `gen_ai.tool.execute` | 100+ toolkits |
| `MCPTools` | `gen_ai.mcp.execute` | First-class MCP support |
| `Memory` (Semantic/Episodic/Procedural) | `gen_ai.memory.*` | Layered memory |
| `Knowledge` | `gen_ai.memory.search` | RAG via vector stores |
| `Culture` | `gen_ai.memory.*` with team scope | Team-shared memory |
| `SessionService` | `gen_ai.session.*` | Session persistence |
| `Guardrails` | `gen_ai.guardrail.check` | PII detection, content moderation |

**Attributes**:
- `gen_ai.agent.mode` = `"react"` (if using ReAct-style)
- `gen_ai.agent.framework` = `"agno"`
- `gen_ai.memory.type` = `"semantic"` / `"episodic"` / `"procedural"`

---

### 2. Autogen Framework Mapping

**Framework Concepts → Semantic Conventions**

| Autogen Concept | Maps To | Notes |
|-----------------|---------|-------|
| `AssistantAgent` | `gen_ai.agent.invoke` | LLM-powered agent |
| `UserProxyAgent` | `gen_ai.agent.invoke` | Human proxy agent |
| `GroupChat` | `gen_ai.team.execute` | Multi-agent conversation |
| `GroupChatManager` | `gen_ai.team.coordinate` | Turn selection/coordination |
| `RoundRobin` | `gen_ai.team.orchestration_pattern` = `"round_robin"` | Sequential turns |
| `Selector` | `gen_ai.team.orchestration_pattern` = `"selector"` | LLM-based selection |
| Tool execution | `gen_ai.tool.execute` | Function calling tools |
| `McpWorkbench` | `gen_ai.mcp.*` | MCP integration |
| Code execution | `gen_ai.tool.execute` with type=`"code"` | Docker sandboxing |
| Session state | `gen_ai.session.*` | Checkpoint-based persistence |

**Attributes**:
- `gen_ai.agent.framework` = `"autogen"`
- `gen_ai.team.orchestration_pattern` = `"round_robin"` or `"selector"`
- `gen_ai.conversation.round` = Round number in GroupChat

---

### 3. CrewAI Framework Mapping

**Framework Concepts → Semantic Conventions**

| CrewAI Concept | Maps To | Notes |
|----------------|---------|-------|
| `Crew` | `gen_ai.team.execute` | Team of agents |
| `Agent` (with role, goal, backstory) | `gen_ai.agent.create` / `gen_ai.agent.invoke` | Role-based agents |
| `Task` | `gen_ai.task.*` | Work units |
| `Process.SEQUENTIAL` | `gen_ai.team.orchestration_pattern` = `"sequential"` | Sequential execution |
| `Process.HIERARCHICAL` | `gen_ai.team.orchestration_pattern` = `"hierarchical"` | Manager-led |
| `Flow` | `gen_ai.workflow.execute` | Event-driven workflows |
| `@start`, `@listen`, `@router` | `gen_ai.workflow.transition` / `gen_ai.workflow.branch` | Flow primitives |
| Delegation | `gen_ai.agent.handoff` | Agent-to-agent delegation |
| Memory (Short/Long/Entity/User) | `gen_ai.memory.*` | Layered memory in ChromaDB |
| Knowledge Base | `gen_ai.memory.search` | RAG for agents/tasks |
| Guardrails | `gen_ai.guardrail.check` | Input/output validation |
| HITL | `gen_ai.human.review` | Human-in-the-loop |

**Attributes**:
- `gen_ai.agent.framework` = `"crewai"`
- `gen_ai.agent.role`, `gen_ai.agent.goal`, `gen_ai.agent.backstory` (CrewAI-specific attributes)
- `gen_ai.task.expected_output`, `gen_ai.task.output_file`

---

### 4. Google ADK Framework Mapping

**Framework Concepts → Semantic Conventions**

| ADK Concept | Maps To | Notes |
|-------------|---------|-------|
| `LlmAgent` | `gen_ai.agent.invoke` | LLM-driven agent |
| `SequentialAgent` | `gen_ai.workflow.execute` with type=`"sequential"` | Sequential workflow |
| `ParallelAgent` | `gen_ai.workflow.execute` with type=`"parallel"` | Parallel execution |
| `LoopAgent` | `gen_ai.workflow.execute` with type=`"loop"` | Iterative workflow |
| Sub-agents | `gen_ai.agent.handoff` or child spans | Hierarchical agents |
| `SessionService` | `gen_ai.session.*` | Session persistence |
| `MemoryService` | `gen_ai.memory.*` | Long-term memory |
| `event_history` | `gen_ai.session.message_count` | Conversation history |
| Tool Confirmation | `gen_ai.human.review` with type=`"approval"` | HITL for tools |
| Session Rewind | Custom attribute on `gen_ai.context.checkpoint` | Rollback capability |
| `AgentTool` | `gen_ai.agent.handoff` (agent-as-tool) | Sub-agent invocation |
| A2A Protocol | `gen_ai.agent.handoff` | Remote agent communication |

**Attributes**:
- `gen_ai.agent.framework` = `"google-adk"`
- `gen_ai.workflow.type` = `"sequential"` / `"parallel"` / `"loop"`
- `gen_ai.context.checkpoint_backend` = `"vertex_ai"` / `"sqlite"` / `"postgres"`

---

### 5. LangGraph Framework Mapping

**Framework Concepts → Semantic Conventions**

| LangGraph Concept | Maps To | Notes |
|-------------------|---------|-------|
| `StateGraph` | `gen_ai.workflow.execute` | Graph-based workflow |
| Node | `gen_ai.agent.invoke` or task/step span | Graph node execution |
| Edge | `gen_ai.workflow.transition` | State transition |
| Conditional Edge | `gen_ai.workflow.branch` | Conditional routing |
| State | `gen_ai.state.*` attributes | Shared state object |
| Super-Step | Single iteration through graph | Full graph cycle |
| Checkpointer | `gen_ai.context.checkpoint` | State persistence |
| Thread ID | `gen_ai.session.thread_id` | Conversation thread |
| Checkpoint ID | `gen_ai.context.checkpoint_id` | State snapshot |
| Breakpoint | `gen_ai.human.review` or pause mechanism | Interrupt execution |
| `create_react_agent` | `gen_ai.agent.mode` = `"react"` | Prebuilt ReAct helper |

**Attributes**:
- `gen_ai.agent.framework` = `"langgraph"`
- `gen_ai.workflow.execution_path` = Array of nodes traversed
- `gen_ai.state.keys_changed` = State keys modified
- `gen_ai.context.checkpoint_backend` = `"sqlite"` / `"postgres"` / `"memory"`

---

### 6. LlamaIndex Framework Mapping

**Framework Concepts → Semantic Conventions**

| LlamaIndex Concept | Maps To | Notes |
|---------------------|---------|-------|
| `ReActAgent` | `gen_ai.agent.invoke` with mode=`"react"` | ReAct pattern agent |
| `FunctionAgent` | `gen_ai.agent.invoke` with mode=`"function_calling"` | Function calling agent |
| `AgentWorkflow` | `gen_ai.workflow.execute` | Multi-agent orchestration |
| Query Engine | `gen_ai.memory.search` or custom span | RAG pipeline |
| Tool | `gen_ai.tool.execute` | Function/query tools |
| Index (VectorStoreIndex, etc.) | Relates to `gen_ai.memory.search` | Data indexing |
| Retrieval | `gen_ai.memory.search` with retrieval events | Document retrieval |
| Chat Engine | `gen_ai.session.*` | Stateful conversation |
| Instrumentation module | Direct mapping to OTel spans | Native observability |

**Attributes**:
- `gen_ai.agent.framework` = `"llamaindex"`
- `gen_ai.agent.mode` = `"react"` / `"function_calling"`
- `gen_ai.memory.store` = Vector database type (e.g., `"pinecone"`, `"weaviate"`)

---

### 7. OpenAI Agents SDK Framework Mapping

**Framework Concepts → Semantic Conventions**

| OpenAI SDK Concept | Maps To | Notes |
|---------------------|---------|-------|
| `Agent` | `gen_ai.agent.create` / `gen_ai.agent.invoke` | Agent definition |
| `Runner.run()` | `gen_ai.agent.invoke` | Agent execution |
| Handoff | `gen_ai.agent.handoff` | Core A2A primitive |
| `on_handoff` callback | Event within handoff span | Setup logic on handoff |
| `input_type` | Attribute on handoff span | Expected input schema |
| `output_type` | Attribute on agent span | Structured output schema |
| `@function_tool` | `gen_ai.tool.execute` | Tool decorator |
| Session (SQLiteSession) | `gen_ai.session.*` | Session persistence |
| Max Turns | `gen_ai.agent.max_iterations` | Loop limit |
| Instructions | `gen_ai.agent.goal` or custom attribute | System prompt |

**Attributes**:
- `gen_ai.agent.framework` = `"openai-agents"`
- `gen_ai.handoff.*` attributes (handoff is core pattern)

---

### 8. MastraAI Framework Mapping

**Framework Concepts → Semantic Conventions**

| Mastra Concept | Maps To | Notes |
|----------------|---------|-------|
| `Agent` | `gen_ai.agent.invoke` | Autonomous entity |
| `Workflow` | `gen_ai.workflow.execute` | Graph-based orchestration |
| `.then()` | `gen_ai.workflow.transition` | Sequential step |
| `.branch()` | `gen_ai.workflow.branch` | Conditional routing |
| `.parallel()` | `gen_ai.workflow.execute` with type=`"parallel"` | Concurrent execution |
| `Writer` (streaming) | Events within workflow | Streaming interface |
| Tool | `gen_ai.tool.execute` | Tool execution |
| Memory (conversation, semantic, working) | `gen_ai.memory.*` | Multi-layered memory |
| RAG | `gen_ai.memory.search` | Retrieval integration |
| Nested streaming | Events propagate from nested agents | v0.11.1+ feature |

**Attributes**:
- `gen_ai.agent.framework` = `"mastra"`
- `gen_ai.workflow.type` = `"sequential"` / `"parallel"` / `"conditional"`

---

### 9. Smolagents Framework Mapping

**Framework Concepts → Semantic Conventions**

| Smolagents Concept | Maps To | Notes |
|---------------------|---------|-------|
| `CodeAgent` | `gen_ai.agent.invoke` with mode=`"code_interpreter"` | Code-first actions |
| `ToolCallingAgent` | `gen_ai.agent.invoke` with mode=`"function_calling"` | JSON-based tools |
| `MultiStepAgent` | Base for agent execution | ReAct abstraction |
| ReAct Loop | Agent invocation with iterations | Thought → Action → Observation |
| Code Snippet (action) | `gen_ai.tool.execute` with type=`"code"` | Python code execution |
| Tool | `gen_ai.tool.execute` | Tool invocation |
| Execution Backend (E2B, Docker, etc.) | Attribute on tool span | Sandbox environment |
| `final_answer()` | Marks agent termination | Loop exit |

**Attributes**:
- `gen_ai.agent.framework` = `"smolagents"`
- `gen_ai.agent.mode` = `"code_interpreter"` (for CodeAgent)
- `gen_ai.tool.type` = `"code"` (for code execution)
- `gen_ai.tool.execution_backend` = Sandbox type

---

### 10. Haystack Framework Mapping

**Framework Concepts → Semantic Conventions**

| Haystack Concept | Maps To | Notes |
|------------------|---------|-------|
| Pipeline | `gen_ai.workflow.execute` | Component graph |
| Agent Component | `gen_ai.agent.invoke` | Universal agent component |
| `ChatGenerator` | `gen_ai.client.chat` | LLM component |
| `ToolInvoker` | `gen_ai.tool.execute` | Tool executor |
| `ComponentTool` | `gen_ai.tool.execute` | Component-as-tool |
| `ConditionalRouter` | `gen_ai.workflow.branch` | Routing logic |
| `MessageCollector` | Context management (not a span) | Message storage |
| Document Store | `gen_ai.memory.store` | Vector/document storage |
| Retriever | `gen_ai.memory.search` | Document retrieval |

**Attributes**:
- `gen_ai.agent.framework` = `"haystack"`
- `gen_ai.workflow.type` = `"pipeline"` (graph of components)

---

### 11. AWS Bedrock AgentCore Framework Mapping

**Framework Concepts → Semantic Conventions**

| AgentCore Concept | Maps To | Notes |
|-------------------|---------|-------|
| Entrypoint | `gen_ai.agent.invoke` | Main agent function |
| Runtime | `gen_ai.session.*` | Managed execution environment |
| Memory Service | `gen_ai.memory.*` | Short/long-term memory |
| Gateway Service | `gen_ai.tool.execute` with type=`"gateway"` | API-to-MCP transformation |
| Code Interpreter | `gen_ai.tool.execute` with type=`"code"` | Sandboxed Python execution |
| Browser Service | `gen_ai.tool.execute` with type=`"browser"` | Headless browser |
| Observability Service | Native OTEL integration | Built-in tracing |
| Identity Service | Authentication/authorization (not directly a span) | IAM integration |
| Session | `gen_ai.session.*` | Session lifecycle |

**Attributes**:
- `gen_ai.agent.framework` = `"aws-agentcore"` or specific framework detected
- Framework detection: AgentCore auto-detects underlying framework (LangGraph, CrewAI, etc.)

---

## Use Case Scenarios

This section demonstrates how the semantic conventions enable specific monitoring, debugging, and optimization scenarios.

### 1. Monitoring Agent Success Rates

**Objective**: Track agent invocation success/failure rates across all frameworks.

**Metrics Used**:
- `gen_ai.agent.invocations` (counter) with dimensions `status="success"` / `status="failure"`

**Query Example** (PromQL-style):
```promql
rate(gen_ai_agent_invocations_total{status="failure"}[5m])
  / rate(gen_ai_agent_invocations_total[5m])
```

**Span Analysis**:
- Filter `gen_ai.agent.invoke` spans
- Group by `gen_ai.agent.name` and `error.type`
- Analyze failure patterns

**Alert Rule**:
```yaml
alert: HighAgentFailureRate
expr: rate(gen_ai_agent_invocations_total{status="failure"}[5m]) > 0.1
annotations:
  summary: "Agent {{ $labels.gen_ai_agent_name }} has >10% failure rate"
```

---

### 2. Token Usage and Cost Tracking

**Objective**: Monitor token consumption and estimated costs per agent/team.

**Metrics Used**:
- `gen_ai.llm.tokens.input` (counter)
- `gen_ai.llm.tokens.output` (counter)
- `gen_ai.cost.total` (counter)

**Dimensions**:
- `gen_ai.agent.name`
- `gen_ai.request.model`
- `gen_ai.system` (provider)

**Query Example**:
```promql
# Total tokens per agent in last hour
sum by (gen_ai_agent_name) (
  increase(gen_ai_llm_tokens_total[1h])
)

# Cost per model
sum by (gen_ai_request_model) (
  increase(gen_ai_cost_total[1h])
)
```

**Dashboard Visualization**:
- Time series: Token usage over time by agent
- Pie chart: Cost distribution by model
- Table: Top 10 most expensive agents

---

### 3. Debugging Multi-Agent Conversations

**Objective**: Trace handoffs between agents to diagnose coordination issues.

**Spans Used**:
- `gen_ai.session` (top-level)
- `gen_ai.team.execute`
- `gen_ai.agent.handoff`
- `gen_ai.agent.invoke`

**Trace Structure**:
```
Session span (trace root)
└── Team execution
    ├── Agent A invocation
    ├── Handoff (A → B) with reason
    ├── Agent B invocation
    ├── Handoff (B → C) with reason
    └── Agent C invocation
```

**Attributes to Inspect**:
- `gen_ai.handoff.source_agent`, `gen_ai.handoff.target_agent`
- `gen_ai.handoff.reason` (why handoff occurred)
- `gen_ai.handoff.arguments_json` (context transferred)
- `gen_ai.handoff.response_summary` (result from target agent)

**Debugging Questions Answered**:
- Which agent is causing delays? → Check span durations
- Why was agent X never called? → Check handoff routing decisions
- What context was lost in handoff? → Inspect `arguments_json` and `context_transferred`

---

### 4. Visualizing Workflow Execution Paths

**Objective**: Understand which nodes/branches were executed in graph-based workflows (LangGraph, Mastra).

**Spans Used**:
- `gen_ai.workflow.execute`
- `gen_ai.workflow.transition`
- `gen_ai.workflow.branch`

**Attributes**:
- `gen_ai.workflow.execution_path` (array of nodes executed)
- `gen_ai.workflow.branch_taken` (which branch chosen)
- `gen_ai.state.transition_from`, `gen_ai.state.transition_to`

**Visualization**:
```
[retrieve_docs] → [grade_docs] → [BRANCH: is_relevant?]
                                   ├─ YES → [generate_answer]
                                   └─ NO → [web_search] → [generate_answer]
```

**Analysis**:
- Count how often each branch is taken
- Identify unused nodes (potential dead code)
- Measure latency per node
- Find bottleneck nodes

---

### 5. Identifying Tool Execution Bottlenecks

**Objective**: Find which tools are slowest and most error-prone.

**Metrics Used**:
- `gen_ai.tool.duration` (histogram)
- `gen_ai.tool.calls` (counter) with `status="failure"`

**Query Example**:
```promql
# P95 latency per tool
histogram_quantile(0.95,
  sum by (gen_ai_tool_name, le) (
    rate(gen_ai_tool_duration_bucket[5m])
  )
)

# Tool error rate
rate(gen_ai_tool_calls_total{status="failure"}[5m])
  / rate(gen_ai_tool_calls_total[5m])
```

**Span Analysis**:
- Filter `gen_ai.tool.execute` spans with `error.type` set
- Group by `gen_ai.tool.name` and `gen_ai.tool.error_strategy`
- Check `gen_ai.tool.retry_count` distribution

**Optimization Actions**:
- Add caching for slow tools
- Implement fallback strategies for error-prone tools
- Increase timeouts for legitimate long-running tools

---

### 6. Memory Retrieval Performance

**Objective**: Optimize memory search performance.

**Metrics Used**:
- `gen_ai.memory.retrieval.duration` (histogram)
- `gen_ai.memory.operations` (counter)

**Dimensions**:
- `gen_ai.memory.type` (short_term, long_term, semantic)
- `gen_ai.memory.store` (chromadb, pinecone, etc.)

**Span Attributes**:
- `gen_ai.memory.search.top_k` (how many results requested)
- `gen_ai.memory.items_retrieved` (how many returned)
- `gen_ai.memory.relevance_score` (threshold used)
- `gen_ai.memory.hit` (cache hit/miss)

**Analysis**:
- Compare latency across different vector stores
- Measure cache hit rates
- Correlate `top_k` with latency
- Identify slow queries (large `top_k`, complex filters)

**Optimization**:
- Tune `top_k` for optimal recall/latency tradeoff
- Add caching layer for frequent queries
- Optimize vector store indexing

---

### 7. Context Window Utilization

**Objective**: Prevent context overflow and optimize token usage.

**Metrics Used**:
- `gen_ai.context.window_usage` (gauge, percentage)
- `gen_ai.context.tokens` (histogram)

**Attributes**:
- `gen_ai.context.window_size` (max tokens)
- `gen_ai.context.tokens_used` (current usage)
- `gen_ai.context.compression_ratio` (if compressed)

**Alert Rule**:
```yaml
alert: ContextWindowNearLimit
expr: gen_ai_context_window_usage > 0.9
annotations:
  summary: "Session {{ $labels.gen_ai_session_id }} context at {{ $value }}%"
```

**Analysis**:
- Track context growth over conversation turns
- Identify sessions hitting limits
- Measure effectiveness of compression

**Optimization**:
- Implement context summarization when usage > 80%
- Use rolling window for long conversations
- Offload old context to long-term memory

---

### 8. Guardrail Effectiveness

**Objective**: Measure how often guardrails trigger and their impact.

**Metrics Used**:
- `gen_ai.guardrail.triggers` (counter)

**Dimensions**:
- `gen_ai.guardrail.name`
- `gen_ai.guardrail.type` (input_validation, output_validation, etc.)
- `gen_ai.guardrail.action` (block, warn, modify)

**Span Attributes**:
- `gen_ai.guardrail.triggered` (boolean)
- `gen_ai.guardrail.confidence` (detection confidence)
- `gen_ai.guardrail.violation_type`

**Query Example**:
```promql
# Guardrail trigger rate
sum by (gen_ai_guardrail_name) (
  rate(gen_ai_guardrail_triggers_total[1h])
)

# False positive rate (requires manual labeling)
sum by (gen_ai_guardrail_name) (
  gen_ai_guardrail_triggers_total{false_positive="true"}
) / sum by (gen_ai_guardrail_name) (
  gen_ai_guardrail_triggers_total
)
```

**Analysis**:
- Most frequently triggered guardrails
- Action distribution (how many blocked vs. warned)
- Correlation with agent errors

---

### 9. Human Review Response Times

**Objective**: Track how long humans take to approve agent actions.

**Metrics Used**:
- `gen_ai.human.response_time` (histogram)

**Dimensions**:
- `gen_ai.human.intervention_type` (approval, feedback, correction)

**Span Attributes**:
- `gen_ai.human.approval_required`
- `gen_ai.human.approval_granted`
- `gen_ai.human.response_time_ms`

**Analysis**:
- P50, P95, P99 response times
- Approval vs. rejection rates
- Impact on overall agent execution time

**Optimization**:
- Reduce unnecessary approvals
- Batch approval requests
- Provide better context for faster decisions

---

### 10. Session Analysis and Conversation Quality

**Objective**: Understand session characteristics and quality.

**Metrics Used**:
- `gen_ai.session.duration` (histogram)
- `gen_ai.sessions.active` (gauge)

**Attributes**:
- `gen_ai.session.message_count`
- `gen_ai.session.turn_count`
- `gen_ai.eval.score` (if evaluations run)
- `gen_ai.eval.passed`

**Queries**:
```promql
# Average session length
avg(gen_ai_session_duration_ms)

# Successful sessions (passed evaluations)
sum(gen_ai_eval_executions_total{gen_ai_eval_passed="true"})
  / sum(gen_ai_eval_executions_total)
```

**Analysis**:
- Session length distribution
- Messages per session
- Evaluation scores distribution
- Correlation between session length and quality

---

## Integration Guide

### Integration with Existing OpenTelemetry GenAI Conventions

These agent-specific conventions **extend** the existing [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) rather than replace them.

**Reused Conventions**:

1. **`gen_ai.client.*` spans** for LLM operations:
   - `gen_ai.system` (provider)
   - `gen_ai.request.model`, `gen_ai.response.model`
   - `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens`
   - `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`
   - `gen_ai.response.finish_reason`

2. **Standard OpenTelemetry attributes**:
   - `server.address`, `server.port` (for MCP connections)
   - `error.type`, `exception.type`, `exception.message` (error handling)

**New Agent-Specific Namespaces**:
- `gen_ai.agent.*`, `gen_ai.team.*`, `gen_ai.task.*`, `gen_ai.handoff.*`
- `gen_ai.memory.*`, `gen_ai.session.*`, `gen_ai.context.*`
- `gen_ai.workflow.*`, `gen_ai.state.*`, `gen_ai.guardrail.*`, `gen_ai.eval.*`, `gen_ai.human.*`

**Extension Pattern**:
```
gen_ai.*                    ← Existing namespace
├── client.*                ← Existing (LLM operations)
│   └── Extensions:
│       ├── agent.id        ← NEW: Which agent made this call
│       └── task.id         ← NEW: Task context
├── agent.*                 ← NEW: Agent-specific
├── team.*                  ← NEW: Multi-agent
├── task.*                  ← NEW: Task management
├── tool.*                  ← EXTENDED: Tool execution
│   ├── name, type          ← Existing
│   ├── selection_method    ← NEW
│   └── error_strategy      ← NEW
└── [other new namespaces]
```

---

### Integration with Traceloop OpenLLMetry

**Current Traceloop Conventions** (from `opentelemetry-semantic-conventions-ai` package):

Traceloop's existing conventions include:
- `gen_ai.request.*`, `gen_ai.response.*`, `gen_ai.usage.*` (aligned with OTel)
- Framework-specific instrumentations (LangChain, LlamaIndex, OpenAI, etc.)
- Custom events for prompts and completions

**Migration Path**:

1. **Preserve Existing Attributes**: All current Traceloop attributes remain valid
2. **Add New Agent Attributes**: Introduce `gen_ai.agent.*`, `gen_ai.team.*`, etc. as extensions
3. **Enhance Existing Instrumentations**: Update framework instrumentations to emit new agent spans

**Example Enhancement** (LangChain instrumentation):

```python
# Current: Only gen_ai.client.chat spans for LLM calls
# Enhanced: Add gen_ai.agent.invoke span wrapping execution

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def instrument_langchain_agent(agent):
    with tracer.start_as_current_span(
        "gen_ai.agent.invoke",
        attributes={
            "gen_ai.agent.name": agent.name,
            "gen_ai.agent.framework": "langchain",
            "gen_ai.agent.type": "react",
        }
    ) as agent_span:
        # Existing LLM and tool spans become children
        result = agent.run(input)

        agent_span.set_attribute("gen_ai.runtime.llm_calls_count", ...)
        agent_span.set_attribute("gen_ai.runtime.tool_calls_count", ...)
        return result
```

**Backwards Compatibility**:
- Old instrumentations continue to work
- New attributes are opt-in via instrumentation updates
- No breaking changes to existing telemetry

---

### Implementation Guidance

#### 1. Instrumentation Placement

**Where to Create Spans**:

| Span Type | When to Create | Instrumentation Point |
|-----------|----------------|----------------------|
| `gen_ai.session` | Session start | Framework session initialization |
| `gen_ai.agent.create` | Agent instantiation | Agent constructor or factory |
| `gen_ai.agent.invoke` | Agent execution | Agent `run()`, `invoke()`, `execute()` methods |
| `gen_ai.tool.execute` | Tool call | Tool invocation wrapper |
| `gen_ai.memory.*` | Memory operation | Memory service methods |
| `gen_ai.workflow.execute` | Workflow start | Workflow/graph `execute()` method |
| `gen_ai.workflow.transition` | Node transition | Between workflow steps |

**Span Hierarchy Guidelines**:
- Use `gen_ai.session` as trace root (one per conversation/run)
- Nest `gen_ai.agent.invoke` under session or workflow
- Make `gen_ai.client.chat` (LLM calls) children of agent invoke
- Make `gen_ai.tool.execute` children of agent invoke

---

#### 2. Attribute Cardinality Considerations

**High Cardinality Attributes** (use with caution):
- `gen_ai.session.id`, `gen_ai.agent.id`, `gen_ai.task.id` (unique IDs)
- `gen_ai.handoff.arguments_json`, `gen_ai.tool.parameters` (variable content)
- `gen_ai.context.checkpoint_id`

**Mitigation**:
- Use high-cardinality attributes as **span attributes** (within spans), not as **metric dimensions**
- Apply sampling for high-volume systems
- Truncate/redact long text fields

**Low Cardinality Attributes** (safe for metric dimensions):
- `gen_ai.agent.name`, `gen_ai.agent.type`, `gen_ai.agent.framework`
- `gen_ai.tool.name`, `gen_ai.tool.type`
- `gen_ai.memory.type`, `gen_ai.memory.store`
- `status` (`"success"`, `"failure"`)

---

#### 3. Privacy and PII Handling

**Sensitive Data Locations**:
- `gen_ai.tool.parameters`, `gen_ai.tool.result` (may contain user input)
- `gen_ai.handoff.arguments_json` (agent communication)
- Event data: `llm.prompt`, `llm.completion`, `agent.thought`
- `gen_ai.session.user_id` (should be hashed)

**Best Practices**:
1. **Redact PII**: Use processors/filters to redact emails, phone numbers, SSNs, etc.
2. **Hash User IDs**: Never store raw user IDs; use hashes
3. **Truncate Prompts/Completions**: Limit length in events (e.g., first 500 chars)
4. **Mark Sensitive Attributes**: Add `redacted` flags to events
5. **Separate Telemetry Tiers**:
   - **Production**: Minimal sensitive data, aggressive redaction
   - **Staging**: More detail for debugging
   - **Development**: Full data

**Example Processor**:
```python
from opentelemetry.sdk.trace.export import SpanProcessor
import re

class PIIRedactionProcessor(SpanProcessor):
    def on_start(self, span, parent_context):
        pass

    def on_end(self, span):
        # Redact email addresses from attributes
        for key, value in span.attributes.items():
            if isinstance(value, str):
                span.set_attribute(key, re.sub(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    '[EMAIL_REDACTED]',
                    value
                ))
```

---

#### 4. Sampling Strategies

For high-volume agent systems, sampling is essential:

**Sampling Approaches**:

1. **Head-Based Sampling** (at trace root):
   - Sample X% of sessions
   - Use `TraceIdRatioBasedSampler`

2. **Tail-Based Sampling** (after trace completion):
   - Keep all error traces
   - Sample successful traces
   - Requires trace aggregation

3. **Attribute-Based Sampling**:
   - Always sample traces with errors
   - Always sample traces with `gen_ai.eval.passed=false`
   - Sample long-running sessions more aggressively

**Example Configuration**:
```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import (
    ParentBasedTraceIdRatioBased,
    ALWAYS_ON,
    ALWAYS_OFF
)

# Sample 10% of traces, but always sample errors
sampler = ParentBasedTraceIdRatioBased(0.1)

tracer_provider = TracerProvider(sampler=sampler)
```

---

#### 5. Performance Impact

**Overhead Sources**:
- Span creation and attribute setting
- Event emission
- Metric recording
- Export to backend

**Mitigation**:
1. **Async Export**: Use batch span processors, never block agent execution
2. **Limit Event Size**: Truncate long prompts/completions
3. **Metric Aggregation**: Pre-aggregate metrics before export
4. **Sampling**: As described above
5. **Conditional Detail**: Emit verbose data only in development/staging

**Benchmark Guidance**:
- Instrumentation should add <5% latency overhead
- Memory overhead <10MB per 1000 spans

---

### Example Instrumentation (Pseudocode)

**Simple Agent with Tools**:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def run_agent(agent, user_input, session_id):
    # Create session span (trace root)
    with tracer.start_as_current_span(
        "gen_ai.session",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "gen_ai.session.id": session_id,
            "gen_ai.session.type": "chat",
            "gen_ai.agent.framework": "custom",
        }
    ) as session_span:

        # Create agent invocation span
        with tracer.start_as_current_span(
            "gen_ai.agent.invoke",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "gen_ai.agent.id": agent.id,
                "gen_ai.agent.name": agent.name,
                "gen_ai.agent.type": "react",
                "gen_ai.operation.name": "execute",
            }
        ) as agent_span:

            llm_calls = 0
            tool_calls = 0

            for iteration in range(agent.max_iterations):
                # LLM call (child span)
                with tracer.start_as_current_span(
                    "gen_ai.client.chat",
                    kind=trace.SpanKind.CLIENT,
                    attributes={
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": "gpt-4",
                        "gen_ai.agent.id": agent.id,
                    }
                ) as llm_span:
                    response = call_llm(agent, user_input)
                    llm_calls += 1
                    llm_span.set_attribute("gen_ai.usage.total_tokens", response.tokens)

                # Check if tool call needed
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        # Tool execution (child span)
                        with tracer.start_as_current_span(
                            "gen_ai.tool.execute",
                            kind=trace.SpanKind.CLIENT,
                            attributes={
                                "gen_ai.tool.name": tool_call.name,
                                "gen_ai.tool.type": "function",
                                "gen_ai.operation.name": "execute",
                                "gen_ai.agent.id": agent.id,
                            }
                        ) as tool_span:
                            try:
                                result = execute_tool(tool_call)
                                tool_calls += 1
                                tool_span.set_attribute("gen_ai.tool.duration_ms", ...)
                            except Exception as e:
                                tool_span.set_attribute("error.type", type(e).__name__)
                                tool_span.record_exception(e)
                                raise
                else:
                    # Final answer reached
                    break

            # Set agent span summary attributes
            agent_span.set_attribute("gen_ai.runtime.llm_calls_count", llm_calls)
            agent_span.set_attribute("gen_ai.runtime.tool_calls_count", tool_calls)
            agent_span.set_attribute("gen_ai.runtime.iterations", iteration + 1)

            return response.final_answer
```

---

## Conclusion

This specification provides a comprehensive framework for observing AI agent systems across 11 major frameworks. By adopting these semantic conventions, organizations can:

1. **Standardize Observability**: Consistent telemetry across frameworks enables unified monitoring dashboards and alerts
2. **Enable Debugging**: Rich span hierarchies and attributes make complex agent interactions traceable
3. **Optimize Performance**: Metrics enable identification of bottlenecks in agents, tools, memory, and workflows
4. **Ensure Quality**: Guardrails, evaluations, and HITL tracking support production readiness
5. **Manage Costs**: Token usage and cost metrics provide visibility into LLM expenses

**Next Steps**:
1. **Implement Instrumentation**: Update framework instrumentations in `opentelemetry-instrumentation-*` packages
2. **Validate with Real Systems**: Test conventions against production agent deployments
3. **Iterate and Refine**: Gather feedback from community and adjust conventions
4. **Propose to OpenTelemetry SIG**: Submit for standardization

**Contributing**:
This specification is a living document. Feedback and contributions are welcome via the Traceloop/OpenLLMetry repository.

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Agent** | An autonomous AI system that uses LLMs and tools to accomplish tasks |
| **Handoff** | Transfer of control from one agent to another (A2A communication) |
| **Team/Crew** | Group of agents working together on shared goals |
| **Task** | A unit of work assigned to an agent |
| **Tool** | A function, API, or capability an agent can invoke |
| **MCP** | Model Context Protocol - standardized interface for tools and resources |
| **Memory** | Short-term (conversation) or long-term (persistent) storage of information |
| **Workflow** | Orchestrated execution of multiple agents/steps (graph, sequential, parallel) |
| **Session** | A conversation or autonomous run encompassing multiple agent interactions |
| **Checkpoint** | Snapshot of agent/workflow state enabling pause/resume |
| **Guardrail** | Safety check or validation applied to agent inputs/outputs |
| **Evaluation** | Runtime assessment of agent response quality |
| **HITL** | Human-in-the-Loop - human intervention/approval in agent workflows |

---

**Document Version**: 0.1.0
**Last Updated**: 2025-01-23
**Maintainer**: Traceloop / OpenLLMetry Community
**License**: Apache 2.0
