# lemurian

## Overview

Lemurian is a Python framework for building AI agents. It provides a
layered architecture for single-agent tool-calling loops and multi-agent
orchestration with dynamic handoffs. The framework is built on three
core principles:

1. **Declarative agents** — an Agent is a data object (Pydantic model)
   that bundles a system prompt, tools, model identifier, and provider.
   Agents never execute themselves.
2. **Single writer for conversation state** — the Runner is the only
   component that reads or writes the Session transcript. All other
   components receive state through explicit parameters.
3. **OpenAI-compatible wire format** — all message and tool schemas use
   the OpenAI chat completion format, allowing any provider that speaks
   this protocol to plug in.

---

## Architecture Layers

The framework is organized into four layers, each with a well-defined
responsibility boundary:

| Layer | Components | Responsibility |
|---|---|---|
| **Orchestration** | `Swarm` | Registers agents, creates handoff tool, manages handoff loop, tracks active agent |
| **Execution** | `Runner` | Builds messages for provider, injects system prompt, appends responses and tool results, detects handoffs |
| **Interface** | `Context`, `Tool`, `ModelProvider` | Tools use Context to access state/session/agent. Provider receives pre-built messages and schemas. |
| **Data** | `Session`, `State`, `Message` | Session holds the transcript (ground truth). State holds typed application data. |

```mermaid
graph TD
    subgraph Orchestration
        SW[Swarm]
    end

    subgraph Execution
        RN[Runner]
    end

    subgraph Interface
        CTX[Context]
        TL[Tool]
        PRV[ModelProvider]
    end

    subgraph Data
        SS[Session]
        ST[State]
        MSG[Message]
    end

    SW -->|delegates to| RN
    RN -->|reads/writes| SS
    RN -->|calls| PRV
    RN -->|dispatches| TL
    RN -->|creates & injects| CTX
    CTX -.->|references| SS
    CTX -.->|references| ST
    CTX -.->|references| AG[Agent]
    TL -.->|receives| CTX
    SS -->|contains| MSG
```

---

## Core Abstractions

### Tool System

The `@tool` decorator converts a plain Python function into a `Tool`
object with an OpenAI-compatible JSON schema. Schema generation happens
at decoration time, not at runtime.

```python
class Tool(BaseModel):
    """A callable tool with an OpenAI-compatible function schema.

    Wraps a sync or async function. The JSON schema is generated from
    the function's type hints and docstring at decoration time via
    @tool. Parameters named 'context' are excluded from the schema
    and injected by the Runner at call time.

    Responsibilities:
    - Serialize to OpenAI function-tool format via model_dump()
    - Execute the wrapped function (sync or async) via __call__
    - Return ToolCallResult containing tool name and output
    """

    func: Callable = Field(exclude=True)
    name: str
    description: str
    parameters_schema: dict

    def model_dump(self, **kwargs) -> dict:
        ...

    async def __call__(self, **kwargs) -> ToolCallResult:
        ...
```

**Special result types:**

- `ToolCallResult` — wraps the tool's return value with the tool name.
- `HandoffResult` — signals to the Runner that the conversation should
  transfer to a different agent. Contains `target_agent` and `message`.
- `LLMRecoverableError` — exception carrying tool-authored guidance
  that the LLM can use to retry with correct arguments.

**Schema generation pipeline:**

```mermaid
graph LR
    A["@tool decorator"] --> B["inspect.signature()"]
    B --> C["get_type_hints()"]
    C --> D["docstring_parser.parse()"]
    D --> E["_build_parameters_schema()"]
    E --> F["Tool(func, name, desc, schema)"]

```

Parameter descriptions are extracted from docstrings using
`docstring_parser` with auto-detection, supporting Google, NumPy,
Sphinx/reST, and Epydoc styles. The `context` parameter is silently
excluded from the generated schema.

### Agent

```python
class Agent(BaseModel):
    """Declarative agent definition — a data object, not an executor.

    Bundles everything the Runner needs to execute: system prompt,
    tools, model identifier, and provider. The tool_registry property
    merges tools from both the direct 'tools' list and any attached
    Capabilities.

    Agents are immutable declarations. The Runner receives an Agent
    and drives the loop; the Swarm may model_copy() an Agent to
    inject additional tools (e.g., the handoff tool).
    """

    name: str
    description: str = ""
    system_prompt: str
    tools: list[Tool] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    model: str
    provider: ModelProvider

    @property
    def tool_registry(self) -> dict[str, Tool]:
        ...
```

### Runner

```python
class Runner:
    """The agent execution loop — the only component that mutates
    the Session transcript.

    On each turn:
    1. Build messages = [system_prompt] + transcript[context_start:]
    2. Call provider.complete() with messages and tool schemas
    3. If response has no tool calls, append assistant message and return
    4. For each tool call:
       a. Append ToolCallRequestMessage to transcript
       b. Look up tool in registry, parse JSON args
       c. Inject context if tool signature has 'context' param
       d. Execute tool, handle errors (LLMRecoverableError, generic)
       e. If result is HandoffResult, return with hand_off set
       f. Otherwise append ToolCallResultMessage to transcript
    5. Repeat until max_turns exceeded

    The context_start parameter enables fresh-context handoffs:
    the Swarm advances it so each agent sees only the handoff
    message onward, not the full prior conversation.
    """

    def __init__(self, max_turns: int = 50):
        ...

    async def run(
        self,
        agent: Agent,
        session: Session,
        state: State,
        context_start: int = 0,
    ) -> RunResult:
        ...
```

### Swarm

```python
class Swarm:
    """Multi-agent orchestrator with dynamic handoffs.

    Manages an agent registry and a single shared Session. On each
    run() call:
    1. Append the user message to the transcript
    2. Resolve the active agent's tools (own + capabilities + handoff)
    3. Run the Runner
    4. If a handoff occurs, advance context_start and switch agents
    5. Repeat until a final response or max_handoffs exceeded

    The Swarm is stateful across run() calls — the same Session
    and State persist, making it suitable for interactive loops.

    Capabilities can be added at swarm level (shared across agents)
    or at agent level (self-contained). The Swarm validates that
    no two tools share the same name before execution.
    """

    def __init__(
        self,
        agents: list[Agent],
        state: State | None = None,
        runner: Runner | None = None,
        max_handoffs: int = 10,
    ):
        ...

    async def run(
        self, user_message: str, agent: str | None = None
    ) -> SwarmResult:
        ...
```

**Handoff data flow:**

```mermaid
sequenceDiagram
    participant U as User
    participant SW as Swarm
    participant R as Runner
    participant A1 as Agent A
    participant P as Provider
    participant A2 as Agent B

    U->>SW: run("question", agent="A")
    SW->>SW: Append user message
    SW->>SW: Resolve Agent A (tools + handoff)
    SW->>R: run(agent_A, session, state, context_start=0)
    R->>P: complete(messages, tools)
    P-->>R: tool_call: handoff(agent="B", message="...")
    R-->>SW: RunResult(hand_off=HandoffResult)
    SW->>SW: Append handoff message as user context
    SW->>SW: context_start = len(transcript) - 1
    SW->>SW: Resolve Agent B (tools + handoff)
    SW->>R: run(agent_B, session, state, context_start)
    R->>P: complete(messages, tools)
    P-->>R: Final text response
    R-->>SW: RunResult(hand_off=None)
    SW-->>U: SwarmResult
```

### Session, State, and Context

```python
class Session(BaseModel):
    """The conversation transcript — ground truth of the conversation.

    One Session is shared across all agents in a Swarm. Only the
    Runner appends to the transcript. The Swarm controls visibility
    via context_start windowing.
    """
    session_id: str
    transcript: list[Message] = []


class State(BaseModel):
    """Base class for typed application state.

    Subclass to add fields that persist across turns and handoffs.
    Mutated in-place by tools via context.state. The State is never
    serialized into the transcript — it exists as a side channel.
    """
    model_config = {"arbitrary_types_allowed": True}


@dataclass
class Context:
    """Runtime context injected into tools declaring a 'context' param.

    Created once per Runner.run() invocation. Provides tools with
    read/write access to the session, state, and current agent
    without those references appearing in the tool's JSON schema.
    """
    session: Session
    state: State
    agent: Agent
```

### Capability System

```python
class Capability(ABC):
    """A cohesive group of tools with shared state and lifecycle.

    Capabilities sit between individual tools and full agents.
    They group related tools, own internal state, and have lifecycle
    hooks that fire on attachment/detachment from a Swarm.

    Two attachment modes:
    - Agent-level: passed in Agent(capabilities=[...]). Tools are
      merged via agent.tool_registry. No lifecycle hooks called.
    - Swarm-level: added via swarm.add_capability(). Lifecycle hooks
      (on_attach, on_detach) are called with the shared State.

    The tools() method is called each time the Swarm resolves an
    agent's tool set, allowing capabilities to return tools that
    close over mutable internal state.
    """

    def __init__(self, name: str):
        ...

    @abstractmethod
    def tools(self) -> list[Tool]:
        ...

    def on_attach(self, state: State) -> None:
        ...

    def on_detach(self, state: State) -> None:
        ...
```

**Capability resolution flow:**

```mermaid
graph TD
    A[Agent.tools] --> D[Resolved Tool Set]
    B[Agent.capabilities] -->|"cap.tools()"| D
    C[Swarm._capabilities] -->|"cap.tools()"| D
    E[Swarm handoff tool] --> D
    D --> F{Duplicate names?}
    F -->|Yes| G[ValueError]
    F -->|No| H[Resolved Agent passed to Runner]
```

### Model Providers

```python
class ModelProvider:
    """Base class for LLM API providers.

    Providers receive pre-built message dicts and tool schema dicts
    from the Runner. They do not interact with lemurian types
    directly — the Runner handles all serialization.

    Two methods:
    - complete(): standard chat completion with optional tool calling
    - structured_completion(): parse response into a Pydantic model
    """

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        ...

    async def structured_completion(
        self,
        model: str,
        messages: list[dict],
        response_model: BaseModel,
    ):
        ...
```

**Provider implementations:**

| Provider | Backend | Auth |
|---|---|---|
| `OpenAIProvider` | OpenAI | `OPENAI_API_KEY` |
| `OpenRouter` | OpenRouter | `OPENROUTER_API_KEY` |
| `VLLMProvider` | vLLM | `VLLM_API_KEY` |

All providers use the `openai` Python package internally via
`AsyncOpenAI`, taking advantage of the OpenAI-compatible API format
that vLLM and OpenRouter expose.

---

## Component Relationship Diagram

```mermaid
classDiagram
    class Swarm {
        +agents: dict~str, Agent~
        +state: State
        +runner: Runner
        +session: Session
        +run(user_message, agent) SwarmResult
        +add_capability(capability, agents)
        +remove_capability(capability_name)
        -_resolve_agent(agent) Agent
        -_create_handoff_tool(name) Tool
    }

    class Runner {
        +max_turns: int
        +run(agent, session, state, context_start) RunResult
    }

    class Agent {
        +name: str
        +description: str
        +system_prompt: str
        +tools: list~Tool~
        +capabilities: list~Capability~
        +model: str
        +provider: ModelProvider
        +tool_registry: dict~str, Tool~
    }

    class Tool {
        +func: Callable
        +name: str
        +description: str
        +parameters_schema: dict
        +model_dump() dict
        +__call__(**kwargs) ToolCallResult
    }

    class Capability {
        <<abstract>>
        +name: str
        +tools()* list~Tool~
        +on_attach(state)
        +on_detach(state)
    }

    class Session {
        +session_id: str
        +transcript: list~Message~
    }

    class State {
        +model_config: dict
    }

    class Context {
        +session: Session
        +state: State
        +agent: Agent
    }

    class ModelProvider {
        <<abstract>>
        +complete(model, messages, tools)
        +structured_completion(model, messages, response_model)
    }

    class Message {
        +role: MessageRole
        +content: str
    }

    Swarm --> Runner : delegates to
    Swarm --> Agent : manages registry of
    Swarm --> Session : owns
    Swarm --> State : owns
    Swarm --> Capability : manages
    Runner --> Agent : executes
    Runner --> Session : reads/writes
    Runner --> Context : creates
    Agent --> Tool : contains
    Agent --> Capability : contains
    Agent --> ModelProvider : references
    Capability --> Tool : produces
    Context --> Session : references
    Context --> State : references
    Context --> Agent : references
    Session --> Message : contains
```

---

## Data Flow: Single Agent Run

```mermaid
graph TD
    A[User Message] --> B["Runner.run()"]
    B --> C["Build messages:\nsystem_prompt + transcript window"]
    C --> D["provider.complete()"]
    D --> E{Tool calls?}
    E -->|No| F[Append assistant message\nto transcript]
    F --> G[Return RunResult]
    E -->|Yes| H[For each tool_call]
    H --> I[Append ToolCallRequestMessage]
    I --> J[Parse JSON arguments]
    J --> K{Tool found?}
    K -->|No| L[Append error to transcript]
    K -->|Yes| M[Inject context if needed]
    M --> N[Execute tool]
    N --> O{Result type?}
    O -->|HandoffResult| P[Append transfer msg\nReturn with hand_off]
    O -->|LLMRecoverableError| Q[Append guidance to transcript]
    O -->|Exception| R[Append error to transcript]
    O -->|Normal| S[Append ToolCallResultMessage]
    L --> T[Next tool_call]
    Q --> T
    R --> T
    S --> T
    T --> C
```

---

## Glossary

| Term | Definition |
|---|---|
| **Agent** | Declarative configuration bundling prompt, tools, model, and provider |
| **Capability** | Reusable group of tools with lifecycle hooks |
| **Compaction** | Rewriting the transcript to reduce token count |
| **Context** | Runtime object injected into tools for state access |
| **Fresh-context handoff** | Technique where a new agent sees only the handoff message onward |
| **Handoff** | Transfer of conversation control from one agent to another |
| **MCP** | Model Context Protocol — standard for tool interoperability |
| **Provider** | Adapter for an LLM API (OpenAI, vLLM, etc.) |
| **Runner** | The agent execution loop; sole writer of the transcript |
| **Session** | Container for the conversation transcript |
| **State** | Typed application data persisting across turns and handoffs |
| **Swarm** | Multi-agent orchestrator managing handoffs and shared state |
| **Tool** | A decorated Python function exposed to an LLM with a JSON schema |
| **Transcript** | Ordered list of Messages forming the conversation history |
