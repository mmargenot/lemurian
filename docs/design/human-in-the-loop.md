# Human-in-the-Loop Approval System

## Overview

A capability-based approval gate that intercepts tool calls before execution,
classifies them by risk, and routes them through configurable approval
policies — from fully autonomous to human-confirmed — with first-class
support for the streaming/SSE event pipeline.

### Design goals

- **Capability-native.** Approval is a `Capability` — not a Runner subclass,
  not a provider wrapper. It composes with agents the same way every other
  capability does.
- **Policy, not code.** Developers declare rules (`allow`, `ask`, `deny`)
  in a config dict. No subclassing required for common cases.
- **Streaming-aware.** Approval requests and decisions are `StreamEvent`s.
  SSE consumers see them in real time and can respond over the wire.
- **Progressive trust.** Start strict, widen as confidence grows —
  per-tool, per-pattern, per-session.

### Non-goals

- Authentication / identity (who the human is).
- Audit logging (orthogonal; built on top of events).
- Multi-party approval (quorum, voting).

---

## Prior Art & Influences

| System | Key pattern | What we take |
|--------|------------|--------------|
| **Claude Code** | 5 permission modes (`default`, `acceptEdits`, `plan`, `dontAsk`, `bypassPermissions`). Layered hooks: fast rule-based `PreToolUse` → slow LLM-based `PermissionRequest`. Allow/deny lists in `settings.json`. | Layered evaluation chain — fast deterministic rules first, optional escalation. |
| **OpenCode** | 3 actions (`allow` / `ask` / `deny`). Glob-pattern rules per tool. Per-agent overrides. Session-scoped "always approve" memory. `last-match-wins` evaluation. | Glob-pattern rules with per-agent override. Session-scoped approval memory. "Always" approve flow. |
| **Goose** | MCP tool annotations (`readOnly`, `destructive`). Smart approval mode: deterministic classification + LLM fallback. Policy-based gateway (`allow` read, `approve` write, `block` delete). | Tool-level annotations on the `@tool` decorator. Risk classification drives default policy. |
| **General HITL** | Approval gates, escalation triggers, just-in-time permissions, human-on-the-loop for low-risk autonomy. | Escalation trigger pattern — only interrupt for low-confidence or high-risk actions. |

---

## Architecture

### Structural overview

```
┌─────────────────────────────────────────────────────────┐
│                        Swarm                            │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ Agent A  │  │ Agent B  │  │ ApprovalCapability   │  │
│  │          │  │          │  │  ┌─────────────────┐  │  │
│  │ tools:   │  │ tools:   │  │  │ ApprovalPolicy  │  │  │
│  │  read_db │  │  deploy  │  │  │  (rules, mode)  │  │  │
│  │  query   │  │  restart │  │  └─────────────────┘  │  │
│  └──────────┘  └──────────┘  │  ┌─────────────────┐  │  │
│                              │  │ ApprovalHandler  │  │  │
│                              │  │ (async callback) │  │  │
│                              │  └─────────────────┘  │  │
│                              └──────────────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │                    Runner                          │ │
│  │                                                    │ │
│  │  iter() loop:                                      │ │
│  │    stream_complete → accumulate → [APPROVAL GATE]  │ │
│  │    → _execute_tools → yield events                 │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Where approval lives

Approval is **not** a tool the LLM calls. It is infrastructure that runs
between the LLM requesting a tool call and the Runner executing it.

```
LLM response
    │
    ▼
ToolCallAccumulator.finalize()
    │
    ▼
┌───────────────────────────┐
│   ApprovalGate.evaluate() │  ◄── Called by Runner before _execute_tools
│                           │
│  1. Check annotations     │  (tool.risk_level)
│  2. Evaluate policy rules │  (glob match, last-match-wins)
│  3. Check session memory  │  (previously "always" approved)
│  4. Route decision:       │
│     allow → proceed       │
│     deny  → reject msg    │
│     ask   → yield event,  │
│             await handler │
└───────────────────────────┘
    │
    ▼
  _execute_tools() or rejection message
```

---

## Data flow

### Full request lifecycle (streaming)

```
  Client                    SSE Transport              Runner + Approval
    │                           │                           │
    │  user message ──────────► │  ──────────────────────►  │
    │                           │                           │
    │                           │  ◄── RawResponseEvent ──  │  (token deltas)
    │  ◄── event: RawResponse   │                           │
    │                           │                           │
    │                           │  ◄── ApprovalRequestEvent │  (tool needs approval)
    │  ◄── event: ApprovalReq   │                           │
    │                           │                           │
    │  approval decision ─────► │  ──────────────────────►  │
    │                           │                           │
    │                           │  ◄── ApprovalResultEvent  │  (approved/denied)
    │  ◄── event: ApprovalRes   │                           │
    │                           │                           │
    │                           │  ◄── RunItemEvent ──────  │  (tool_call result)
    │  ◄── event: RunItem       │                           │
    │                           │                           │
    │                           │  ◄── RunCompleteEvent ──  │
    │  ◄── event: RunComplete   │                           │
```

### Approval decision flow

```
   Tool call arrives
          │
          ▼
  ┌───────────────┐    "deny"     ┌──────────────────┐
  │ Policy.decide │──────────────►│ Reject: send     │
  │ (tool, args)  │               │ ToolCallResult   │
  └───────┬───────┘               │ with denial msg  │
          │                       └──────────────────┘
          │ "allow"
          ├────────────────────────────► execute tool
          │
          │ "ask"
          ▼
  ┌───────────────────┐
  │ yield             │
  │ ApprovalRequest   │
  │ Event             │
  └───────┬───────────┘
          │
          ▼
  ┌───────────────────┐
  │ await             │
  │ handler(request)  │──────► human / UI / webhook / CLI prompt
  └───────┬───────────┘
          │
          ▼
  ┌───────────────────┐
  │ ApprovalDecision  │
  │  .approved        │──── True  → execute tool
  │  .denied          │──── True  → reject with message
  │  .always          │──── True  → add to session memory, execute
  └───────────────────┘
```

---

## Core abstractions

### 1. Tool annotations — `risk_level`

Extend the `@tool` decorator to accept a `risk_level` annotation.
This is the Goose/MCP pattern: tool authors declare intent,
the approval system uses it as a default signal.

```python
from enum import Enum

class RiskLevel(str, Enum):
    """Tool risk classification.

    Drives default approval policy when no explicit rule matches.
    """
    READ_ONLY = "read_only"      # No side effects — always safe
    WRITE = "write"              # Modifies local state
    DESTRUCTIVE = "destructive"  # Hard to reverse, affects shared state
```

Usage at the decorator site:

```python
from lemurian.tools import tool
from lemurian.approval import RiskLevel

@tool(risk_level=RiskLevel.READ_ONLY)
def search_db(query: str):
    """Search the database."""
    ...

@tool(risk_level=RiskLevel.DESTRUCTIVE)
def drop_table(table_name: str):
    """Drop a database table."""
    ...
```

Default when omitted: `RiskLevel.WRITE` (the safe middle ground — tools
that don't declare are assumed to write).

### 2. Approval policy — rules engine

A declarative policy evaluated at gate time. Inspired by OpenCode's
glob-pattern rules with last-match-wins semantics.

```python
from dataclasses import dataclass, field

@dataclass
class ApprovalRule:
    """A single pattern-matched rule."""
    pattern: str          # glob pattern matched against tool name (or name:args)
    action: str           # "allow" | "ask" | "deny"

@dataclass
class ApprovalPolicy:
    """Ordered rules + fallback behavior per risk level."""

    rules: list[ApprovalRule] = field(default_factory=list)

    # Fallback when no rule matches — keyed by RiskLevel
    risk_defaults: dict[str, str] = field(default_factory=lambda: {
        "read_only": "allow",
        "write": "ask",
        "destructive": "deny",
    })

    def decide(self, tool_name: str, arguments: str, risk_level: str) -> str:
        """Evaluate rules and return 'allow', 'ask', or 'deny'.

        Rules are evaluated in order. Last match wins (OpenCode pattern).
        If no rule matches, fall back to risk_defaults.
        """
        action = None
        for rule in self.rules:
            if _glob_match(rule.pattern, tool_name, arguments):
                action = rule.action
        if action is not None:
            return action
        return self.risk_defaults.get(risk_level, "ask")
```

#### Example policy configurations

**Autonomous mode** — trust everything:

```python
ApprovalPolicy(rules=[ApprovalRule(pattern="*", action="allow")])
```

**Strict mode** — ask for everything:

```python
ApprovalPolicy(rules=[ApprovalRule(pattern="*", action="ask")])
```

**Practical mode** — reads auto-approved, writes ask, destructive denied:

```python
ApprovalPolicy(
    rules=[
        ApprovalRule(pattern="search_*", action="allow"),
        ApprovalRule(pattern="get_*", action="allow"),
        ApprovalRule(pattern="drop_*", action="deny"),
        ApprovalRule(pattern="delete_*", action="deny"),
    ],
    risk_defaults={
        "read_only": "allow",
        "write": "ask",
        "destructive": "deny",
    },
)
```

**Per-agent override** (via Swarm):

```python
swarm.add_capability(
    ApprovalCapability(
        policy=ApprovalPolicy(rules=[
            ApprovalRule(pattern="deploy_*", action="ask"),
            ApprovalRule(pattern="*", action="allow"),
        ]),
    ),
    agents=["deploy_agent"],
)
```

### 3. Approval events — streaming integration

Two new event types extend the existing `StreamEvent` hierarchy:

```python
from dataclasses import dataclass, field
from lemurian.events import StreamEvent

@dataclass
class ApprovalRequestEvent(StreamEvent):
    """Emitted when a tool call requires human approval.

    The Runner pauses execution and waits for the handler to resolve.
    SSE consumers see this event and can present a UI.
    """
    request_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    risk_level: str = ""
    agent_name: str = ""

@dataclass
class ApprovalResultEvent(StreamEvent):
    """Emitted after an approval decision is made.

    Records the outcome for observability. Always follows
    an ApprovalRequestEvent with the same request_id.
    """
    request_id: str = ""
    tool_name: str = ""
    decision: str = ""   # "approved" | "denied" | "approved_always"
    reason: str = ""     # optional human-provided reason
```

### 4. Approval handler — the human interface

An async callback that the framework calls when a tool needs approval.
Implementations swap in different UIs without changing the core.

```python
from dataclasses import dataclass

@dataclass
class ApprovalRequest:
    """Passed to the handler — everything needed to make a decision."""
    request_id: str
    tool_name: str
    arguments: dict
    risk_level: str
    agent_name: str

@dataclass
class ApprovalDecision:
    """Returned by the handler."""
    approved: bool
    always: bool = False    # if True, add pattern to session memory
    reason: str = ""        # optional explanation

class ApprovalHandler:
    """Base class for approval handlers.

    Subclass and implement ``decide()`` for your transport.
    """
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        raise NotImplementedError
```

#### Built-in handlers

```python
class AutoApproveHandler(ApprovalHandler):
    """Approves everything. Use for testing or trusted environments."""
    async def decide(self, request):
        return ApprovalDecision(approved=True)

class AutoDenyHandler(ApprovalHandler):
    """Denies everything. Use for plan/read-only mode."""
    async def decide(self, request):
        return ApprovalDecision(approved=False, reason="Read-only mode")

class CLIApprovalHandler(ApprovalHandler):
    """Prompts in the terminal. For CLI applications."""
    async def decide(self, request):
        print(f"\n--- Approval required ---")
        print(f"Tool:      {request.tool_name}")
        print(f"Arguments: {request.arguments}")
        print(f"Risk:      {request.risk_level}")
        print(f"Agent:     {request.agent_name}")
        response = input("[a]pprove / [d]eny / [A]lways approve: ").strip()
        if response in ("a", "A", ""):
            return ApprovalDecision(
                approved=True,
                always=(response == "A"),
            )
        return ApprovalDecision(approved=False, reason="User denied")

class SSEApprovalHandler(ApprovalHandler):
    """Resolves via an async Future — for SSE/WebSocket transports.

    The web layer sets the future when the client sends a decision.
    """
    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}

    async def decide(self, request):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request.request_id] = future
        return await future   # Runner is paused here until resolved

    def resolve(self, request_id: str, decision: ApprovalDecision):
        """Called by the web endpoint when the user responds."""
        if request_id in self._pending:
            self._pending.pop(request_id).set_result(decision)
```

### 5. Approval capability — the composition point

Everything above wires together as a `Capability`. This is how it
attaches to agents and participates in the Swarm lifecycle.

```python
import uuid
from lemurian.capability import Capability
from lemurian.state import State

class ApprovalCapability(Capability):
    """Capability that provides an approval gate for tool execution.

    Attach to agents (or the entire Swarm) to require approval
    for tool calls based on policy rules and risk annotations.
    """

    def __init__(
        self,
        policy: ApprovalPolicy | None = None,
        handler: ApprovalHandler | None = None,
    ):
        super().__init__("approval")
        self.policy = policy or ApprovalPolicy()
        self.handler = handler or CLIApprovalHandler()
        self._session_memory: set[str] = set()  # "always approved" patterns

    def tools(self) -> list:
        """No tools — approval is infrastructure, not an LLM-callable tool."""
        return []

    def on_attach(self, state: State) -> None:
        """Reset session memory on attach."""
        self._session_memory.clear()

    def on_detach(self, state: State) -> None:
        pass

    async def evaluate(self, tool_name, arguments, risk_level, agent_name):
        """Run the approval gate. Returns an ApprovalDecision.

        Called by Runner before tool execution.
        """
        # 1. Check session memory (previously "always" approved)
        if tool_name in self._session_memory:
            return ApprovalDecision(approved=True)

        # 2. Evaluate policy rules
        action = self.policy.decide(tool_name, arguments, risk_level)

        if action == "allow":
            return ApprovalDecision(approved=True)

        if action == "deny":
            return ApprovalDecision(
                approved=False,
                reason=f"Policy denies '{tool_name}'",
            )

        # 3. action == "ask" — delegate to handler
        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            tool_name=tool_name,
            arguments=arguments if isinstance(arguments, dict) else {},
            risk_level=risk_level,
            agent_name=agent_name,
        )
        decision = await self.handler.decide(request)

        # 4. If "always", remember for this session
        if decision.approved and decision.always:
            self._session_memory.add(tool_name)

        return decision
```

### 6. Runner integration — the approval gate

The approval gate hooks into `Runner._execute_one()`, the single point
where every tool call passes through. This is a minimal, surgical change.

```python
# In runner.py — modified _execute_one method

async def _execute_one(
    self, tc: ToolCall, tool_registry: dict, ctx: Context,
) -> _ToolOutcome:
    tool_obj = tool_registry.get(tc.name)
    if tool_obj is None:
        return _ToolOutcome(output=f"Error: tool '{tc.name}' not found", is_error=True)

    # ── APPROVAL GATE ──────────────────────────────────────
    approval_cap = self._find_approval_capability(ctx.agent)
    if approval_cap is not None:
        risk_level = getattr(tool_obj, "risk_level", RiskLevel.WRITE)
        try:
            args_dict = json.loads(tc.arguments)
        except json.JSONDecodeError:
            args_dict = {}

        decision = await approval_cap.evaluate(
            tool_name=tc.name,
            arguments=args_dict,
            risk_level=risk_level,
            agent_name=ctx.agent.name,
        )
        if not decision.approved:
            reason = decision.reason or "Approval denied"
            return _ToolOutcome(output=f"Denied: {reason}", is_error=False)
    # ── END APPROVAL GATE ──────────────────────────────────

    # ... existing tool execution logic unchanged ...
```

Finding the capability:

```python
def _find_approval_capability(self, agent: Agent) -> ApprovalCapability | None:
    """Look for an ApprovalCapability in the agent's resolved capabilities."""
    for cap in agent.capabilities:
        if isinstance(cap, ApprovalCapability):
            return cap
    return None
```

### 7. Streaming integration — events in `iter()`

The `iter()` method yields approval events so SSE consumers can observe
and interact with the approval flow:

```python
# In runner.py — modified iter() loop, inside _execute_tools

async def _execute_tools_with_approval(self, calls, tool_registry, ctx, session):
    """Execute tools, yielding approval events when needed."""
    results = []
    for tc in calls:
        session.transcript.append(ToolCallRequestMessage(
            role=MessageRole.ASSISTANT, tool_calls=[tc],
        ))

        # Check approval — yield events for observability
        approval_cap = self._find_approval_capability(ctx.agent)
        if approval_cap is not None:
            risk_level = getattr(
                tool_registry.get(tc.name), "risk_level", RiskLevel.WRITE,
            )
            try:
                args_dict = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args_dict = {}

            action = approval_cap.policy.decide(
                tc.name, tc.arguments, risk_level,
            )

            if action == "ask" and tc.name not in approval_cap._session_memory:
                request_id = str(uuid.uuid4())

                # Yield request event for SSE consumers
                yield ApprovalRequestEvent(
                    request_id=request_id,
                    tool_name=tc.name,
                    arguments=args_dict,
                    risk_level=risk_level,
                    agent_name=ctx.agent.name,
                )

                decision = await approval_cap.handler.decide(
                    ApprovalRequest(
                        request_id=request_id,
                        tool_name=tc.name,
                        arguments=args_dict,
                        risk_level=risk_level,
                        agent_name=ctx.agent.name,
                    )
                )

                # Yield result event
                yield ApprovalResultEvent(
                    request_id=request_id,
                    tool_name=tc.name,
                    decision=(
                        "approved_always" if decision.always
                        else "approved" if decision.approved
                        else "denied"
                    ),
                    reason=decision.reason,
                )

                if not decision.approved:
                    session.transcript.append(ToolCallResultMessage(
                        role=MessageRole.TOOL,
                        content=f"Denied: {decision.reason or 'User denied'}",
                        tool_call_id=tc.id,
                    ))
                    results.append((tc, _ToolOutcome(
                        output=f"Denied: {decision.reason}",
                        is_error=False,
                    )))
                    continue

                if decision.always:
                    approval_cap._session_memory.add(tc.name)

        # Execute the tool (existing logic)
        outcome = await self._execute_one(tc, tool_registry, ctx)
        session.transcript.append(ToolCallResultMessage(
            role=MessageRole.TOOL, content=outcome.output,
            tool_call_id=tc.id,
        ))
        results.append((tc, outcome))
        if outcome.handoff is not None:
            break

    return results
```

---

## SSE transport

The existing `sse_generator` in `sse.py` requires **no changes** — it
already serializes any `StreamEvent` subclass by class name:

```python
async for event in event_stream:
    event_type = type(event).__name__   # "ApprovalRequestEvent", etc.
    data = json.dumps(asdict(event))
    yield f"event: {event_type}\ndata: {data}\n\n"
```

A web endpoint for receiving approval decisions:

```python
# Example: FastAPI endpoint for SSE approval resolution

from fastapi import FastAPI, Request
from lemurian.approval import ApprovalDecision

app = FastAPI()

@app.post("/approve/{request_id}")
async def approve(request_id: str, body: dict):
    decision = ApprovalDecision(
        approved=body.get("approved", False),
        always=body.get("always", False),
        reason=body.get("reason", ""),
    )
    sse_handler.resolve(request_id, decision)
    return {"status": "ok"}
```

---

## Usage examples

### Minimal — add approval to an existing Swarm

```python
from lemurian.approval import ApprovalCapability, ApprovalPolicy, ApprovalRule
from lemurian.swarm import Swarm

swarm = Swarm(agents=[support_agent, billing_agent], state=state)

# Approve reads, ask for writes, deny destructive
swarm.add_capability(ApprovalCapability(
    policy=ApprovalPolicy(
        rules=[
            ApprovalRule(pattern="search_*", action="allow"),
            ApprovalRule(pattern="get_*", action="allow"),
            ApprovalRule(pattern="delete_*", action="deny"),
        ],
    ),
))

result = await swarm.run("Cancel my subscription", agent="support")
```

### Per-agent policies

```python
# Read-only agent gets auto-approve
swarm.add_capability(
    ApprovalCapability(policy=ApprovalPolicy(
        rules=[ApprovalRule(pattern="*", action="allow")],
    )),
    agents=["search_agent"],
)

# Deploy agent requires confirmation for everything
swarm.add_capability(
    ApprovalCapability(policy=ApprovalPolicy(
        rules=[ApprovalRule(pattern="*", action="ask")],
    )),
    agents=["deploy_agent"],
)
```

### SSE web application

```python
from lemurian.approval import (
    ApprovalCapability, ApprovalPolicy, SSEApprovalHandler,
)
from lemurian.sse import sse_generator

handler = SSEApprovalHandler()

swarm = Swarm(agents=[agent], state=state)
swarm.add_capability(ApprovalCapability(
    policy=ApprovalPolicy(),
    handler=handler,
))

# SSE stream endpoint
async def stream(user_message: str):
    event_stream = swarm.iter(user_message, agent="assistant")
    async for chunk in sse_generator(event_stream):
        yield chunk

# Approval resolution endpoint
@app.post("/approve/{request_id}")
async def approve(request_id: str, body: dict):
    handler.resolve(request_id, ApprovalDecision(**body))
```

### Annotated tools

```python
from lemurian.tools import tool
from lemurian.approval import RiskLevel

@tool(risk_level=RiskLevel.READ_ONLY)
def list_users():
    """List all users."""
    return db.query("SELECT * FROM users")

@tool(risk_level=RiskLevel.WRITE)
def update_user(user_id: str, name: str):
    """Update a user's name."""
    db.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
    return "Updated"

@tool(risk_level=RiskLevel.DESTRUCTIVE)
def delete_user(user_id: str):
    """Permanently delete a user."""
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return "Deleted"
```

With the default `risk_defaults`, `list_users` auto-approves,
`update_user` triggers an ask, and `delete_user` is denied unless
an explicit rule overrides it.

---

## File layout

```
src/lemurian/
├── approval.py          # NEW — RiskLevel, ApprovalPolicy, ApprovalRule,
│                        #        ApprovalRequest, ApprovalDecision,
│                        #        ApprovalHandler + built-in handlers,
│                        #        ApprovalCapability
├── events.py            # MODIFIED — add ApprovalRequestEvent,
│                        #             ApprovalResultEvent
├── tools.py             # MODIFIED — @tool accepts risk_level kwarg,
│                        #             Tool model stores risk_level
├── runner.py            # MODIFIED — approval gate in _execute_one,
│                        #             approval events in iter()
├── capability.py        # UNCHANGED
├── swarm.py             # UNCHANGED
├── sse.py               # UNCHANGED (already generic)
├── streaming.py         # UNCHANGED
├── provider.py          # UNCHANGED
├── message.py           # UNCHANGED
├── agent.py             # UNCHANGED
├── context.py           # UNCHANGED
├── session.py           # UNCHANGED
└── state.py             # UNCHANGED

tests/
├── unit/
│   └── test_approval.py # NEW — policy evaluation, session memory,
│                        #        risk defaults, glob matching
└── integration/
    └── test_approval.py # NEW — full loop with MockProvider,
                         #        SSE approval flow, per-agent policies
```

---

## Implementation plan

| Phase | Scope | Files |
|-------|-------|-------|
| **1. Core types** | `RiskLevel`, `ApprovalPolicy`, `ApprovalRule`, `ApprovalRequest`, `ApprovalDecision`, `ApprovalHandler` base + `AutoApproveHandler` + `AutoDenyHandler` | `approval.py` |
| **2. Tool annotation** | Extend `@tool` to accept `risk_level`, store on `Tool` model | `tools.py` |
| **3. Approval events** | `ApprovalRequestEvent`, `ApprovalResultEvent` | `events.py` |
| **4. Runner gate** | Approval check in `_execute_one`, event yielding in iter path | `runner.py` |
| **5. Capability** | `ApprovalCapability` with policy, handler, session memory | `approval.py` |
| **6. CLI handler** | `CLIApprovalHandler` for terminal use | `approval.py` |
| **7. SSE handler** | `SSEApprovalHandler` with future-based resolution | `approval.py` |
| **8. Tests** | Unit tests for policy engine, integration tests for full loop | `tests/` |

---

## Open questions

1. **Parallel tool calls + approval.** When multiple tools fire in
   parallel and one needs approval, should we batch all approval
   requests into a single prompt? Or evaluate sequentially?
   *Proposed:* If any call in a parallel batch needs "ask", fall back
   to sequential execution for that batch.

2. **Glob syntax scope.** Should patterns match tool name only
   (`delete_*`) or tool name + serialized arguments
   (`deploy:*production*`)? *Proposed:* Name-only for v1, extend
   to name:args in v2.

3. **Timeout.** If the handler never responds (user walks away),
   should there be a configurable timeout that auto-denies?
   *Proposed:* Yes, default 5 minutes, configurable on the handler.

4. **Approval in transcript.** Should denied tool calls appear in the
   LLM transcript so the model can adapt? *Proposed:* Yes — append
   a `ToolCallResultMessage` with the denial reason so the LLM can
   try a different approach.
