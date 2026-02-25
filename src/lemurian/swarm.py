import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.context import Context
from lemurian.events import RunCompleteEvent, RunItemEvent, StreamEvent
from lemurian.message import Message, MessageRole
from lemurian.runner import Runner, RunResult
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import HandoffResult, Tool

logger = logging.getLogger(__name__)


@dataclass
class SwarmResult:
    """The result of a Swarm.run() invocation.

    Args:
        last_message: The final message from the active agent.
        active_agent: Name of the agent that produced the final message.
        session: The session containing the full transcript.
        state: The application state after execution.
    """

    last_message: Message
    active_agent: str
    session: Session
    state: State


class Swarm:
    """Multi-agent orchestrator with dynamic handoffs.

    Manages a registry of agents and a single shared session. On each
    ``run()`` call, the Swarm injects a ``handoff`` tool into the active
    agent and runs the Runner. If a handoff occurs, the Swarm advances
    the context window and switches to the target agent.

    Stateful across ``run()`` calls — the same session and state persist.

    Args:
        agents: List of agents to register.
        state: Application state, or a default empty State.
        runner: Runner instance, or a default Runner.
        max_handoffs: Maximum handoffs allowed per ``run()`` call.
    """

    def __init__(
        self,
        agents: list[Agent],
        state: State | None = None,
        runner: Runner | None = None,
        max_handoffs: int = 10,
    ):
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self.state = state or State()
        self.runner = runner or Runner()
        self.max_handoffs = max_handoffs
        self.session: Session | None = None
        self.active_agent_name: str | None = None
        self.context_start: int = 0

        self._capabilities: dict[str, Capability] = {}
        self._agent_capabilities: dict[str, set[str]] = {}

    def _create_handoff_tool(self, current_agent_name: str) -> Tool:
        """Create a handoff tool excluding the current agent from the enum.

        Args:
            current_agent_name: The active agent to exclude from the
                list of available handoff targets.

        Returns:
            A Tool whose schema includes an enum of other agent names.
        """
        available = {
            name: agent.description
            for name, agent in self.agents.items()
            if name != current_agent_name
        }
        agent_names = list(available.keys())
        agent_info = ", ".join(
            f"{name} ({desc})" if desc else name
            for name, desc in available.items()
        )

        def handoff_func(context: Context, agent_name: str, message: str) -> HandoffResult:
            return HandoffResult(target_agent=agent_name, message=message)

        return Tool(
            func=handoff_func,
            name="handoff",
            description=f"Hand off the conversation to another agent. Available: {agent_info}",
            parameters_schema={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": agent_names,
                        "description": "The agent to transfer to",
                    },
                    "message": {
                        "type": "string",
                        "description": "Summary of context and instructions for the next agent",
                    },
                },
                "required": ["agent_name", "message"],
            },
        )

    def add_capability(
        self,
        capability: Capability,
        agents: list[str] | None = None,
    ) -> None:
        """Register a capability and assign it to agents.

        Args:
            capability: The capability to add.
            agents: Agent names to receive this capability.  When
                ``None``, the capability is assigned to every agent
                currently in the swarm.

        Raises:
            ValueError: If any name in *agents* is not a registered agent.
        """
        targets = agents if agents is not None else list(self.agents.keys())
        for name in targets:
            if name not in self.agents:
                raise ValueError(f"Agent '{name}' not found in swarm")

        self._capabilities[capability.name] = capability
        for name in targets:
            self._agent_capabilities.setdefault(name, set()).add(capability.name)
            capability.on_attach(self.state)

    def remove_capability(self, capability_name: str) -> None:
        """Remove a capability from the swarm and all agent assignments.

        Args:
            capability_name: Name of the capability to remove.

        Raises:
            KeyError: If the capability is not registered.
        """
        cap = self._capabilities.pop(capability_name)
        cap.on_detach(self.state)
        for agent_caps in self._agent_capabilities.values():
            agent_caps.discard(capability_name)

    def _resolve_agent(self, agent: Agent) -> Agent:
        """Return a copy of *agent* with all tools resolved.

        Merges the agent's own tool registry (which includes agent-level
        capability tools), swarm-level capability tools assigned to this
        agent, and the handoff tool (when multiple agents exist).

        Raises:
            ValueError: If any two tools share the same name.
        """
        resolved = agent.model_copy()

        # Start from the agent's full tool registry (own tools + agent-level capabilities)
        all_tools = list(agent.tool_registry.values())

        # Add swarm-level capability tools assigned to this agent
        for cap_name in self._agent_capabilities.get(agent.name, set()):
            all_tools.extend(self._capabilities[cap_name].tools())

        # Add handoff tool if multiple agents
        if len(self.agents) > 1:
            all_tools.append(self._create_handoff_tool(agent.name))

        # Validate uniqueness
        seen: set[str] = set()
        for t in all_tools:
            if t.name in seen:
                raise ValueError(f"Duplicate tool name: '{t.name}'")
            seen.add(t.name)

        resolved.tools = all_tools
        resolved.capabilities = []  # prevent double-counting in tool_registry
        return resolved

    def _init_session(self, user_message: str, agent: str | None) -> None:
        """Set up session and active agent, append user message."""
        if self.session is None:
            if agent is None:
                raise ValueError("agent must be specified on the first call")
            self.session = Session(session_id=str(uuid.uuid4()))
            self.active_agent_name = agent
            self.context_start = 0
        elif agent is not None:
            self.active_agent_name = agent
        self.session.transcript.append(
            Message(role=MessageRole.USER, content=user_message)
        )

    async def run(self, user_message: str, agent: str | None = None) -> SwarmResult:
        """Append a user message and run the agent loop."""
        last_result = None
        async for event in self.iter(user_message, agent):
            if isinstance(event, RunCompleteEvent):
                last_result = event.result
        if last_result is None:
            raise RuntimeError("iter() ended without emitting RunCompleteEvent")
        assert self.active_agent_name is not None
        assert self.session is not None
        return SwarmResult(
            last_message=last_result.last_message,
            active_agent=self.active_agent_name,
            session=self.session,
            state=self.state,
        )

    async def iter(
        self, user_message: str, agent: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming entry point. Yields events including handoffs."""
        self._init_session(user_message, agent)
        assert self.active_agent_name is not None
        assert self.session is not None
        session = self.session
        for _ in range(self.max_handoffs + 1):
            current_agent = self.agents[self.active_agent_name]
            resolved_agent = self._resolve_agent(current_agent)

            last_run_result = None
            async for event in self.runner.iter(
                agent=resolved_agent, session=session,
                state=self.state, context_start=self.context_start,
            ):
                if isinstance(event, RunCompleteEvent):
                    last_run_result = event.result
                else:
                    yield event

            if last_run_result is None:
                return

            if last_run_result.hand_off is None:
                yield RunCompleteEvent(result=last_run_result)
                return

            # Validate handoff target
            target = last_run_result.hand_off.target_agent
            if target not in self.agents:
                logger.warning(f"Handoff target '{target}' not found")
                error_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Error: agent '{target}' not found.",
                )
                session.transcript.append(error_msg)
                yield RunCompleteEvent(result=RunResult(
                    last_message=error_msg,
                    agent_name=self.active_agent_name,
                ))
                return

            # Handoff
            yield RunItemEvent(
                name="handoff",
                data={"target_agent": target, "message": last_run_result.hand_off.message},
            )
            session.transcript.append(
                Message(role=MessageRole.USER, content=last_run_result.hand_off.message)
            )
            self.context_start = len(session.transcript) - 1
            self.active_agent_name = target
            logger.info(f"Handoff to {self.active_agent_name} at context_start={self.context_start}")

        # Max handoffs exceeded
        error_msg = Message(
            role=MessageRole.ASSISTANT,
            content="Maximum handoffs exceeded. Please try again.",
        )
        session.transcript.append(error_msg)
        yield RunCompleteEvent(result=RunResult(
            last_message=error_msg,
            agent_name=self.active_agent_name,
        ))
