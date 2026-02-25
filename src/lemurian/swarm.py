import logging
import uuid
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.handoff import Handoff
from lemurian.message import Message, MessageRole
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State

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
    """Multi-agent orchestrator with explicit handoffs.

    Manages a registry of agents and a single shared session.  On each
    ``run()`` call the Swarm resolves the active agent's tools and
    handoffs, then delegates to the Runner.  If the Runner reports a
    handoff, the Swarm advances the context window and switches to the
    target agent.

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
        self.agents: dict[str, Agent] = {
            a.name: a for a in agents
        }
        self.state = state or State()
        self.runner = runner or Runner()
        self.max_handoffs = max_handoffs
        self.session: Session | None = None
        self.active_agent_name: str | None = None
        self.context_start: int = 0

        self._capabilities: dict[str, Capability] = {}
        self._agent_capabilities: dict[str, set[str]] = {}

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
            ValueError: If any name in *agents* is not a registered
                agent.
        """
        targets = (
            agents
            if agents is not None
            else list(self.agents.keys())
        )
        for name in targets:
            if name not in self.agents:
                raise ValueError(
                    f"Agent '{name}' not found in swarm"
                )

        self._capabilities[capability.name] = capability
        for name in targets:
            self._agent_capabilities.setdefault(
                name, set()
            ).add(capability.name)
            capability.on_attach(self.state)

    def remove_capability(self, capability_name: str) -> None:
        """Remove a capability from the swarm and all assignments.

        Args:
            capability_name: Name of the capability to remove.

        Raises:
            KeyError: If the capability is not registered.
        """
        cap = self._capabilities.pop(capability_name)
        cap.on_detach(self.state)
        for agent_caps in self._agent_capabilities.values():
            agent_caps.discard(capability_name)

    def _resolve_handoffs(
        self, agent: Agent
    ) -> list[Handoff]:
        """Return the agent's declared handoffs."""
        return list(agent.handoffs)

    def _resolve_agent(self, agent: Agent) -> Agent:
        """Return a copy of *agent* with all tools resolved.

        Merges the agent's own tool registry (which includes
        agent-level capability tools) and swarm-level capability
        tools assigned to this agent.

        Raises:
            ValueError: If any two tools share the same name.
        """
        resolved = agent.model_copy()

        # Start from the agent's full tool registry
        all_tools = list(agent.tool_registry.values())

        # Add swarm-level capability tools
        for cap_name in self._agent_capabilities.get(
            agent.name, set()
        ):
            all_tools.extend(
                self._capabilities[cap_name].tools()
            )

        # Validate uniqueness
        seen: set[str] = set()
        for t in all_tools:
            if t.name in seen:
                raise ValueError(
                    f"Duplicate tool name: '{t.name}'"
                )
            seen.add(t.name)

        resolved.tools = all_tools
        # Prevent double-counting in tool_registry
        resolved.capabilities = []
        return resolved

    async def run(
        self,
        user_message: str,
        agent: str | None = None,
    ) -> SwarmResult:
        """Append a user message and run the agent loop.

        On the first call, ``agent`` is required to set the initial
        active agent.  On subsequent calls, the Swarm continues with
        the current active agent unless ``agent`` is specified.

        Args:
            user_message: The user's message to append to the
                transcript.
            agent: Name of the agent to run.  Required on first
                call, optional thereafter.

        Returns:
            A SwarmResult with the final message, active agent,
            session, and state.

        Raises:
            ValueError: If ``agent`` is not provided on the first
                call, or if a handoff tool name collides with a
                regular tool name.
        """
        # First call — initialise session and active agent
        if self.session is None:
            if agent is None:
                raise ValueError(
                    "agent must be specified on the first "
                    "call to run()"
                )
            self.session = Session(
                session_id=str(uuid.uuid4())
            )
            self.active_agent_name = agent
            self.context_start = 0
        elif agent is not None:
            self.active_agent_name = agent

        # Append user message to transcript
        self.session.transcript.append(
            Message(
                role=MessageRole.USER, content=user_message
            )
        )

        assert self.active_agent_name is not None

        # Handoff loop
        for _ in range(self.max_handoffs + 1):
            current_agent = self.agents[
                self.active_agent_name
            ]
            resolved_agent = self._resolve_agent(
                current_agent
            )
            handoffs = self._resolve_handoffs(current_agent)

            # Validate no name collisions between tools and
            # handoffs
            tool_names = set(
                resolved_agent.tool_registry.keys()
            )
            for h in handoffs:
                if h.tool_name in tool_names:
                    raise ValueError(
                        f"Handoff tool name '{h.tool_name}' "
                        "collides with an existing tool"
                    )

            result = await self.runner.run(
                agent=resolved_agent,
                session=self.session,
                state=self.state,
                context_start=self.context_start,
                handoffs=handoffs,
            )

            if result.hand_off is None:
                return SwarmResult(
                    last_message=result.last_message,
                    active_agent=self.active_agent_name,
                    session=self.session,
                    state=self.state,
                )

            # Validate handoff target exists
            target = result.hand_off.target_agent
            if target not in self.agents:
                logger.warning(
                    f"Handoff target '{target}' not found"
                )
                error_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        f"Error: agent '{target}' not found."
                    ),
                )
                self.session.transcript.append(error_msg)
                return SwarmResult(
                    last_message=error_msg,
                    active_agent=self.active_agent_name,
                    session=self.session,
                    state=self.state,
                )

            # Handoff — append handoff message as user context
            # for the new agent
            self.session.transcript.append(
                Message(
                    role=MessageRole.USER,
                    content=result.hand_off.message,
                )
            )
            self.context_start = (
                len(self.session.transcript) - 1
            )
            self.active_agent_name = target
            logger.info(
                f"Handoff to {self.active_agent_name} "
                f"at context_start={self.context_start}"
            )

        # Max handoffs exceeded
        error_msg = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "Maximum handoffs exceeded. "
                "Please try again."
            ),
        )
        self.session.transcript.append(error_msg)
        return SwarmResult(
            last_message=error_msg,
            active_agent=self.active_agent_name,
            session=self.session,
            state=self.state,
        )
