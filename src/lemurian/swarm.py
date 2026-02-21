from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.message import Message, MessageRole
from lemurian.runner import Runner
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

    def _augment_agent(self, agent: Agent, handoff_tool: Tool) -> Agent:
        """Return a shallow copy of *agent* with the handoff tool appended."""
        augmented = agent.model_copy()
        augmented.tools = list(agent.tools) + [handoff_tool]
        return augmented

    async def run(self, user_message: str, agent: str | None = None) -> SwarmResult:
        """Append a user message and run the agent loop.

        On the first call, ``agent`` is required to set the initial
        active agent. On subsequent calls, the Swarm continues with
        the current active agent unless ``agent`` is specified.

        Args:
            user_message: The user's message to append to the transcript.
            agent: Name of the agent to run. Required on first call,
                optional thereafter.

        Returns:
            A SwarmResult with the final message, active agent, session,
            and state.

        Raises:
            ValueError: If ``agent`` is not provided on the first call.
        """
        # First call — initialise session and active agent
        if self.session is None:
            if agent is None:
                raise ValueError("agent must be specified on the first call to run()")
            self.session = Session(session_id=str(uuid.uuid4()))
            self.active_agent_name = agent
            self.context_start = 0
        elif agent is not None:
            self.active_agent_name = agent

        # Append user message to transcript
        self.session.transcript.append(
            Message(role=MessageRole.USER, content=user_message)
        )

        # Handoff loop
        for _ in range(self.max_handoffs + 1):
            current_agent = self.agents[self.active_agent_name]

            # Only create handoff tool if there are multiple agents
            if len(self.agents) > 1:
                handoff_tool = self._create_handoff_tool(self.active_agent_name)
                augmented_agent = self._augment_agent(current_agent, handoff_tool)
            else:
                augmented_agent = current_agent

            result = await self.runner.run(
                agent=augmented_agent,
                session=self.session,
                state=self.state,
                context_start=self.context_start,
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
                logger.warning(f"Handoff target '{target}' not found")
                error_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Error: agent '{target}' not found.",
                )
                self.session.transcript.append(error_msg)
                return SwarmResult(
                    last_message=error_msg,
                    active_agent=self.active_agent_name,
                    session=self.session,
                    state=self.state,
                )

            # Handoff occurred — append handoff message as user context for new agent
            self.session.transcript.append(
                Message(role=MessageRole.USER, content=result.hand_off.message)
            )
            self.context_start = len(self.session.transcript) - 1
            self.active_agent_name = target
            logger.info(f"Handoff to {self.active_agent_name} at context_start={self.context_start}")

        # Max handoffs exceeded
        error_msg = Message(
            role=MessageRole.ASSISTANT,
            content="Maximum handoffs exceeded. Please try again.",
        )
        self.session.transcript.append(error_msg)
        return SwarmResult(
            last_message=error_msg,
            active_agent=self.active_agent_name,
            session=self.session,
            state=self.state,
        )
