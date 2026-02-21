from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lemurian.agent import Agent
    from lemurian.session import Session
    from lemurian.state import State


@dataclass
class Context:
    """Runtime context injected into tools that declare a ``context`` parameter.

    Provides tools with access to the current session, application state,
    and the agent being executed. The Runner creates a Context before
    dispatching tool calls and passes it automatically.

    Args:
        session: The active session containing the conversation transcript.
        state: The typed application state shared across turns and handoffs.
        agent: The agent currently being executed by the Runner.
    """

    session: Session
    state: State
    agent: Agent
