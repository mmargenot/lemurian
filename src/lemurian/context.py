from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lemurian.agent import Agent
    from lemurian.session import Session
    from lemurian.state import State


@dataclass
class Context:
    session: Session
    state: State
    agent: Agent
