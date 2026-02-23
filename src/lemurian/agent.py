from pydantic import BaseModel, Field

from lemurian.capability import Capability
from lemurian.provider import ModelProvider
from lemurian.tools import Tool


class Agent(BaseModel):
    """A declarative agent definition.

    Agents don't run themselves — the Runner executes them. An Agent
    bundles everything the Runner needs: a system prompt, tools, a
    model name, and a provider to call.

    Args:
        name: Unique name identifying this agent.
        description: Short description used in the Swarm's handoff tool
            to help the LLM choose which agent to hand off to.
        system_prompt: System prompt injected by the Runner at call time.
        tools: List of Tool objects available to this agent.
        capabilities: List of Capability instances whose tools are
            merged into this agent's tool registry at resolution time.
        model: Model identifier passed to the provider.
        provider: The model provider used for completions.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str = ""
    system_prompt: str
    tools: list[Tool] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    model: str
    provider: ModelProvider

    @property
    def tool_registry(self) -> dict[str, Tool]:
        """Return a mapping of tool names to Tool objects.

        Includes tools from both ``tools`` and ``capabilities``.
        """
        registry = {t.name: t for t in self.tools}
        for cap in self.capabilities:
            for t in cap.tools():
                registry[t.name] = t
        return registry
