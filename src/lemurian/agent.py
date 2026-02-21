from pydantic import BaseModel, Field

from lemurian.provider import ModelProvider
from lemurian.tools import Tool


class Agent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str = ""
    system_prompt: str
    tools: list[Tool] = Field(default_factory=list)
    model: str
    provider: ModelProvider

    @property
    def tool_registry(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}
