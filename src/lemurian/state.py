from pydantic import BaseModel


class State(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
