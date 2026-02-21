from pydantic import BaseModel


class State(BaseModel):
    """Base class for typed application state.

    Subclass this to define fields that persist across turns and
    handoffs. Tools can read and mutate state via ``context.state``.

    Example:
        class MyState(State):
            counter: int = 0
            customer_id: str | None = None
    """

    model_config = {"arbitrary_types_allowed": True}
