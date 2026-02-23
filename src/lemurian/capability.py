from abc import ABC, abstractmethod

from lemurian.state import State
from lemurian.tools import Tool


class Capability(ABC):
    """A cohesive group of tools with shared state and lifecycle.

    Capabilities are the unit between a tool and an agent. They group
    related tools, own their own state, and have lifecycle hooks that
    fire when they are added to or removed from a Swarm.

    The framework does not manage capability state — each capability
    decides what to persist and how, using ``on_attach`` to load and
    ``on_detach`` to flush.

    Args:
        name: Unique name identifying this capability.

    Example::

        class OrderTracking(Capability):
            def __init__(self, db: OrderDB):
                super().__init__("order_tracking")
                self._db = db

            def tools(self) -> list[Tool]:
                db = self._db

                @tool
                def check_order(context: Context, order_id: str):
                    return db.get(order_id)

                return [check_order]
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def tools(self) -> list[Tool]:
        """Return the tools this capability provides.

        Called each time the Swarm resolves an agent's tool set.
        Tools typically close over ``self`` to access the capability's
        internal state.
        """
        ...

    def on_attach(self, state: State) -> None:
        """Called when the capability is added to a Swarm.

        Override to load persisted state or acquire resources.

        Args:
            state: The shared application state.
        """

    def on_detach(self, state: State) -> None:
        """Called when the capability is removed from a Swarm.

        Override to persist state or release resources.

        Args:
            state: The shared application state.
        """
