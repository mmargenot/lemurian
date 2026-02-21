from pydantic import BaseModel
from lemurian.message import Message


class Session(BaseModel):
    """A conversation session containing the message transcript.

    One Session is shared across all agents in a Swarm. The transcript
    is the ground truth of the full conversation.

    Args:
        session_id: Unique identifier for this session.
        transcript: Ordered list of messages in the conversation.
    """

    session_id: str
    transcript: list[Message] = []
