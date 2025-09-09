from pydantic import BaseModel
from lemurian.message import Message


class Session(BaseModel):
    session_id: str
    transcript: list[Message]
