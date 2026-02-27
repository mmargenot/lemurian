"""Single-agent example: a note-taking assistant.

Demonstrates:
- Defining tools with @tool (including context-aware tools)
- Subclassing State for typed application data
- Building an Agent with tools and a system prompt
- Running an interactive loop with Runner

Usage:
    Add OPENAI_API_KEY=sk-... to .env, then:
    uv run --env-file=.env examples/agent.py
"""

import asyncio
import uuid

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.message import Message, MessageRole
from lemurian.provider import OpenAIProvider
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import tool


class NoteState(State):
    """Typed state that persists across turns."""

    notes: dict[str, str] = {}


@tool
def add_note(context: Context, title: str, content: str):
    """Save a note with the given title and content."""
    context.state.notes[title] = content
    return f"Saved note '{title}'."


@tool
def get_note(context: Context, title: str):
    """Retrieve a note by title."""
    note = context.state.notes.get(title)
    if note is None:
        return f"No note found with title '{title}'."
    return note


@tool
def list_notes(context: Context):
    """List all saved note titles."""
    if not context.state.notes:
        return "No notes yet."
    return ", ".join(context.state.notes.keys())


@tool
def delete_note(context: Context, title: str):
    """Delete a note by title."""
    if title not in context.state.notes:
        return f"No note found with title '{title}'."
    del context.state.notes[title]
    return f"Deleted note '{title}'."


agent = Agent(
    name="notes",
    description="A note-taking assistant",
    system_prompt=(
        "You are a helpful note-taking assistant. "
        "Use the provided tools to manage the user's notes. "
        "When the user asks to save, find, list, or delete notes, "
        "always use the appropriate tool."
    ),
    tools=[add_note, get_note, list_notes, delete_note],
    model="gpt-4o-mini",
    provider=OpenAIProvider(),
)


async def main():
    runner = Runner()
    session = Session(session_id=str(uuid.uuid4()))
    state = NoteState()

    print("Note-taking Assistant)\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        session.transcript.append(
            Message(role=MessageRole.USER, content=user_input)
        )
        result = await runner.run(
            agent=agent, session=session, state=state
        )
        print(f"Assistant: {result.last_message.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
