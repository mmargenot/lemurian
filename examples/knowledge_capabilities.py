"""Capability example: a SQLite-backed knowledge base.

Demonstrates what makes a good Capability:
- Encapsulates a real resource (SQLite database connection)
- Provides a cohesive group of CRUD tools that close over the resource
- Lifecycle hooks with real purpose (on_attach reports state, on_detach closes connection)
- Fully self-contained — tools never reach into shared application state

Usage:
    Add OPENAI_API_KEY=sk-... to .env, then:
    uv run --env-file=.env examples/capabilities.py
"""

import asyncio
import sqlite3

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.provider import OpenAIProvider
from lemurian.state import State
from lemurian.swarm import Swarm
from lemurian.tools import Tool, tool


# ---------------------------------------------------------------------------
# Capability: Knowledge Base (SQLite-backed)
# ---------------------------------------------------------------------------

class KnowledgeBaseCapability(Capability):
    """A persistent knowledge base backed by SQLite with topics and notes.

    Owns a SQLite connection with two tables: ``topics`` (named categories)
    and ``notes`` (content entries that belong to a topic via foreign key).
    All tools close over the connection — no shared application state needed.

    Args:
        db_path: Path to the SQLite database file, or ":memory:" for
            an in-memory database.
    """

    def __init__(self, db_path: str = ":memory:"):
        super().__init__("knowledge_base")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.commit()

    def tools(self) -> list[Tool]:
        conn = self._conn

        @tool
        def store_note(topic: str, content: str):
            """Store a note under a topic. Creates the topic if it doesn't exist."""
            conn.execute(
                "INSERT OR IGNORE INTO topics (name) VALUES (?)", (topic,)
            )
            conn.execute(
                """
                INSERT INTO notes (topic_id, content)
                VALUES ((SELECT id FROM topics WHERE name = ?), ?)
                """,
                (topic, content),
            )
            conn.commit()
            return f"Stored note under '{topic}'."

        @tool
        def get_notes(topic: str):
            """List all notes under a specific topic."""
            rows = conn.execute(
                """
                SELECT n.content, n.created_at
                FROM notes n
                JOIN topics t ON n.topic_id = t.id
                WHERE t.name = ?
                ORDER BY n.created_at
                """,
                (topic,),
            ).fetchall()
            if not rows:
                return f"No notes found under '{topic}'."
            lines = [f"  - {content} ({created_at})" for content, created_at in rows]
            return f"Notes under '{topic}':\n" + "\n".join(lines)

        @tool
        def search_notes(query: str):
            """Search notes by keyword across all topics."""
            rows = conn.execute(
                """
                SELECT t.name, n.content
                FROM notes n
                JOIN topics t ON n.topic_id = t.id
                WHERE n.content LIKE ? OR t.name LIKE ?
                ORDER BY t.name, n.created_at
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            if not rows:
                return "No matching notes found."
            return "\n".join(f"[{topic}] {content}" for topic, content in rows)

        @tool
        def list_topics():
            """List all topics with their note counts."""
            rows = conn.execute(
                """
                SELECT t.name, COUNT(n.id) as count
                FROM topics t
                LEFT JOIN notes n ON n.topic_id = t.id
                GROUP BY t.id
                ORDER BY t.name
                """
            ).fetchall()
            if not rows:
                return "No topics yet."
            return "\n".join(f"  {name} ({count} notes)" for name, count in rows)

        @tool
        def delete_topic(topic: str):
            """Delete a topic and all its notes."""
            cursor = conn.execute("DELETE FROM topics WHERE name = ?", (topic,))
            conn.commit()
            if cursor.rowcount == 0:
                return f"No topic found named '{topic}'."
            return f"Deleted topic '{topic}' and all its notes."

        return [store_note, get_notes, search_notes, list_topics, delete_topic]

    def on_attach(self, state: State) -> None:
        topics = self._conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        notes = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(f"[KnowledgeBase] Connected — {topics} topics, {notes} notes.")

    def on_detach(self, state: State) -> None:
        self._conn.close()
        print("[KnowledgeBase] Connection closed.")


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

provider = OpenAIProvider()

assistant = Agent(
    name="assistant",
    description="A research assistant with a persistent knowledge base.",
    system_prompt=(
        "You are a research assistant. Use your knowledge base tools to "
        "store, search, and organize notes for the user. When asked to "
        "remember something, store it as a note with an appropriate topic. "
        "When asked to recall something, search the knowledge base."
    ),
    capabilities=[KnowledgeBaseCapability("/tmp/lemurian_kb_example.db")],
    model="gpt-4o-mini",
    provider=provider,
)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    swarm = Swarm(agents=[assistant])

    print("Note-taking Assistant (with a DB)")
    print()

    first = True
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        result = await swarm.run(
            user_message=user_input,
            agent="assistant" if first else None,
        )
        first = False
        print(f"Assistant: {result.last_message.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
