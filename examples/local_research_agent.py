"""Local research agent with file search capability.

Demonstrates:

- Defining tools with ``@tool`` at module level (outside the Capability)

- Wiring capability state into tools with ``Tool.bind()``

- OpenTelemetry tracing with ConsoleSpanExporter

Usage:
    Add OPENAI_API_KEY=sk-... to .env, then:
    uv run --env-file=.env examples/local_research_agent.py
"""

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.message import Message, MessageRole
from lemurian.provider import OpenAIProvider
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import Tool, tool, LLMRecoverableError
from lemurian.instrumentation import instrument, uninstrument

import asyncio
import concurrent.futures
import os
import pathlib
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss


# ---------------------------------------------------------------------------
# Tools (standalone, capability-agnostic)
# ---------------------------------------------------------------------------


@tool
def exact_search_files(file_map: dict, keyword: str) -> list[str]:
    """Search all files in our local knowledge base for an exact
    keyword match. If the keyword is not found, returns nothing.
    Collects filenames that can be loaded into memory using the
    ``open_file`` tool.

    Args:
        file_map: Internal mapping of file names to file paths.
        keyword: Keyword or topic to search the file system for
            appearances.

    Returns:
        file_names: The names of files that contain the keyword.
    """
    file_names = []
    for f_name, f_path in file_map.items():
        with open(f_path, 'r') as f:
            if keyword in f.read():
                file_names.append(f_name)
    return file_names


@tool
def semantic_search_files(
        model: SentenceTransformer,
        dim: int,
        index_future: concurrent.futures.Future,
        file_map: dict,
        query: str,
        k: int,
        cutoff: float = 0.0,
) -> list[str]:
    """Searches vector embeddings of files in our local knowledge base
    for a vector similarity match. Collects filenames that can be
    loaded into memory using the ``open_file`` tool.

    Args:
        model: Sentence transformer model for encoding queries.
        dim: Embedding dimension.
        index_future: Future that resolves to the FAISS index.
        file_map: Internal mapping of file names to file paths.
        query: Search query for semantic search across files.
        k: Number of results to return.
        cutoff: Cutoff value for similarity search.
            Defaults to 0, but can be set higher for more relevant
            results.

    Notes:
        This tool is only useable if ``include_vectors = True`` when
        the user set up the assistant.
    """
    if not index_future.done():
        raise LLMRecoverableError(
            "Vector index is still building. Use "
            "``exact_search_files`` in the meantime."
        )
    index = index_future.result()
    if not (cutoff >= 0.0 or cutoff < 1):
        raise LLMRecoverableError(
            "Cutoff must be within the range [0, 1)"
        )
    query_vector = model.encode(query)[None, :dim]
    distances, indices = index.search(query_vector, k)
    all_f_names = list(file_map.keys())
    f_names = []
    for d, i in zip(distances[0], indices[0]):
        if d < cutoff:
            continue
        f_names.append(all_f_names[i])
    return f_names


@tool
def open_file(file_map: dict, file_name: str) -> str:
    """Open a file based on its file_name and read it into context.

    Args:
        file_map: Internal mapping of file names to file paths.
        file_name: The name of the file in the internal mapping.

    Returns:
        content: Contents of the named file.
    """
    if file_name not in file_map:
        raise LLMRecoverableError(
            f"File '{file_name}' not found. Use ``exact_search_files`` "
            "to find valid file names first."
        )

    with open(file_map[file_name], 'r') as f:
        content = f.read()

    return content


# ---------------------------------------------------------------------------
# Capability (owns resources, binds state into tools)
# ---------------------------------------------------------------------------


class ResearchCapability(Capability):
    """
    Search tools and analysis for examining local files.
    """
    def __init__(
            self,
            root: pathlib.Path,
            include_vectors: bool = False
    ) -> None:
        if not root.is_dir():
            raise ValueError("Input research ``root`` was not a directory.")

        self.root = root
        all_files = self.find_all_files()
        self.file_map = {f.name: f for f in all_files}

        self.include_vectors = include_vectors
        if self.include_vectors:
            print("Initializing semantic search (background)")
            embedding_size = 256  # model is matryoshka, supports up to 1024
            self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            self.dim = embedding_size
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._index_future = self._executor.submit(
                self.initialize_vectors, model=self.model, dim=embedding_size
            )
        else:
            self.model = None
            self.dim = None
            self._index_future = None

    def find_all_files(self) -> list[pathlib.Path]:
        filter_dir = [".DS_Store"]
        file_paths = []
        directories = [self.root]

        while directories:
            curr_dir = directories.pop(0)
            if curr_dir.name in filter_dir:
                continue
            for path in curr_dir.iterdir():
                if path.is_file():
                    file_paths.append(path)
                if path.is_dir():
                    directories.append(path)
        return file_paths

    def efficient_embedding(self, content: str, chunk_length: int = 512, chunk_overlap: int = 64):

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        tokens = tokenizer.encode(content)

        # Chunk token IDs with overlap
        step = chunk_length - chunk_overlap
        chunks = []
        for start in range(0, len(tokens), step):
            chunk_ids = tokens[start:start + chunk_length]
            if chunk_ids:
                chunks.append(tokenizer.decode(chunk_ids))

        if not chunks:
            return self.model.encode(content)

        embeddings = self.model.encode(chunks, batch_size=16)
        return np.mean(embeddings, axis=0)

    def initialize_vectors(self, model, dim: int):
        """
        uses
        """
        index = faiss.IndexHNSWFlat(dim, 32)
        for _, f_path in self.file_map.items():
            with open(f_path) as f:
                content = f.read()
                embedding = self.efficient_embedding(content)
                index.add(embedding[None, :dim])
        return index

    def tools(self) -> list[Tool]:
        result = [
            exact_search_files.bind(file_map=self.file_map),
            open_file.bind(file_map=self.file_map),
        ]
        if self.include_vectors:
            result.append(
                semantic_search_files.bind(
                    model=self.model,
                    dim=self.dim,
                    index_future=self._index_future,
                    file_map=self.file_map,
                )
            )
        return result


provider = OpenAIProvider()
local_file_directory = pathlib.Path(__file__).parent / "local_research_agent_docs"
local_research_agent = Agent(
    name="research_assistant",
    description=(
        "A research assistant with the ability to search over a local file "
        "system."
    ),
    system_prompt=(
        "You are a research assistant. Use your knowledge base tools to "
        "store, search, and organize notes for the user. When asked to "
        "remember something, store it as a note with an appropriate topic. "
        "When asked to recall something, search the knowledge base."
    ),
    capabilities=[ResearchCapability(local_file_directory, include_vectors=True)],
    model="gpt-4o-mini",
    provider=provider,
)


async def main():
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    tracer_provider = TracerProvider(
        resource=Resource({SERVICE_NAME: "local-research-agent"})
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    instrument()
    runner = Runner()
    session = Session(session_id=str(uuid.uuid4()))
    state = State()

    print("Local Research Assistant\n")

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
            agent=local_research_agent, session=session, state=state
        )
        print(f"Assistant: {result.last_message.content}\n")

    uninstrument()


if __name__ == "__main__":
    asyncio.run(main())
