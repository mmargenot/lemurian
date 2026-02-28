"""Local research agent with file search capability.

Demonstrates:

- Building a Capability with exact and semantic search over local files

- OpenTelemetry tracing with ConsoleSpanExporter

Usage:
    uv run --env-file=.env examples/local_research_agent_example.py --provider openai --model gpt-4o-mini --trace
    uv run examples/local_research_agent_example.py --provider vllm --url localhost:8000 --model Qwen/Qwen3-8B --trace
"""

import argparse
import asyncio
import concurrent.futures
import os
import pathlib
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer

from lemurian.agent import Agent
from lemurian.capability import Capability
from lemurian.message import Message, MessageRole
from lemurian.provider import ModelProvider, OpenAIProvider, OpenRouter, VLLMProvider
from lemurian.runner import Runner
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import Tool, tool, LLMRecoverableError

PROVIDERS = {
    "openai": lambda url: OpenAIProvider(),
    "openrouter": lambda url: OpenRouter(),
    "vllm": lambda url: VLLMProvider(url),
}


def make_provider(provider: str, url: str | None) -> ModelProvider:
    if provider == "vllm" and not url:
        raise SystemExit("--url is required for vllm provider")
    return PROVIDERS[provider](url)


def setup_tracing(service_name: str):
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor, ConsoleSpanExporter,
    )
    from lemurian.instrumentation import instrument

    provider = TracerProvider(
        resource=Resource({SERVICE_NAME: service_name})
    )
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    instrument()


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
        @tool
        def exact_search_files(keyword: str) -> list[str]:
            """Search all files in our local knowledge base for an exact
            keyword match. If the keyword is not found, returns nothing.
            Collects filenames that can be loaded into memory using the 
            ``open_file`` tool.

            Args:
                keyword (str): Keyword or topic to search the file system for
                    appearances.

            Returns:
                file_names (str): The names of files that contain the keyword.
            """
            file_names = []
            for f_name, f_path in self.file_map.items():
                with open(f_path, 'r') as f:
                    if keyword in f.read():
                        file_names.append(f_name)
            return file_names

        @tool
        def semantic_search_files(
                query: str,
                k: int,
                cutoff: float = 0.0
        ) -> list[str]:
            """Searches vector embeddings of files in our local knowledge base
            for a vector similarity match. Collects filenames that can be
            loaded into memory using the ``open_file`` tool.

            Args:
                query (str): Search query for semantic search across files.
                k (int): Number of results to return.
                cutoff (float, optional): Cutoff value for similarity search.
                    Defaults to 0, but can be set higher for more relevant
                    results.

            Notes:
                This tool is only useable if ``include_vectors = True`` when
                the user set up the assistant.
            """
            if not self.include_vectors:
                raise LLMRecoverableError(
                    "Research agent was not created with "
                    "``include_vectors = True``. If you want to use semantic"
                    " search then that parameter has to be enabled. Default "
                    "back to ``exact_search_files`` with appropriate keywords"
                    " instead."
                )
            if not self._index_future.done():
                raise LLMRecoverableError(
                    "Vector index is still building. Use "
                    "``exact_search_files`` in the meantime."
                )
            index = self._index_future.result()
            if not (cutoff >= 0.0 or cutoff < 1):
                raise LLMRecoverableError(
                    "Cutoff must be within the range [0, 1)"
                )
            query_vector = self.model.encode(query)[None, :self.dim]
            distances, indices = index.search(query_vector, k)
            all_f_names = list(self.file_map.keys())
            f_names = []
            for d, i in zip(distances[0], indices[0]):
                if d < cutoff:
                    continue
                f_names.append(all_f_names[i])
            return f_names

        @tool
        def open_file(file_name: str) -> str:
            """Open a file based on its file_name and read it into context.

            Args:
                file_name (str): The name of the file in the internal mapping.

            Returns:
                content (str): Contents of the named file.
            """
            # check to confirm whether file is safe
            if file_name not in self.file_map:
                raise LLMRecoverableError("")

            with open(self.file_map[file_name], 'r') as f:
                content = f.read()

            return content

        return [exact_search_files, semantic_search_files, open_file]


async def main():
    parser = argparse.ArgumentParser(description="Local research agent")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--url", default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.trace:
        setup_tracing("local-research-agent")

    provider = make_provider(args.provider, args.url)

    local_file_directory = pathlib.Path(__file__).parent / "local_research_agent_docs"
    agent = Agent(
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
        model=args.model,
        provider=provider,
    )

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
            agent=agent, session=session, state=state
        )
        print(f"Assistant: {result.last_message.content}\n")



if __name__ == "__main__":
    asyncio.run(main())
