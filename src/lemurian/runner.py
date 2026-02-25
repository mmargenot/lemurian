import inspect
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.events import RawResponseEvent, RunCompleteEvent, RunItemEvent, StreamEvent
from lemurian.message import Message, MessageRole, ToolCallRequestMessage, ToolCallResultMessage
from lemurian.session import Session
from lemurian.state import State
from lemurian.streaming import ToolCall, ToolCallAccumulator
from lemurian.tools import HandoffResult, LLMRecoverableError

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """The result of a single Runner.run() invocation."""

    last_message: Message
    agent_name: str
    hand_off: HandoffResult | None = None


@dataclass
class _ToolOutcome:
    """Result of executing a single tool call."""

    output: str
    is_error: bool
    handoff: HandoffResult | None = None


class Runner:
    """Executes an agent's tool-calling loop.

    The Runner reads the session transcript and appends assistant
    responses, tool-call requests, and tool results during its loop.
    It injects the system prompt at call time (never storing it in
    the transcript), dispatches tool calls, and detects handoffs.

    ``run()`` drains ``iter()``.  ``iter()`` is the streaming entry point.

    Args:
        max_turns: Maximum number of provider round-trips before
            returning a timeout message.
    """

    def __init__(
        self,
        max_turns: int = 50,
        parallel_tool_calls: bool = True,
    ):
        self.max_turns = max_turns
        self.parallel_tool_calls = parallel_tool_calls

    async def run(
        self, agent: Agent, session: Session, state: State,
        context_start: int = 0,
    ) -> RunResult:
        """Run the agent loop until a final response or handoff."""
        result: RunResult | None = None
        async for event in self.iter(agent, session, state, context_start):
            if isinstance(event, RunCompleteEvent):
                result = event.result
        if result is None:
            raise RuntimeError("iter() ended without emitting RunCompleteEvent")
        return result

    async def iter(
        self, agent: Agent, session: Session, state: State,
        context_start: int = 0,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent loop, yielding events as execution proceeds."""
        ctx = Context(session=session, state=state, agent=agent)
        tool_registry = agent.tool_registry
        tool_schemas = [t.model_dump() for t in tool_registry.values()]

        for _turn in range(self.max_turns):
            messages = [
                {"role": "system", "content": agent.system_prompt},
                *[m.model_dump() for m in session.transcript[context_start:]],
            ]

            # Stream provider response
            acc = ToolCallAccumulator()
            full_content = ""
            async for chunk in agent.provider.stream_complete(
                model=agent.model, messages=messages,
                tools=tool_schemas if tool_schemas else None,
            ):
                if chunk.content_delta:
                    full_content += chunk.content_delta
                    yield RawResponseEvent(content=chunk.content_delta)
                if chunk.tool_call_fragments:
                    for frag in chunk.tool_call_fragments:
                        acc.feed(frag)

            completed_calls = acc.finalize()

            # No tool calls — final text response
            if not completed_calls:
                msg = Message(role=MessageRole.ASSISTANT, content=full_content)
                session.transcript.append(msg)
                yield RunItemEvent(name="message", data={"content": full_content})
                yield RunCompleteEvent(result=RunResult(
                    last_message=msg, agent_name=agent.name,
                ))
                return

            # Execute tool calls
            outcomes = await self._execute_tools(
                completed_calls, tool_registry, ctx, session,
            )
            for tc, outcome in outcomes:
                yield RunItemEvent(name="tool_call", data={
                    "tool_name": tc.name, "call_id": tc.id,
                    "output": outcome.output, "is_error": outcome.is_error,
                })
                if outcome.handoff is not None:
                    yield RunCompleteEvent(result=RunResult(
                        last_message=session.transcript[-1],
                        agent_name=agent.name, hand_off=outcome.handoff,
                    ))
                    return

        # Max turns exceeded
        timeout_msg = Message(
            role=MessageRole.ASSISTANT,
            content="Maximum turns reached. Please try again.",
        )
        session.transcript.append(timeout_msg)
        yield RunCompleteEvent(result=RunResult(
            last_message=timeout_msg, agent_name=agent.name,
        ))

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tools(
        self, calls: list[ToolCall], tool_registry: dict,
        ctx: Context, session: Session,
    ) -> list[tuple[ToolCall, _ToolOutcome]]:
        use_parallel = self.parallel_tool_calls and len(calls) > 1
        if use_parallel:
            return await self._execute_parallel(calls, tool_registry, ctx, session)
        return await self._execute_sequential(calls, tool_registry, ctx, session)

    async def _execute_sequential(self, calls, tool_registry, ctx, session):
        results = []
        for tc in calls:
            session.transcript.append(ToolCallRequestMessage(
                role=MessageRole.ASSISTANT, tool_calls=[tc],
            ))
            outcome = await self._execute_one(tc, tool_registry, ctx)
            session.transcript.append(ToolCallResultMessage(
                role=MessageRole.TOOL, content=outcome.output, tool_call_id=tc.id,
            ))
            results.append((tc, outcome))
            if outcome.handoff is not None:
                break
        return results

    async def _execute_parallel(self, calls, tool_registry, ctx, session):
        # Record all tool-call requests up front (parallel batch semantics).
        for tc in calls:
            session.transcript.append(ToolCallRequestMessage(
                role=MessageRole.ASSISTANT, tool_calls=[tc],
            ))
        # Execute sequentially so a handoff short-circuits remaining calls.
        results: list[tuple[ToolCall, _ToolOutcome]] = []
        for tc in calls:
            outcome = await self._execute_one(tc, tool_registry, ctx)
            session.transcript.append(ToolCallResultMessage(
                role=MessageRole.TOOL, content=outcome.output, tool_call_id=tc.id,
            ))
            results.append((tc, outcome))
            if outcome.handoff is not None:
                break
        return results

    async def _execute_one(self, tc: ToolCall, tool_registry: dict, ctx: Context) -> _ToolOutcome:
        tool_obj = tool_registry.get(tc.name)
        if tool_obj is None:
            logger.warning(f"Tool not found: {tc.name}")
            return _ToolOutcome(output=f"Error: tool '{tc.name}' not found", is_error=True)

        try:
            params = json.loads(tc.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in arguments for {tc.name}: {e}")
            return _ToolOutcome(output=f"Error: invalid arguments — {e}", is_error=True)

        logger.info(f"Calling {tc.name} with {params}")
        if "context" in inspect.signature(tool_obj.func).parameters:
            params["context"] = ctx

        try:
            result = await tool_obj(**params)
        except LLMRecoverableError as e:
            logger.info(f"Tool {tc.name} requested retry: {e}")
            return _ToolOutcome(output=str(e), is_error=False)
        except Exception as e:
            logger.error(f"Tool {tc.name} raised: {e}")
            return _ToolOutcome(output=f"Error calling {tc.name}: {e}", is_error=True)

        if isinstance(result.output, HandoffResult):
            return _ToolOutcome(
                output=f"Transferring to {result.output.target_agent}",
                is_error=False, handoff=result.output,
            )

        output_str = json.dumps(result.output) if not isinstance(result.output, str) else result.output
        return _ToolOutcome(output=output_str, is_error=False)
