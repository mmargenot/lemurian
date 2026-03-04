import inspect
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.events import (
    MessageEvent,
    RawResponseEvent,
    RunCompleteEvent,
    StreamEvent,
    ToolCallEvent,
)
from lemurian.handoff import Handoff, HandoffResult
from lemurian.instrumentation import (
    agent_span,
    completion_span,
    record_error,
    record_usage,
    tool_span,
)
from lemurian.message import (
    Message,
    MessageRole,
    ToolCallRequestMessage,
    ToolCallResultMessage,
)
from lemurian.session import Session
from lemurian.state import State
from lemurian.streaming import ToolCall, ToolCallAccumulator
from lemurian.tools import LLMRecoverableError

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """The result of a single Runner.run() invocation.

    Args:
        last_message: The final message appended to the transcript.
        agent_name: Name of the agent that was executed.
        hand_off: Set when the model called a handoff tool,
            ``None`` otherwise.
    """

    last_message: Message
    agent_name: str
    hand_off: HandoffResult | None = None


@dataclass
class _ToolOutcome:
    """Result of executing a single tool call."""

    output: str
    is_error: bool


class Runner:
    """Executes an agent's tool-calling loop.

    The Runner reads the session transcript and appends assistant
    responses, tool-call requests, and tool results during its loop.
    It injects the system prompt at call time (never storing it in
    the transcript), dispatches tool calls, and classifies handoffs.

    ``run()`` drains ``iter()``.  ``iter()`` is the streaming entry point.

    Args:
        max_turns: Maximum number of provider round-trips before
            returning a timeout message.
        parallel_tool_calls: When True and the model returns multiple
            tool calls, record all requests up front then execute
            sequentially.  When False, record and execute one at a time.
    """

    def __init__(
        self,
        max_turns: int = 50,
        parallel_tool_calls: bool = True,
    ):
        self.max_turns = max_turns
        self.parallel_tool_calls = parallel_tool_calls

    async def run(
        self,
        agent: Agent,
        session: Session,
        state: State,
        context_start: int = 0,
        handoffs: list[Handoff] | None = None,
    ) -> RunResult:
        """Run the agent loop until a final response or handoff.

        Args:
            agent: The agent to execute.
            session: The session containing the conversation
                transcript.
            state: The application state passed to tools via Context.
            context_start: Transcript index to start reading from.
            handoffs: Optional list of Handoff objects.

        Returns:
            A RunResult with the final message and optional handoff.
        """
        result: RunResult | None = None
        async for event in self.iter(
            agent, session, state, context_start, handoffs,
        ):
            if isinstance(event, RunCompleteEvent):
                result = event.result
        if result is None:
            raise RuntimeError(
                "iter() ended without emitting RunCompleteEvent"
            )
        return result

    async def iter(
        self,
        agent: Agent,
        session: Session,
        state: State,
        context_start: int = 0,
        handoffs: list[Handoff] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent loop, yielding events as execution proceeds.

        Args:
            agent: The agent to execute.
            session: The session containing the conversation
                transcript.
            state: The application state passed to tools via Context.
            context_start: Transcript index to start reading from.
            handoffs: Optional list of Handoff objects.

        Yields:
            StreamEvent subclasses as execution progresses.
        """
        ctx = Context(session=session, state=state, agent=agent)
        tool_registry = agent.tool_registry

        # Build handoff map (tool_name -> Handoff)
        handoff_map: dict[str, Handoff] = {
            h.tool_name: h for h in (handoffs or [])
        }

        # Merge tool schemas: regular tools + handoff tools
        tool_schemas = [
            t.model_dump() for t in tool_registry.values()
        ]
        tool_schemas += [
            h.tool_schema() for h in (handoffs or [])
        ]
        system_name = getattr(
            agent.provider, "system_name", "unknown"
        )

        async with agent_span(agent.name, agent.model):
            for _turn in range(self.max_turns):
                # Build messages: system prompt + transcript window
                transcript_window = (
                    session.transcript[context_start:]
                )
                messages = [
                    {
                        "role": "system",
                        "content": agent.system_prompt,
                    },
                    *[
                        m.model_dump()
                        for m in transcript_window
                    ],
                ]

                # Stream provider response
                acc = ToolCallAccumulator()
                full_content = ""
                last_usage = None
                last_response_model = None

                async with completion_span(
                    system_name, agent.model
                ) as c_span:
                    async for chunk in (
                        agent.provider.stream_complete(
                            model=agent.model,
                            messages=messages,
                            tools=(
                                tool_schemas
                                if tool_schemas
                                else None
                            ),
                        )
                    ):
                        if chunk.content_delta:
                            full_content += chunk.content_delta
                            yield RawResponseEvent(
                                content=chunk.content_delta
                            )
                        if chunk.tool_call_fragments:
                            for frag in chunk.tool_call_fragments:
                                acc.feed(frag)
                        if chunk.usage is not None:
                            last_usage = chunk.usage
                        if chunk.response_model is not None:
                            last_response_model = (
                                chunk.response_model
                            )

                    record_usage(
                        c_span,
                        last_usage,
                        last_response_model,
                    )

                completed_calls = acc.finalize()

                # No tool calls — final text response
                if not completed_calls:
                    assistant_msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                    )
                    session.transcript.append(assistant_msg)
                    yield MessageEvent(content=full_content)
                    yield RunCompleteEvent(
                        result=RunResult(
                            last_message=assistant_msg,
                            agent_name=agent.name,
                        )
                    )
                    return

                # Classify handoffs BEFORE tool execution
                handoff_tc = None
                regular_calls: list[ToolCall] = []
                for tc in completed_calls:
                    if tc.name in handoff_map:
                        handoff_tc = tc
                        break
                    regular_calls.append(tc)

                # Execute regular tools up to the handoff
                if regular_calls:
                    outcomes = await self._execute_tools(
                        regular_calls,
                        tool_registry,
                        ctx,
                        session,
                    )
                    for tc, outcome in outcomes:
                        yield ToolCallEvent(
                            tool_name=tc.name,
                            tool_call_id=tc.id,
                            arguments=tc.arguments,
                            output=outcome.output,
                            is_error=outcome.is_error,
                        )

                # Handle handoff if present
                if handoff_tc is not None:
                    handoff_obj = handoff_map[handoff_tc.name]
                    try:
                        args = json.loads(
                            handoff_tc.arguments
                        )
                    except json.JSONDecodeError:
                        args = {}
                    message = args.get("message", "")

                    session.transcript.append(
                        ToolCallRequestMessage(
                            role=MessageRole.ASSISTANT,
                            tool_calls=[handoff_tc],
                        )
                    )
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=(
                                "Transferring to "
                                f"{handoff_obj.target_agent}"
                            ),
                            tool_call_id=handoff_tc.id,
                        )
                    )
                    yield RunCompleteEvent(
                        result=RunResult(
                            last_message=(
                                session.transcript[-1]
                            ),
                            agent_name=agent.name,
                            hand_off=HandoffResult(
                                target_agent=(
                                    handoff_obj.target_agent
                                ),
                                message=message,
                            ),
                        )
                    )
                    return

            # Max turns exceeded
            timeout_msg = Message(
                role=MessageRole.ASSISTANT,
                content=(
                    "Maximum turns reached. "
                    "Please try again."
                ),
            )
            session.transcript.append(timeout_msg)
            yield RunCompleteEvent(
                result=RunResult(
                    last_message=timeout_msg,
                    agent_name=agent.name,
                )
            )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tools(
        self,
        calls: list[ToolCall],
        tool_registry: dict,
        ctx: Context,
        session: Session,
    ) -> list[tuple[ToolCall, _ToolOutcome]]:
        use_parallel = (
            self.parallel_tool_calls and len(calls) > 1
        )
        if use_parallel:
            return await self._execute_parallel(
                calls, tool_registry, ctx, session,
            )
        return await self._execute_sequential(
            calls, tool_registry, ctx, session,
        )

    async def _execute_sequential(
        self, calls, tool_registry, ctx, session,
    ):
        results = []
        for tc in calls:
            session.transcript.append(
                ToolCallRequestMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[tc],
                )
            )
            outcome = await self._execute_one(
                tc, tool_registry, ctx,
            )
            session.transcript.append(
                ToolCallResultMessage(
                    role=MessageRole.TOOL,
                    content=outcome.output,
                    tool_call_id=tc.id,
                )
            )
            results.append((tc, outcome))
        return results

    async def _execute_parallel(
        self, calls, tool_registry, ctx, session,
    ):
        # Record all tool-call requests up front
        for tc in calls:
            session.transcript.append(
                ToolCallRequestMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[tc],
                )
            )
        # Execute sequentially, recording results
        results: list[tuple[ToolCall, _ToolOutcome]] = []
        for tc in calls:
            outcome = await self._execute_one(
                tc, tool_registry, ctx,
            )
            session.transcript.append(
                ToolCallResultMessage(
                    role=MessageRole.TOOL,
                    content=outcome.output,
                    tool_call_id=tc.id,
                )
            )
            results.append((tc, outcome))
        return results

    async def _execute_one(
        self,
        tc: ToolCall,
        tool_registry: dict,
        ctx: Context,
    ) -> _ToolOutcome:
        tool_obj = tool_registry.get(tc.name)
        if tool_obj is None:
            logger.warning(f"Tool not found: {tc.name}")
            return _ToolOutcome(
                output=(
                    f"Error: tool '{tc.name}' not found"
                ),
                is_error=True,
            )

        try:
            params = json.loads(tc.arguments)
        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid JSON in arguments "
                f"for {tc.name}: {e}"
            )
            return _ToolOutcome(
                output=f"Error: invalid arguments — {e}",
                is_error=True,
            )

        logger.info(f"Calling {tc.name} with {params}")
        if (
            "context"
            in inspect.signature(tool_obj.func).parameters
        ):
            params["context"] = ctx

        async with tool_span(tc.name, tc.id) as t_span:
            try:
                result = await tool_obj(**params)
            except LLMRecoverableError as e:
                record_error(t_span, e)
                logger.info(
                    f"Tool {tc.name} "
                    f"requested retry: {e}"
                )
                return _ToolOutcome(
                    output=str(e), is_error=False,
                )
            except Exception as e:
                record_error(t_span, e)
                logger.error(
                    f"Tool {tc.name} raised: {e}"
                )
                return _ToolOutcome(
                    output=(
                        f"Error calling {tc.name}: {e}"
                    ),
                    is_error=True,
                )

        output_str = (
            json.dumps(result.output)
            if not isinstance(result.output, str)
            else result.output
        )
        return _ToolOutcome(
            output=output_str, is_error=False,
        )
