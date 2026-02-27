import inspect
import json
import logging
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.context import Context
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


class Runner:
    """Executes an agent's tool-calling loop.

    The Runner reads the session transcript and appends assistant
    responses, tool-call requests, and tool results during its loop.
    It injects the system prompt at call time (never storing it in
    the transcript), dispatches tool calls, and classifies handoffs.

    Args:
        max_turns: Maximum number of provider round-trips before
            returning a timeout message.
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns

    async def run(
        self,
        agent: Agent,
        session: Session,
        state: State,
        context_start: int = 0,
        handoffs: list[Handoff] | None = None,
    ) -> RunResult:
        """Run the agent loop until a final response or handoff.

        Builds messages from ``session.transcript[context_start:]``
        with the system prompt prepended.  Executes tool calls,
        appends results to the transcript, and returns when the
        provider produces a text response or the model invokes a
        handoff tool.

        Args:
            agent: The agent to execute.
            session: The session containing the conversation
                transcript.
            state: The application state passed to tools via Context.
            context_start: Transcript index to start reading from.
                Used by Swarm for fresh-context handoffs.
            handoffs: Optional list of Handoff objects.  Their
                tool schemas are sent to the provider alongside
                regular tools.  When the model calls a handoff tool
                the Runner returns immediately without executing
                any function.

        Returns:
            A RunResult with the final message and optional handoff.
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
            for _ in range(self.max_turns):
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

                async with completion_span(
                    system_name, agent.model
                ) as c_span:
                    response = await agent.provider.complete(
                        model=agent.model,
                        messages=messages,
                        tools=(
                            tool_schemas
                            if tool_schemas
                            else None
                        ),
                    )
                    record_usage(
                        c_span,
                        response.usage,
                        response.response_model,
                    )

                # No tool calls — final assistant response
                if not response.tool_calls:
                    assistant_msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                    )
                    session.transcript.append(assistant_msg)
                    return RunResult(
                        last_message=assistant_msg,
                        agent_name=agent.name,
                    )

                # Process tool calls
                for tool_call in response.tool_calls:
                    if tool_call.type != "function":
                        continue

                    func_name = tool_call.function.name
                    call_id = tool_call.id

                    # ----- Classify: handoff or regular tool? -----
                    if func_name in handoff_map:
                        handoff_obj = handoff_map[func_name]
                        try:
                            args = json.loads(
                                tool_call.function.arguments
                            )
                        except json.JSONDecodeError:
                            args = {}
                        message = args.get("message", "")

                        session.transcript.append(
                            ToolCallRequestMessage(
                                role=MessageRole.ASSISTANT,
                                tool_calls=[tool_call],
                            )
                        )
                        session.transcript.append(
                            ToolCallResultMessage(
                                role=MessageRole.TOOL,
                                content=(
                                    "Transferring to "
                                    f"{handoff_obj.target_agent}"
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        return RunResult(
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

                    # ----- Regular tool execution -----

                    # Append the tool call request to transcript
                    session.transcript.append(
                        ToolCallRequestMessage(
                            role=MessageRole.ASSISTANT,
                            tool_calls=[tool_call],
                        )
                    )

                    # Look up in resolved tool registry
                    tool_obj = tool_registry.get(func_name)
                    if tool_obj is None:
                        logger.warning(
                            f"Tool not found: {func_name}"
                        )
                        session.transcript.append(
                            ToolCallResultMessage(
                                role=MessageRole.TOOL,
                                content=(
                                    "Error: tool "
                                    f"'{func_name}' "
                                    "not found"
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        continue

                    # Parse arguments and inject context
                    # TODO: Add self-healing for malformed
                    # tool calls — use a secondary model to
                    # repair the JSON or re-map arguments
                    # before falling back to the error path.
                    try:
                        params = json.loads(
                            tool_call.function.arguments
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Invalid JSON in arguments "
                            f"for {func_name}: {e}"
                        )
                        session.transcript.append(
                            ToolCallResultMessage(
                                role=MessageRole.TOOL,
                                content=(
                                    "Error: invalid "
                                    f"arguments — {e}"
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        continue

                    logger.info(
                        f"Calling {func_name} with {params}"
                    )

                    if (
                        "context"
                        in inspect.signature(
                            tool_obj.func
                        ).parameters
                    ):
                        params["context"] = ctx

                    async with tool_span(
                        func_name, call_id
                    ) as t_span:
                        try:
                            result = await tool_obj(
                                **params
                            )
                        except LLMRecoverableError as e:
                            record_error(t_span, e)
                            logger.info(
                                f"Tool {func_name} "
                                f"requested retry: {e}"
                            )
                            session.transcript.append(
                                ToolCallResultMessage(
                                    role=MessageRole.TOOL,
                                    content=str(e),
                                    tool_call_id=call_id,
                                )
                            )
                            continue
                        except Exception as e:
                            record_error(t_span, e)
                            logger.error(
                                "Tool "
                                f"{func_name} raised: {e}"
                            )
                            session.transcript.append(
                                ToolCallResultMessage(
                                    role=MessageRole.TOOL,
                                    content=(
                                        "Error calling "
                                        f"{func_name}: {e}"
                                    ),
                                    tool_call_id=call_id,
                                )
                            )
                            continue

                    # Normal tool result
                    output_str = (
                        json.dumps(result.output)
                        if not isinstance(
                            result.output, str
                        )
                        else result.output
                    )
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=output_str,
                            tool_call_id=call_id,
                        )
                    )

            # Max turns exceeded
            timeout_msg = Message(
                role=MessageRole.ASSISTANT,
                content=(
                    "Maximum turns reached. "
                    "Please try again."
                ),
            )
            session.transcript.append(timeout_msg)
            return RunResult(
                last_message=timeout_msg,
                agent_name=agent.name,
            )
