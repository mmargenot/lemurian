import inspect
import json
import logging
from dataclasses import dataclass

from lemurian.agent import Agent
from lemurian.context import Context
from lemurian.message import Message, MessageRole, ToolCallRequestMessage, ToolCallResultMessage
from lemurian.session import Session
from lemurian.state import State
from lemurian.tools import HandoffResult, LLMRecoverableError

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """The result of a single Runner.run() invocation.

    Args:
        last_message: The final message appended to the transcript.
        agent_name: Name of the agent that was executed.
        hand_off: Set if a tool returned a HandoffResult, None otherwise.
    """

    last_message: Message
    agent_name: str
    hand_off: HandoffResult | None = None


class Runner:
    """Executes an agent's tool-calling loop.

    The Runner is the only component that reads or writes the session
    transcript. It injects the system prompt at call time (never storing
    it in the transcript), dispatches tool calls, and detects handoffs.

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
    ) -> RunResult:
        """Run the agent loop until a final response or handoff.

        Builds messages from ``session.transcript[context_start:]``
        with the system prompt prepended. Executes tool calls,
        appends results to the transcript, and returns when the
        provider produces a text response or a tool triggers a handoff.

        Args:
            agent: The agent to execute.
            session: The session containing the conversation transcript.
            state: The application state passed to tools via Context.
            context_start: Transcript index to start reading from.
                Used by Swarm for fresh-context handoffs.

        Returns:
            A RunResult with the final message and optional handoff.
        """
        ctx = Context(session=session, state=state, agent=agent)
        tool_registry = agent.tool_registry
        tool_schemas = [t.model_dump() for t in tool_registry.values()]

        for _ in range(self.max_turns):
            # Build messages: system prompt + transcript window
            transcript_window = session.transcript[context_start:]
            messages = [
                {"role": "system", "content": agent.system_prompt},
                *[m.model_dump() for m in transcript_window],
            ]

            response = await agent.provider.complete(
                model=agent.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
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
                    hand_off=None,
                )

            # Process tool calls
            for tool_call in response.tool_calls:
                if tool_call.type != "function":
                    continue

                # Append the tool call request to transcript
                session.transcript.append(
                    ToolCallRequestMessage(
                        role=MessageRole.ASSISTANT,
                        tool_calls=[tool_call],
                    )
                )

                func_name = tool_call.function.name
                call_id = tool_call.id

                # Look up in resolved tool registry
                tool_obj = tool_registry.get(func_name)
                if tool_obj is None:
                    logger.warning(f"Tool not found: {func_name}")
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=f"Error: tool '{func_name}' not found",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                # Parse arguments and inject context if needed
                # TODO: Add self-healing for malformed tool calls — use a
                # secondary model to repair the JSON or re-map arguments
                # before falling back to the error path.
                try:
                    params = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON in arguments for {func_name}: {e}"
                    )
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=f"Error: invalid arguments — {e}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                logger.info(f"Calling {func_name} with {params}")

                if "context" in inspect.signature(tool_obj.func).parameters:
                    params["context"] = ctx

                try:
                    result = await tool_obj(**params)
                except LLMRecoverableError as e:
                    logger.info(
                        f"Tool {func_name} requested retry: {e}"
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
                    logger.error(f"Tool {func_name} raised: {e}")
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=f"Error calling {func_name}: {e}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                # Check for handoff
                if isinstance(result.output, HandoffResult):
                    session.transcript.append(
                        ToolCallResultMessage(
                            role=MessageRole.TOOL,
                            content=f"Transferring to {result.output.target_agent}",
                            tool_call_id=call_id,
                        )
                    )
                    return RunResult(
                        last_message=session.transcript[-1],
                        agent_name=agent.name,
                        hand_off=result.output,
                    )

                # Normal tool result
                output_str = json.dumps(result.output) if not isinstance(result.output, str) else result.output
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
            content="Maximum turns reached. Please try again.",
        )
        session.transcript.append(timeout_msg)
        return RunResult(
            last_message=timeout_msg,
            agent_name=agent.name,
            hand_off=None,
        )
