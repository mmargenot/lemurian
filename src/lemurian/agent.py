from abc import abstractmethod
import json
import logging
from openai import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    InternalServerError
)

from lemurian.tools import Tool
from lemurian.session import Session
from lemurian.message import (
    Message,
    MessageRole,
    ToolCallRequestMessage,
    ToolCallResultMessage
)
from lemurian.provider import ModelProvider


logger = logging.getLogger(__name__)



class Agent:
    """
    Core agent class for `lemurian`. Subclass `Agent` and implement
    `get_all_tools` in order to create a new agent, then `respond` should work
    automatically in your agent loop.

    Args:
        model: String representing the model name.
        provider: Model provider object, can be any of the provided objects or
            one custom for your architecture.
        system_prompt: System prompt that this agent should use to operate.
    """
    # TODO: validate model against provider
    def __init__(self, model: str, provider: ModelProvider, system_prompt: str):
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.tool_registry = {t.name: t for t in self.get_all_tools()}

    @abstractmethod
    def get_all_tools(self) -> list[Tool]:
        """
        This function must be defined for all agents. It should return a list
        of functions that have been decorated with `Tool`.
        """
        pass

    # TODO: separate transcript to be a sort of message queue outside of the
    #     agent. fetch it based on session id
    async def respond(self, session: Session) -> Message:
        try:
            response = await self._respond(session)
        except (AuthenticationError, RateLimitError, APITimeoutError,
                BadRequestError, InternalServerError):
            response = Message(
                role=MessageRole.ASSISTANT,
                content="Something went wrong. Try asking your question again"
            )
        except KeyboardInterrupt:
            response = Message(
                role=MessageRole.ASSISTANT,
                content="Farewell!"
            )

        return response

    async def _respond(self, session: Session) -> Message:
        """
        Most recent Message (last) is what the user sent.
        """
        all_tools = self.get_all_tools()
        while True: 
            response = await self.provider.complete(
                model=self.model,
                all_tools=all_tools,
                messages=session.transcript
            )

            logger.debug('No tool calls found')
            if not response.tool_calls:
                session.transcript.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content
                    )
                )
                return session.transcript[-1]

            for item in response.tool_calls:
                logger.debug('Making a tool call')
                if item.type == "function":
                    session.transcript.append(
                        ToolCallRequestMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[item]
                        )
                    )
                    logging.info(f'Calling {item.function.name}')
                    params = json.loads(item.function.arguments)
                    logging.info(f'Parameters: {params}')
                    tool_result = self.tool_registry[item.function.name](
                        session=session,
                        **params
                    )
                    tool_message = ToolCallResultMessage(
                        role=MessageRole.TOOL,
                        content=tool_result.output,
                        tool_call_id=item.id
                    )
                    session.transcript.append(tool_message)
        return session.transcript[-1]
