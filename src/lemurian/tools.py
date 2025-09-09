import inspect
import json
from pydantic import BaseModel, Field
from typing import Callable


class ToolCallResult(BaseModel):
    tool_name: str
    output: str


class Tool(BaseModel):
    # Define as fields but exclude from serialization
    tool_func: Callable = Field(exclude=True)
    name: str = Field(exclude=True)
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, tool_func):
        super().__init__(
            tool_func=tool_func,
            name=tool_func.__name__,
        )

    def model_dump(self, **kwargs):
        """Override to return the JSON schema instead of internal attributes"""
        return self.get_schema()

    def model_dump_json(self, **kwargs):
        """Override JSON serialization"""
        return json.dumps(self.get_schema())

    def normalize_to_json_type(self, python_type_str: str) -> str:
        type_mapping = {
            'str': 'string',
            'int': 'number',
            'float': 'number',
            'bool': 'boolean',
            'NoneType': 'null',
            'dict': 'object',
            'list': 'array',
            'tuple': 'array',  # closest equivalent
            'set': 'array',    # closest equivalent
        }
        return type_mapping.get(python_type_str, 'string')

    def parse_properties(self, func: Callable) -> dict[str, dict[str,  str]]:
        signature = inspect.signature(func)
        properties = {}

        # TODO: if it's an array, you need to add:
        # "type": "array",
        # "items": {
        #     "type": "string"
        # }
        for param_name, param in signature.parameters.items():
            # skip any parameters that are listed in the exclusion
            # this allows for "magic" state handling parameters in tools.
            if param_name in ["session"]:
                continue
            properties[param_name] = {
                "type": self.normalize_to_json_type(param.annotation.__name__),
                "description": ""
            }

        return properties

    def get_required_params(self, func: Callable) -> list[str]:
        signature = inspect.signature(func)
        return [
            name
            for name, param in signature.parameters.items()
            if param.default == inspect.Parameter.empty and name not in ["session"]
        ]

    def get_schema(self):
        properties = self.parse_properties(self.tool_func)
        required_params: list[str] = self.get_required_params(self.tool_func)

        json_schema = {
            "type": "function",
            "function": {
                "name": self.tool_func.__name__,
                "description": self.tool_func.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                },
            }
        }
        return json_schema

    # TODO: do something nicer than cast the output to string here
    def __call__(self, *args, **kwargs):
        # validate that it produces a ToolResult
        return ToolCallResult(
            tool_name=self.name,
            output=str(self.tool_func(*args, **kwargs))
        )
