import json

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.tool import Tool, ToolSpec, ToolRegistry
from src.tool.prompts.tool_prompt import generate_tool_prompt


def test_generate_tool_prompt_works_with_chat_prompt_template():
    def add(a: int, b: int) -> dict:
        return {"result": a + b}

    spec = ToolSpec(
        name="adder",
        description="add numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "integer"}},
            "required": ["result"],
        },
    )
    registry = ToolRegistry()
    registry.register(Tool(spec, add))

    prompt_text = generate_tool_prompt(registry)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                prompt_text, template_format="jinja2"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    messages = prompt.format_messages(input="hello")
    assert '{"tool": "<tool_name>"' in messages[0].content
