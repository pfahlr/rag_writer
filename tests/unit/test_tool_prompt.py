from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.tool import Tool, ToolSpec, ToolRegistry
from src.config.content.prompts.tool_prompt import generate_tool_prompt


def _build_registry() -> ToolRegistry:
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
    return registry


def test_generate_tool_prompt_works_with_chat_prompt_template():
    registry = _build_registry()
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


def test_generate_tool_prompt_works_with_mcp_descriptors():
    registry = _build_registry()
    descriptors = registry.describe()
    # Simulate the output shape of ``mcp_client.fetch_tools`` which uses
    # camelCase for the schema key.
    mcp_descriptors = [
        {
            "name": d["name"],
            "description": d["description"],
            "inputSchema": d["input_schema"],
        }
        for d in descriptors
    ]

    prompt_text = generate_tool_prompt(mcp_descriptors)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                prompt_text, template_format="jinja2"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    messages = prompt.format_messages(input="hello")
    assert "- adder: add numbers" in messages[0].content
    assert '{"tool": "<tool_name>"' in messages[0].content
