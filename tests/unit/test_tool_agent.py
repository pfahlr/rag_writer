import json

from src.tool import Tool, ToolSpec, ToolRegistry, run_agent


def test_tool_agent_executes_tool_and_returns_final_answer():
    """Agent should run the requested tool then surface the final answer."""

    calls = {"tool": 0}

    def add(a: int, b: int) -> dict:
        calls["tool"] += 1
        return {"result": a + b}

    spec = ToolSpec(
        name="adder",
        description="add numbers",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
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

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            if self.calls == 0:
                self.calls += 1
                return json.dumps({"tool": "adder", "args": {"a": 2, "b": 3}})
            return json.dumps({"final": "5"})

    answer = run_agent(DummyLLM(), registry, "what is 2+3?")

    assert calls["tool"] == 1
    assert answer == "5"
