import json
from src.cli.agent_runner import run_agent_with_retrieval
from src.tool import Tool, ToolSpec


def test_run_agent_with_retrieval_registers_tool_and_calls_agent(monkeypatch):
    """Ensure registry registers RAG tool and run_agent is invoked."""
    called = {}

    def fake_create_tool(key: str) -> Tool:
        called['key'] = key
        spec = ToolSpec(
            name="rag_retrieve",
            description="",
            input_schema={},
            output_schema={},
        )
        return Tool(spec, lambda **_: {})

    def fake_run_agent(llm, registry, question, max_iters=5, system_prompt=None):
        called['question'] = question
        called['tools'] = list(registry.list_tools())
        return "ANSWER"

    monkeypatch.setattr("src.cli.agent_runner.create_rag_retrieve_tool", fake_create_tool)
    monkeypatch.setattr("src.cli.agent_runner.run_agent", fake_run_agent)

    class DummyLLM:
        def __call__(self, messages):
            return json.dumps({"final": "done"})

    result = run_agent_with_retrieval("hi", DummyLLM(), "faiss_key")

    assert result == "ANSWER"
    assert called['key'] == "faiss_key"
    assert "rag_retrieve" in called['tools']
    assert called['question'] == "hi"
