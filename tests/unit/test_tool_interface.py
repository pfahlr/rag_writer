import json
import pytest
from langchain_core.documents import Document

from src.tool import Tool, ToolSpec, ToolRegistry, create_rag_retrieve_tool, run_agent
from src.core.retriever import RetrieverFactory


def test_tool_registry_basic():
    def add(a: int, b: int) -> dict:
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
    tool = Tool(spec, add)
    reg = ToolRegistry()
    reg.register(tool)
    out = reg.run("adder", a=2, b=3)
    assert out["result"] == 5

    desc = reg.describe()
    assert desc == [
        {
            "name": "adder",
            "description": "add numbers",
            "input_schema": spec.input_schema,
        }
    ]


class DummyRetriever:
    def get_relevant_documents(self, query: str):
        return [Document(page_content="answer", metadata={"source": "1", "title": "T"})]


def _dummy_create(self, config):
    return DummyRetriever()


def test_rag_tool(monkeypatch):
    monkeypatch.setattr(RetrieverFactory, "create_hybrid_retriever", _dummy_create)
    monkeypatch.setattr(RetrieverFactory, "create_vector_retriever", _dummy_create)
    monkeypatch.setattr(RetrieverFactory, "create_bm25_retriever", _dummy_create)

    tool = create_rag_retrieve_tool("key")
    out = tool.run(query="hello", k=1)
    assert out["docs"][0]["text"] == "answer"
    assert out["docs"][0]["source_id"] == "1"


def test_run_agent_basic():
    """Agent should call tool and return final answer."""

    def add(a: int, b: int) -> dict:
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
    reg = ToolRegistry()
    reg.register(Tool(spec, add))

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            if self.calls == 0:
                self.calls += 1
                return json.dumps({"tool": "adder", "args": {"a": 2, "b": 3}})
            return json.dumps({"final": "5"})

    answer = run_agent(DummyLLM(), reg, "what is 2+3?")
    assert answer == "5"


def test_run_agent_invalid_tool():
    """Unknown tool names should surface an error message."""

    reg = ToolRegistry()

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            if self.calls == 0:
                self.calls += 1
                return json.dumps({"tool": "missing", "args": {}})
            # Second call should receive the error message and finalize
            return json.dumps({"final": messages[-1]["content"]})

    answer = run_agent(DummyLLM(), reg, "hello")
    assert "Unknown tool" in answer


def test_run_agent_recovers_from_malformed_json():
    """Agent should continue after malformed JSON and return final answer."""

    reg = ToolRegistry()

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages):
            self.calls += 1
            if self.calls == 1:
                return "not-json"
            # Ensure the agent surfaced the JSON error from the previous step
            assert "Invalid JSON" in json.loads(messages[-1]["content"])["error"]
            return json.dumps({"final": "ok"})

    llm = DummyLLM()
    answer = run_agent(llm, reg, "hello")
    assert answer == "ok"
    assert llm.calls == 2

