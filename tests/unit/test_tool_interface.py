import pytest
from langchain_core.documents import Document

from src.tool import Tool, ToolSpec, ToolRegistry, create_rag_retrieve_tool
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
