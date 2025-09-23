from langchain_core.documents import Document

from src.tool import rag_tool


def test_create_rag_tool_respects_custom_index(monkeypatch, tmp_path):
    custom_index = tmp_path / "indexes"
    custom_index.mkdir()

    class DummyRetriever:
        def get_relevant_documents(self, query: str):
            return [Document(page_content="answer", metadata={"source": "s"})]

    class DummyFactory:
        def __init__(self, **kwargs):
            assert kwargs.get("storage_dir") == custom_index

        def create_vector_retriever(self, config):
            return DummyRetriever()

        def create_bm25_retriever(self, config):
            return DummyRetriever()

        def create_hybrid_retriever(self, config):
            return DummyRetriever()

    monkeypatch.setattr(rag_tool, "RetrieverFactory", DummyFactory)

    tool = rag_tool.create_rag_retrieve_tool("paper", index_dir=custom_index)

    result = tool.run(query="what?", retriever_profile="vector", rerank=False)

    assert result["docs"][0]["text"] == "answer"
