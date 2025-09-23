from pathlib import Path

from typer.testing import CliRunner

from src.cli import multi_agent


class DummyRegistry:
    def __init__(self):
        self.registered_tool = None
        self.mcp_url = None

    def register(self, tool):
        self.registered_tool = tool

    async def register_mcp_server(self, url: str):
        self.mcp_url = url


def test_cli_invokes_agent_with_tools(monkeypatch):
    registry = DummyRegistry()
    monkeypatch.setattr(multi_agent, "ToolRegistry", lambda: registry)

    def fake_create_tool(key: str, index_dir: Path | None = None):
        fake_create_tool.called_with = (key, index_dir)
        return object()

    fake_create_tool.called_with = None
    monkeypatch.setattr(multi_agent, "create_rag_retrieve_tool", fake_create_tool)

    mock_llm = object()

    class FakeFactory:
        def create_llm(self):
            return ("backend", mock_llm)

    monkeypatch.setattr(multi_agent, "LLMFactory", lambda: FakeFactory())

    def fake_run_agent(llm, reg, question):
        assert llm is mock_llm
        assert reg is registry
        assert question == "where?"
        return "done"

    monkeypatch.setattr(multi_agent, "run_agent", fake_run_agent)

    runner = CliRunner()
    result = runner.invoke(multi_agent.app, ["where?", "--key", "paper", "--mcp", "server"])

    assert result.exit_code == 0
    assert "done" in result.stdout
    assert registry.registered_tool is not None
    assert registry.mcp_url == "server"
    assert fake_create_tool.called_with == (
        "paper",
        multi_agent.DEFAULT_INDEX_DIR,
    )


def test_cli_passes_custom_index(monkeypatch, tmp_path: Path):
    registry = DummyRegistry()
    monkeypatch.setattr(multi_agent, "ToolRegistry", lambda: registry)

    captured: dict[str, object] = {}

    def fake_create_tool(key: str, index_dir: Path | None = None):
        captured["args"] = (key, index_dir)
        return object()

    monkeypatch.setattr(multi_agent, "create_rag_retrieve_tool", fake_create_tool)

    mock_llm = object()

    class FakeFactory:
        def create_llm(self):
            return ("backend", mock_llm)

    monkeypatch.setattr(multi_agent, "LLMFactory", lambda: FakeFactory())
    monkeypatch.setattr(multi_agent, "run_agent", lambda llm, reg, question: "done")

    runner = CliRunner()
    index_dir = tmp_path / "indexes"
    result = runner.invoke(
        multi_agent.app,
        ["where?", "--key", "paper", "--index", str(index_dir)],
    )

    assert result.exit_code == 0
    assert captured["args"] == ("paper", index_dir)
