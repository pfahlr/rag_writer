import subprocess
import textwrap
from pathlib import Path

from run_rag_verification import resolve_script, build_question_command

def test_resolve_script_finds_src_cli(tmp_path: Path):
    repo = tmp_path
    cli_dir = repo / "src" / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "multi_agent.py").write_text("print('ok')\n")
    script = resolve_script(None, "multi_agent.py", repo_root=repo)
    assert script.name == "multi_agent.py"
    assert script.parent == cli_dir

def test_build_question_command_positional(tmp_path: Path):
    # Fake CLI with positional prompt (no --question in --help)
    cli = tmp_path / "positional_cli.py"
    cli.write_text(textwrap.dedent("""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("prompt")
        a = p.parse_args()
        print(a.prompt)
    """))
    argv = build_question_command(cli, "hello world", [])
    out = subprocess.run(argv, capture_output=True, text=True)
    assert out.stdout.strip() == "hello world"
    assert "unrecognized arguments" not in (out.stderr or "")

def test_build_question_command_flag(tmp_path: Path):
    # Fake CLI that advertises --question
    cli = tmp_path / "flag_cli.py"
    cli.write_text(textwrap.dedent("""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("-q","--question")
        a = p.parse_args()
        print(a.question)
    """))
    argv = build_question_command(cli, "hello world", [])
    out = subprocess.run(argv, capture_output=True, text=True)
    assert out.stdout.strip() == "hello world"
