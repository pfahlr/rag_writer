def run(text: str) -> dict:
    """Simple markdown echo tool returning text wrapped in <p> tags."""
    return {"html": f"<p>{text}</p>"}
