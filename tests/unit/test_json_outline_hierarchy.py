import json

from src.langchain.lc_outline_converter import parse_json_outline


def test_parse_json_outline_nested_hierarchy():
    outline = {
        "title": "Sample Book",
        "chapters": [
            {
                "number": 1,
                "title": "Chapter One",
                "sections": [
                    {
                        "letter": "A",
                        "title": "Section A",
                        "subsections": [
                            {"number": 1, "title": "Subsection 1"},
                            {"number": 2, "title": "Subsection 2"}
                        ]
                    },
                    {
                        "letter": "B",
                        "title": "Section B"
                    }
                ]
            },
            {
                "number": 2,
                "title": "Chapter Two",
                "sections": [
                    {
                        "letter": "A",
                        "title": "Another Section"
                    }
                ]
            }
        ]
    }

    sections, metadata = parse_json_outline(json.dumps(outline))

    expected_ids = ["1", "1A", "1A1", "1A2", "1B", "2", "2A"]
    expected_levels = [2, 3, 4, 4, 3, 2, 3]
    expected_parents = [None, "1", "1A", "1A", "1", None, "2"]

    assert [s.id for s in sections] == expected_ids
    assert [s.level for s in sections] == expected_levels
    assert [s.parent_id for s in sections] == expected_parents
    assert metadata.title == "Sample Book"
