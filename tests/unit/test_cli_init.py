"""
Unit tests for CLI package initialization.
"""

from src.cli import __version__


class TestCLIPackage:
    """Test CLI package initialization."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        # Should be in format like "1.0.0" or "1.2.3-alpha"
        version_parts = __version__.split('.')
        assert len(version_parts) >= 2  # At least major.minor

        # First two parts should be numeric
        assert version_parts[0].isdigit()
        assert version_parts[1].isdigit()