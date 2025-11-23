"""Tests for the CLI interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_import():
    """Test that CLI can be imported (if dependencies are available)."""
    try:
        from src.cli.main import app, console
        assert app is not None
        assert console is not None
        print("[OK] CLI module imports successfully")
        return True
    except ImportError as e:
        if "openai" in str(e):
            print("[INFO] CLI module requires 'openai' package")
            print("       Install with: pip install openai")
            return False
        raise


def test_cli_structure():
    """Test CLI structure without importing."""
    cli_file = Path(__file__).parent.parent / "src" / "cli" / "main.py"
    
    assert cli_file.exists(), "CLI file should exist"
    
    content = cli_file.read_text()
    
    # Check for required components
    assert "@app.command()" in content, "Should have app.command decorator"
    assert "def main(" in content, "Should have main function"
    assert "--kernel" in content or "kernel: str" in content, "Should have kernel argument/option"
    assert "--profile" in content, "Should have --profile option"
    assert "--save-report" in content, "Should have --save-report option"
    assert "analyze_cuda_file_path" in content, "Should call LLM agent"
    
    print("[OK] CLI structure is correct")


if __name__ == "__main__":
    test_cli_structure()
    print()
    test_cli_import()
    print("\nCLI structure tests passed!")

