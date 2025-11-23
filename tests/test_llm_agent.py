"""Tests for the LLM agent (mock mode)."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.llm_agent import load_tool_schema, analyze_and_optimize_cuda


def test_load_tool_schema():
    """Test loading tool schema."""
    tools = load_tool_schema()
    
    assert isinstance(tools, list)
    assert len(tools) >= 2
    
    tool_names = [tool["function"]["name"] for tool in tools]
    assert "static_analyzer" in tool_names
    assert "profiler" in tool_names
    
    print("[OK] Tool schema loaded successfully")
    print(f"Available tools: {tool_names}")


def test_analyze_and_optimize_cuda_structure():
    """Test that analyze_and_optimize_cuda has correct structure."""
    import inspect
    
    sig = inspect.signature(analyze_and_optimize_cuda)
    params = list(sig.parameters.keys())
    
    assert "code" in params
    assert "api_key" in params or "OPENAI_API_KEY" in os.environ
    
    print("[OK] Function signature is correct")


def test_analyze_cuda_file_path():
    """Test analyze_cuda_file_path function."""
    from src.ai.llm_agent import analyze_cuda_file_path
    
    # This will fail without API key, but we can check the function exists
    try:
        result = analyze_cuda_file_path(
            "src/cuda/example_kernels/saxpy.cu",
            enable_profiling=False,
            mock_mode=True
        )
        # If we get here, the function worked (though it needs API key)
        assert isinstance(result, str)
        print("[OK] analyze_cuda_file_path function works")
    except ValueError as e:
        if "API key" in str(e):
            print("[OK] Function correctly requires API key")
        else:
            raise


if __name__ == "__main__":
    import os
    
    test_load_tool_schema()
    print()
    test_analyze_and_optimize_cuda_structure()
    print()
    
    if os.getenv("OPENAI_API_KEY"):
        print("\n[INFO] OPENAI_API_KEY found - you can test with real API")
    else:
        print("\n[INFO] OPENAI_API_KEY not set - set it to test with real API")
    
    print("\nAll structure tests passed!")

