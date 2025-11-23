"""Tests for the profiler wrapper."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling.profiler_wrapper import profile_kernel, _profile_mock


def test_profile_mock():
    """Test mock profiling."""
    result = _profile_mock(iterations=5)
    
    assert "mean_time_ms" in result
    assert "median_time_ms" in result
    assert "min_time_ms" in result
    assert "max_time_ms" in result
    assert "std_dev_ms" in result
    assert "iterations" in result
    assert "all_times_ms" in result
    assert result["mock_mode"] is True
    assert len(result["all_times_ms"]) == 5
    
    print("[OK] Mock profiling test passed")
    print(json.dumps(result, indent=2))


def test_profile_kernel_mock_mode():
    """Test profile_kernel in mock mode."""
    kernel_info = {
        "kernel_file": "src/cuda/example_kernels/saxpy.cu",
        "compiled_file": "src/cuda/example_kernels/saxpy.so",
        "kernel_name": "saxpy"
    }
    
    result = profile_kernel(
        kernel_info=kernel_info,
        function_name="test_saxpy",
        iterations=5,
        mock_mode=True
    )
    
    assert result["mock_mode"] is True
    assert result["iterations"] == 5
    assert "mean_time_ms" in result
    
    print("[OK] profile_kernel mock mode test passed")
    print(json.dumps(result, indent=2))


def test_profile_kernel_with_binary_path():
    """Test profile_kernel with binary path."""
    result = profile_kernel(
        binary_path="dummy.so",
        function_name="test_kernel",
        iterations=3,
        mock_mode=True  # Will use mock since binary doesn't exist
    )
    
    # Should work in mock mode even with non-existent binary
    assert result["mock_mode"] is True
    
    print("[OK] profile_kernel with binary_path test passed")


if __name__ == "__main__":
    test_profile_mock()
    print()
    test_profile_kernel_mock_mode()
    print()
    test_profile_kernel_with_binary_path()
    print("\nAll profiler tests passed!")

