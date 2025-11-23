"""Test the CUDA runner in mock mode."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cuda.runner import CUDARunner


def test_runner_mock_mode():
    """Test that the runner works in mock mode."""
    runner = CUDARunner(mock_mode=True)
    
    # Test with saxpy kernel
    saxpy_file = Path(__file__).parent.parent / "src" / "cuda" / "example_kernels" / "saxpy.cu"
    
    assert saxpy_file.exists(), f"Kernel file not found: {saxpy_file}"
    
    execution_time, metadata = runner.run_kernel(saxpy_file, compile_kernel=False)
    
    assert execution_time > 0, "Execution time should be positive"
    assert "execution_time_ms" in metadata, "Metadata should contain execution_time_ms"
    assert metadata["mock_mode"] is True, "Should be in mock mode"
    print(f"Mock execution successful: {metadata['execution_time_ms']:.3f} ms")


if __name__ == "__main__":
    test_runner_mock_mode()
    print("Test passed!")

