"""Example usage of the LLM agent for CUDA analysis."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.llm_agent import analyze_and_optimize_cuda, analyze_cuda_file_path


def example_analyze_from_code():
    """Example: Analyze CUDA code from string."""
    code = """
#include <cuda_runtime.h>

__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
"""
    
    print("=" * 60)
    print("Example: Analyzing CUDA code from string")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[WARNING] OPENAI_API_KEY not set. Set it to use the LLM agent.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        result = analyze_and_optimize_cuda(
            code=code,
            enable_profiling=False,  # Set to True if you have GPU
            mock_mode=True
        )
        print("\n" + result)
    except Exception as e:
        print(f"\n[ERROR] {e}")


def example_analyze_from_file():
    """Example: Analyze CUDA file from path."""
    file_path = "src/cuda/example_kernels/saxpy.cu"
    
    print("=" * 60)
    print(f"Example: Analyzing CUDA file: {file_path}")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[WARNING] OPENAI_API_KEY not set. Set it to use the LLM agent.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        result = analyze_cuda_file_path(
            file_path=file_path,
            enable_profiling=False,
            mock_mode=True
        )
        print("\n" + result)
    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "file":
        example_analyze_from_file()
    else:
        example_analyze_from_code()

