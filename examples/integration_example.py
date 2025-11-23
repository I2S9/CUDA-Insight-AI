"""Example of integrating runner, profiler, and static analyzer."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cuda.runner import CUDARunner
from src.analysis.static_analyzer import analyze_cuda_file
from src.profiling.profiler_wrapper import profile_kernel


def analyze_kernel_complete(cu_file: str, mock_mode: bool = True):
    """Complete analysis pipeline: static analysis + profiling.
    
    Args:
        cu_file: Path to CUDA kernel file.
        mock_mode: Use mock mode for runner and profiler.
    """
    print(f"Analyzing kernel: {cu_file}\n")
    
    # 1. Static analysis
    print("=" * 60)
    print("STATIC ANALYSIS")
    print("=" * 60)
    static_result = analyze_cuda_file(cu_file)
    print(json.dumps(static_result, indent=2))
    print()
    
    # 2. Run kernel (compile and execute)
    print("=" * 60)
    print("KERNEL EXECUTION")
    print("=" * 60)
    runner = CUDARunner(mock_mode=mock_mode)
    execution_time, metadata = runner.run_kernel(
        Path(cu_file),
        compile_kernel=not mock_mode  # Skip compilation in mock mode
    )
    print(f"Execution time: {metadata['execution_time_ms']:.3f} ms")
    print()
    
    # 3. Profiling (detailed metrics)
    print("=" * 60)
    print("PROFILING")
    print("=" * 60)
    kernel_info = {
        "kernel_file": cu_file,
        "kernel_name": static_result["kernels"][0]["name"],
        **metadata
    }
    
    profile_result = profile_kernel(
        kernel_info=kernel_info,
        function_name=f"test_{static_result['kernels'][0]['name']}",
        iterations=5,
        mock_mode=mock_mode
    )
    print(json.dumps(profile_result, indent=2))
    print()
    
    # 4. Combined report
    print("=" * 60)
    print("COMBINED REPORT")
    print("=" * 60)
    combined = {
        "kernel": static_result["kernels"][0]["name"],
        "static_analysis": {
            "indexing_pattern": static_result["kernels"][0]["thread_indexing_pattern"],
            "memory_pattern": static_result["kernels"][0]["memory_access_pattern"],
            "warp_divergence": static_result["kernels"][0]["potential_warp_divergence"],
        },
        "execution": {
            "time_ms": metadata["execution_time_ms"],
        },
        "profiling": {
            "mean_time_ms": profile_result["mean_time_ms"],
            "std_dev_ms": profile_result["std_dev_ms"],
        }
    }
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        kernel_file = sys.argv[1]
    else:
        kernel_file = "src/cuda/example_kernels/saxpy.cu"
    
    mock = "--real" not in sys.argv
    analyze_kernel_complete(kernel_file, mock_mode=mock)

