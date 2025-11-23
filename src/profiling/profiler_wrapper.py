"""Python wrapper for the CUDA profiler."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import random
import time


def profile_kernel(
    binary_path: Optional[str] = None,
    kernel_info: Optional[Dict] = None,
    function_name: Optional[str] = None,
    iterations: int = 10,
    mock_mode: bool = False,
    profiler_path: Optional[str] = None
) -> Dict:
    """Profile a CUDA kernel and return performance metrics.
    
    Args:
        binary_path: Path to the compiled kernel library (.so or .dll).
        kernel_info: Dictionary with kernel information (alternative to binary_path).
                    Should contain 'compiled_file' or 'kernel_file'.
        function_name: Name of the kernel function to profile (e.g., 'test_saxpy').
                      If None, tries to auto-detect from kernel_info.
        iterations: Number of iterations to run for statistics.
        mock_mode: If True, return mock data without running profiler.
        profiler_path: Path to the profiler executable. If None, tries to find it.
        
    Returns:
        Dictionary containing profiling results:
        {
            "mean_time_ms": float,
            "median_time_ms": float,
            "min_time_ms": float,
            "max_time_ms": float,
            "std_dev_ms": float,
            "iterations": int,
            "all_times_ms": List[float],
            "mock_mode": bool (if mock)
        }
    """
    if mock_mode:
        return _profile_mock(iterations)
    
    # Determine binary path
    if binary_path is None:
        if kernel_info is None:
            raise ValueError("Either binary_path or kernel_info must be provided")
        
        binary_path = kernel_info.get("compiled_file")
        if binary_path is None:
            # Try to construct from kernel_file
            kernel_file = kernel_info.get("kernel_file")
            if kernel_file:
                kernel_path = Path(kernel_file)
                if sys.platform == "win32":
                    binary_path = str(kernel_path.parent / f"{kernel_path.stem}.dll")
                else:
                    binary_path = str(kernel_path.parent / f"{kernel_path.stem}.so")
            else:
                raise ValueError("Could not determine binary path from kernel_info")
    
    binary_path = Path(binary_path)
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    
    # Determine function name
    if function_name is None:
        if kernel_info:
            # Try to infer from kernel name
            kernel_name = kernel_info.get("kernel_name", "test")
            function_name = f"test_{kernel_name}"
        else:
            raise ValueError("function_name must be provided if kernel_info is not available")
    
    # Find profiler executable
    if profiler_path is None:
        profiler_path = _find_profiler()
    
    if profiler_path is None:
        raise RuntimeError(
            "Profiler executable not found. Please compile the profiler first:\n"
            "  cd src/profiling && mkdir build && cd build\n"
            "  cmake .. && make"
        )
    
    # Run profiler
    try:
        result = subprocess.run(
            [profiler_path, str(binary_path), function_name, str(iterations)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        
        # Parse JSON output
        output = result.stdout.strip()
        if output.startswith("{"):
            profile_data = json.loads(output)
            profile_data["mock_mode"] = False
            return profile_data
        else:
            # Try to parse error
            if "error" in output.lower():
                raise RuntimeError(f"Profiler error: {output}")
            raise RuntimeError(f"Unexpected profiler output: {output}")
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"Profiler execution failed: {error_msg}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse profiler output as JSON: {e}\nOutput: {result.stdout}")


def _find_profiler() -> Optional[str]:
    """Find the profiler executable.
    
    Returns:
        Path to profiler executable or None if not found.
    """
    # Check common locations
    base_path = Path(__file__).parent
    
    # Check build directory
    build_paths = [
        base_path / "build" / "bin" / "profiler",
        base_path / "build" / "profiler",
        base_path / "profiler",
    ]
    
    if sys.platform == "win32":
        build_paths = [p.with_suffix(".exe") for p in build_paths]
    
    for path in build_paths:
        if path.exists() and path.is_file():
            return str(path)
    
    return None


def _profile_mock(iterations: int = 10) -> Dict:
    """Generate mock profiling data for testing without GPU.
    
    Args:
        iterations: Number of mock iterations.
        
    Returns:
        Mock profiling data dictionary.
    """
    # Generate realistic-looking mock times (0.5-2.0 ms with some variance)
    base_time = random.uniform(0.8, 1.5)
    times = [base_time + random.uniform(-0.2, 0.2) for _ in range(iterations)]
    times = [max(0.1, t) for t in times]  # Ensure positive
    
    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2] if len(times) % 2 == 1 else (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2
    min_time = min(times)
    max_time = max(times)
    
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5
    
    return {
        "mean_time_ms": mean,
        "median_time_ms": median,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_dev_ms": std_dev,
        "iterations": iterations,
        "all_times_ms": times,
        "mock_mode": True,
    }


def compile_profiler(build_dir: Optional[str] = None) -> str:
    """Compile the profiler using CMake.
    
    Args:
        build_dir: Directory for build files. Defaults to src/profiling/build.
        
    Returns:
        Path to compiled profiler executable.
    """
    profiling_dir = Path(__file__).parent
    
    if build_dir is None:
        build_dir = profiling_dir / "build"
    else:
        build_dir = Path(build_dir)
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Run CMake
    subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        check=True
    )
    
    # Build
    subprocess.run(
        ["cmake", "--build", "."],
        cwd=build_dir,
        check=True
    )
    
    # Find executable
    if sys.platform == "win32":
        exe_name = "profiler.exe"
    else:
        exe_name = "profiler"
    
    profiler_path = build_dir / "bin" / exe_name
    if not profiler_path.exists():
        profiler_path = build_dir / exe_name
    
    if not profiler_path.exists():
        raise RuntimeError(f"Profiler compilation succeeded but executable not found at {profiler_path}")
    
    return str(profiler_path)

