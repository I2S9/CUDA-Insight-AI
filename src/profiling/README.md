# CUDA Profiler

C++ profiler for measuring CUDA kernel performance with Python wrapper.

## Compilation

### Prerequisites
- CUDA Toolkit installed
- CMake 3.10 or higher
- C++ compiler with C++17 support

### Build Instructions

```bash
cd src/profiling
mkdir build
cd build
cmake ..
make  # or cmake --build . on Windows
```

The profiler executable will be in `build/bin/profiler` (or `build/bin/profiler.exe` on Windows).

## Usage

### C++ Profiler (Direct)

```bash
./profiler <library_path> <function_name> [iterations]
```

Example:
```bash
./profiler ../example_kernels/saxpy.so test_saxpy 10
```

### Python Wrapper

```python
from src.profiling.profiler_wrapper import profile_kernel

# With kernel info from runner
kernel_info = {
    "compiled_file": "path/to/kernel.so",
    "kernel_name": "saxpy"
}

result = profile_kernel(
    kernel_info=kernel_info,
    function_name="test_saxpy",
    iterations=10
)

print(f"Mean execution time: {result['mean_time_ms']} ms")
```

### Mock Mode (for testing without GPU)

```python
result = profile_kernel(
    kernel_info=kernel_info,
    function_name="test_saxpy",
    iterations=10,
    mock_mode=True
)
```

## Output Format

The profiler returns a JSON dictionary with:

```json
{
  "mean_time_ms": 1.234,
  "median_time_ms": 1.230,
  "min_time_ms": 1.200,
  "max_time_ms": 1.250,
  "std_dev_ms": 0.015,
  "iterations": 10,
  "all_times_ms": [1.200, 1.230, ...],
  "mock_mode": false
}
```

## Integration

The profiler can be integrated with the CUDA runner:

```python
from src.cuda.runner import CUDARunner
from src.profiling.profiler_wrapper import profile_kernel

runner = CUDARunner()
execution_time, metadata = runner.run_kernel(cu_file)

# Then profile for more detailed metrics
profile_result = profile_kernel(
    kernel_info=metadata,
    function_name=f"test_{metadata['kernel_name']}",
    iterations=10
)
```

