# CUDA-Insight-AI

## Objective

>  CUDA-Insight-AI is a command-line tool that analyzes CUDA kernels by combining static analysis, optional runtime profiling, and an LLM agent with tool-calling. The goal is to help developers detect potential performance issues and receive optimization suggestions for GPU code (CUDA kernels).

## Why This Project Matters

- Helps developers understand GPU performance bottlenecks without deep CUDA expertise
- Provides AI-driven optimization guidance that combines static analysis and runtime metrics
- Bridges traditional developer tools and modern LLM agentic systems for code analysis
- Useful for GPU/AI engineering education and performance optimization workflows

## Tech Stack

- Python (CLI, orchestration)
- CUDA C/C++ (kernels)
- C++17 (profiler)
- OpenAI API / LLM function calling
- JSON-based tool-calling
- CMake (C++ build)

## Project Architecture

```
CUDA-Insight-AI/
├── src/
│   ├── ai/
│   │   ├── llm_agent.py                    # LLM agent with tool-calling
│   │   └── tool_calling_schema.json        # Tool schema for the agent
│   ├── analysis/
│   │   └── static_analyzer.py              # Static analyzer for CUDA kernels
│   ├── cli/
│   │   └── main.py                         # Command-line interface
│   ├── cuda/
│   │   ├── example_kernels/                # Example CUDA kernels
│   │   │   ├── saxpy.cu
│   │   │   ├── vector_add.cu
│   │   │   └── divergent_kernel.cu
│   │   └── runner.py                       # CUDA kernel runner
│   └── profiling/
│       ├── profiler.cpp                    # C++ profiler for runtime metrics
│       ├── profiler_wrapper.py             # Python wrapper for the profiler
│       └── CMakeLists.txt                  # Build configuration
├── tests/                                  # Unit tests
├── examples/                               # Usage examples
├── report/                                 # LaTeX report
└── requirements.txt                        # Python dependencies
```

## Analysis Pipeline

The analysis pipeline follows three main steps:

```
CUDA (.cu file)
        │
        ▼
Static Analyzer ────► JSON (analysis)
        │
        ▼
   Profiler (opt) ─► JSON (metrics)
        │
        ▼
    LLM Agent ─────► Final Report
```

### 1. Static Analysis

The static analyzer (`src/analysis/static_analyzer.py`) inspects the CUDA source file without executing it. It detects:

- Kernel definitions (`__global__` functions)
- Thread indexing patterns (threadIdx, blockIdx, blockDim)
- Simple patterns that may cause warp divergence
- Memory access patterns (e.g., a[i], a[i + stride])

The analyzer returns a JSON dictionary containing the extracted information, usable by the LLM agent.

### 2. Profiling (Optional)

The profiler (`src/profiling/profiler_wrapper.py`) measures runtime performance of the kernel when a compatible NVIDIA GPU and CUDA environment are available. It provides:

- Kernel execution time
- Other performance metrics if available

If no GPU is available, the profiler can operate in mock mode to allow testing of the rest of the pipeline.

### 3. LLM Agent with Tool-Calling

The LLM agent (`src/ai/llm_agent.py`) is responsible for:

- Calling tools (static analyzer and profiler)
- Interpreting JSON results
- Generating a human-readable analysis report

The agent uses tool-calling (e.g., OpenAI function calling) to orchestrate the analysis and produces a structured report including:

- Summary of detected kernels
- Identified issues (static analysis and profiling)
- Optimization suggestions
- Optional improved code

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA Toolkit (optional, required only for profiling)
- Compatible NVIDIA GPU (optional, required only for profiling)
- OpenAI API key (required for LLM agent, except in mock mode)

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

To use the LLM agent, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Command Examples

### Basic Analysis (without profiling)

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu
```

### Analysis with Profiling

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --profile
```

### Save Report to File

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --save-report report.txt
```

### Save as Markdown

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --save-report report.md
```

### Mock Mode (test without GPU/API)

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --mock
```

### Specify Different OpenAI Model

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --model gpt-4
```

### Use API Key from Command Line

```bash
python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --api-key your-api-key
```

## Example Kernel

Here is an example of a simple CUDA kernel (SAXPY):

```cuda
#include <cuda_runtime.h>

// SAXPY kernel: y = a * x + y
// Single-precision A times X Plus Y
__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

This kernel performs a SAXPY (Scalar Alpha X Plus Y) operation on vectors. The static analyzer will detect:

- Standard 1D indexing pattern
- Coalesced memory access (consecutive accesses)
- Simple bounds check (i < n) that does not cause significant divergence

## Limitations

- Profiling requires a compatible NVIDIA GPU and CUDA Toolkit installed
- The LLM agent requires a valid OpenAI API key (or mock mode for testing)
- Static analysis is limited to common patterns and may not detect all performance issues
- The profiler may require separate compilation of C++ code

## Testing

Run tests with pytest:

```bash
pytest .
```

For tests with coverage:

```bash
pytest --cov=src tests/
```

## License

See the LICENSE file for details.