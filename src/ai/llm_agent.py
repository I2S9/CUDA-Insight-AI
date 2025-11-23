"""LLM agent with tool-calling for CUDA kernel analysis and optimization."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import tempfile

from openai import OpenAI
from rich.console import Console

from src.analysis.static_analyzer import analyze_cuda_file
from src.profiling.profiler_wrapper import profile_kernel
from src.cuda.runner import CUDARunner

console = Console()

# Load tool schema
TOOL_SCHEMA_PATH = Path(__file__).parent / "tool_calling_schema.json"


def load_tool_schema() -> List[Dict]:
    """Load tool calling schema from JSON file.
    
    Returns:
        List of tool definitions for OpenAI function calling.
    """
    with open(TOOL_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema["tools"]


def analyze_and_optimize_cuda(
    code: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    enable_profiling: bool = False,
    mock_mode: bool = False
) -> str:
    """Analyze CUDA code and provide optimization recommendations using LLM with tool-calling.
    
    Args:
        code: CUDA source code as a string.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        model: OpenAI model to use (default: gpt-4o-mini).
        enable_profiling: Whether to enable runtime profiling (requires compiled kernel).
        mock_mode: Use mock mode for profiling (no GPU required).
        
    Returns:
        Structured text report with:
        - Summary
        - Issues
        - Recommendations
        - Possible optimized snippet
    """
    # Initialize OpenAI client
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    client = OpenAI(api_key=api_key)
    
    # Save code to temporary file for analysis
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_file_path = tmp_file.name
    
    try:
        # Step 1: Static analysis
        console.print("[blue]Running static analysis...[/blue]")
        static_result = analyze_cuda_file(tmp_file_path)
        
        # Step 2: Profiling (if enabled)
        profile_result = None
        if enable_profiling:
            console.print("[blue]Running profiler...[/blue]")
            try:
                runner = CUDARunner(mock_mode=mock_mode)
                execution_time, metadata = runner.run_kernel(
                    Path(tmp_file_path),
                    compile_kernel=not mock_mode
                )
                
                kernel_info = {
                    "kernel_file": tmp_file_path,
                    "kernel_name": static_result["kernels"][0]["name"],
                    **metadata
                }
                
                profile_result = profile_kernel(
                    kernel_info=kernel_info,
                    function_name=f"test_{static_result['kernels'][0]['name']}",
                    iterations=5,
                    mock_mode=mock_mode
                )
            except Exception as e:
                console.print(f"[yellow]Profiling failed: {e}[/yellow]")
                profile_result = None
        
        # Step 3: Prepare context for LLM
        analysis_context = {
            "source_code": code,
            "static_analysis": static_result,
        }
        
        if profile_result:
            analysis_context["profiling"] = profile_result
        
        # Step 4: Call LLM with analysis results
        # System message
        system_message = """You are an expert CUDA performance optimization specialist. 
Your task is to analyze CUDA kernels and provide detailed, actionable recommendations.

Analyze the provided CUDA code along with static analysis and profiling results, and generate a comprehensive report with the following structure:

## Summary
Brief overview of the kernel(s) and their purpose, including what the code does and its main characteristics.

## Issues
List of detected performance issues, potential bugs, or suboptimal patterns. Be specific and reference the analysis data.

## Recommendations
Specific, actionable optimization suggestions. Prioritize the most impactful improvements. Include explanations of why each recommendation would help.

## Possible Optimized Snippet
If applicable, provide an improved version of the code with comments explaining the optimizations.

Be technical, precise, and focus on GPU performance optimization. Use the analysis data to support your recommendations."""

        # User message with context
        user_message = f"""Analyze the following CUDA code and provide optimization recommendations.

CUDA Source Code:
```cuda
{code}
```

Static Analysis Results:
```json
{json.dumps(static_result, indent=2)}
```
"""
        
        if profile_result:
            user_message += f"""
Profiling Results:
```json
{json.dumps(profile_result, indent=2)}
```
"""
        
        user_message += """
Please analyze this code and provide a comprehensive report with Summary, Issues, Recommendations, and a Possible optimized snippet if applicable."""

        # Make API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        console.print("[blue]Calling LLM for analysis...[/blue]")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3  # Lower temperature for more consistent, technical output
        )
        
        # Extract final response
        final_response = response.choices[0].message.content
        
        return final_response
        
    finally:
        # Cleanup temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass


def analyze_cuda_file_path(
    file_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    enable_profiling: bool = False,
    mock_mode: bool = False
) -> str:
    """Analyze CUDA file from path and provide optimization recommendations.
    
    Args:
        file_path: Path to the CUDA .cu file.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        model: OpenAI model to use.
        enable_profiling: Whether to enable runtime profiling.
        mock_mode: Use mock mode for profiling.
        
    Returns:
        Structured text report.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    return analyze_and_optimize_cuda(
        code=code,
        api_key=api_key,
        model=model,
        enable_profiling=enable_profiling,
        mock_mode=mock_mode
    )

