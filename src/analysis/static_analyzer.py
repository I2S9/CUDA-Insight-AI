"""Static analyzer for CUDA kernel source files."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def analyze_cuda_file(path: str) -> dict:
    """Analyze a CUDA source file and extract kernel information.
    
    Args:
        path: Path to the .cu file to analyze.
        
    Returns:
        Dictionary containing analysis results with the following structure:
        {
            "file": str,
            "kernels": [
                {
                    "name": str,
                    "potential_warp_divergence": bool,
                    "memory_access_pattern": str,
                    "thread_indexing_pattern": str,
                    "indexing_formula": str,
                    "conditional_checks": List[str],
                    "memory_accesses": List[str]
                }
            ],
            "summary": {
                "total_kernels": int,
                "kernels_with_divergence": int,
                "kernels_with_coalesced_access": int
            }
        }
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not file_path.suffix == ".cu":
        raise ValueError(f"File must have .cu extension: {path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    kernels = _extract_kernels(content)
    analyzed_kernels = []
    
    for kernel in kernels:
        analyzed = _analyze_kernel(kernel, content)
        analyzed_kernels.append(analyzed)
    
    # Generate summary
    kernels_with_divergence = sum(
        1 for k in analyzed_kernels if k["potential_warp_divergence"]
    )
    kernels_with_coalesced = sum(
        1 for k in analyzed_kernels 
        if "coalesced" in k["memory_access_pattern"].lower()
    )
    
    return {
        "file": str(file_path),
        "kernels": analyzed_kernels,
        "summary": {
            "total_kernels": len(analyzed_kernels),
            "kernels_with_divergence": kernels_with_divergence,
            "kernels_with_coalesced_access": kernels_with_coalesced,
        }
    }


def _extract_kernels(content: str) -> List[Dict[str, str]]:
    """Extract kernel function definitions from CUDA source.
    
    Args:
        content: Source code content.
        
    Returns:
        List of kernel information dictionaries with 'name' and 'signature'.
    """
    kernels = []
    
    # Pattern to match __global__ void kernel_name(...)
    # Handles various formats: __global__, __device__, etc.
    pattern = r"__global__\s+(?:__device__\s+)?(?:void|int|float|double)\s+(\w+)\s*\([^)]*\)"
    
    for match in re.finditer(pattern, content):
        kernel_name = match.group(1)
        # Find the function body (between { and matching })
        start_pos = match.end()
        brace_count = 0
        body_start = -1
        
        for i in range(start_pos, len(content)):
            if content[i] == '{':
                if brace_count == 0:
                    body_start = i
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0 and body_start != -1:
                    body = content[body_start:i+1]
                    kernels.append({
                        "name": kernel_name,
                        "signature": match.group(0),
                        "body": body,
                        "full_match": content[match.start():i+1]
                    })
                    break
    
    return kernels


def _analyze_kernel(kernel: Dict[str, str], full_content: str) -> Dict:
    """Analyze a single kernel for various patterns.
    
    Args:
        kernel: Kernel dictionary with name, signature, body.
        full_content: Full source file content.
        
    Returns:
        Analysis results dictionary.
    """
    body = kernel["body"]
    name = kernel["name"]
    
    # Analyze indexing pattern
    indexing_pattern, indexing_formula = _analyze_indexing(body)
    
    # Check for potential warp divergence
    potential_divergence, conditional_checks = _check_warp_divergence(body)
    
    # Analyze memory access patterns
    memory_pattern, memory_accesses = _analyze_memory_access(body)
    
    return {
        "name": name,
        "potential_warp_divergence": potential_divergence,
        "memory_access_pattern": memory_pattern,
        "thread_indexing_pattern": indexing_pattern,
        "indexing_formula": indexing_formula,
        "conditional_checks": conditional_checks,
        "memory_accesses": memory_accesses,
    }


def _analyze_indexing(body: str) -> Tuple[str, str]:
    """Analyze thread/block indexing patterns.
    
    Args:
        body: Kernel function body.
        
    Returns:
        Tuple of (pattern_type, formula_string).
    """
    # Look for common indexing patterns
    # 1D: blockIdx.x * blockDim.x + threadIdx.x
    # 2D: blockIdx.y * blockDim.y + threadIdx.y, etc.
    
    # Pattern for 1D indexing
    pattern_1d = r"(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x"
    match_1d = re.search(pattern_1d, body)
    
    if match_1d:
        var_name = match_1d.group(1)
        return "1D_grid", f"{var_name} = blockIdx.x * blockDim.x + threadIdx.x"
    
    # Pattern for 2D indexing
    pattern_2d_x = r"(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x"
    pattern_2d_y = r"(\w+)\s*=\s*blockIdx\.y\s*\*\s*blockDim\.y\s*\+\s*threadIdx\.y"
    
    if re.search(pattern_2d_x, body) and re.search(pattern_2d_y, body):
        return "2D_grid", "2D indexing detected (x and y dimensions)"
    
    # Pattern for 3D indexing
    pattern_3d_z = r"(\w+)\s*=\s*blockIdx\.z\s*\*\s*blockDim\.z\s*\+\s*threadIdx\.z"
    if re.search(pattern_2d_x, body) and re.search(pattern_2d_y, body) and re.search(pattern_3d_z, body):
        return "3D_grid", "3D indexing detected (x, y, z dimensions)"
    
    # Check if any indexing is present
    if re.search(r"threadIdx\.", body) or re.search(r"blockIdx\.", body):
        return "custom", "Custom indexing pattern detected"
    
    return "unknown", "No standard indexing pattern detected"


def _check_warp_divergence(body: str) -> Tuple[bool, List[str]]:
    """Check for potential warp divergence (if conditions on thread/block indices).
    
    Args:
        body: Kernel function body.
        
    Returns:
        Tuple of (has_divergence, list_of_conditional_checks).
    """
    conditional_checks = []
    has_divergence = False
    
    # Pattern to find if statements - more comprehensive
    # Handles multi-line conditions
    if_pattern = r"if\s*\([^)]+\)"
    
    for match in re.finditer(if_pattern, body):
        condition = match.group(0)
        
        # Extract the condition expression
        condition_text = re.search(r"if\s*\(([^)]+)\)", condition)
        if condition_text:
            cond_expr = condition_text.group(1).strip()
            
            # Check if condition involves threadIdx or blockIdx directly
            involves_thread_idx = re.search(r"threadIdx\.", cond_expr)
            involves_block_idx = re.search(r"blockIdx\.", cond_expr)
            
            # Also check if condition uses the indexing variable
            # (which is derived from threadIdx/blockIdx)
            indexing_var_match = re.search(r"(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x", body)
            if indexing_var_match:
                indexing_var = indexing_var_match.group(1)
                uses_indexing_var = re.search(rf"\b{indexing_var}\b", cond_expr)
            else:
                uses_indexing_var = False
            
            if involves_thread_idx or involves_block_idx or uses_indexing_var:
                conditional_checks.append(cond_expr)
                
                # Check for bounds checking (common pattern, usually causes minimal divergence)
                # Simple bounds check: i < n, i <= n, i >= 0, etc.
                is_simple_bounds_check = (
                    re.search(r"<\s*\w+", cond_expr) or  # i < n
                    re.search(r"<=\s*\w+", cond_expr) or  # i <= n
                    re.search(r">=\s*0", cond_expr) or  # i >= 0
                    re.search(r"<\s*\d+", cond_expr) or  # i < 1024
                    re.search(r"<=\s*\d+", cond_expr)  # i <= 1024
                )
                
                # More complex conditions (modulo, bitwise, etc.) are more likely to cause divergence
                is_complex_condition = (
                    re.search(r"%", cond_expr) or  # modulo
                    re.search(r"&", cond_expr) or  # bitwise and
                    re.search(r"\|\|", cond_expr) or  # logical or
                    re.search(r"&&", cond_expr)  # logical and (beyond simple bounds)
                )
                
                if is_complex_condition and not is_simple_bounds_check:
                    has_divergence = True
                elif involves_thread_idx and not is_simple_bounds_check:
                    # Direct threadIdx checks that aren't bounds checks
                    has_divergence = True
    
    return has_divergence, conditional_checks


def _analyze_memory_access(body: str) -> Tuple[str, List[str]]:
    """Analyze memory access patterns in the kernel.
    
    Args:
        body: Kernel function body.
        
    Returns:
        Tuple of (pattern_description, list_of_access_expressions).
    """
    memory_accesses = []
    
    # Pattern to find array accesses: array[index] or array[index + offset]
    # This is simplified - real analysis would need more context
    access_pattern = r"(\w+)\[([^\]]+)\]"
    
    for match in re.finditer(access_pattern, body):
        array_name = match.group(1)
        index_expr = match.group(2)
        memory_accesses.append(f"{array_name}[{index_expr}]")
    
    if not memory_accesses:
        return "no_array_access", []
    
    # Analyze if accesses are likely coalesced
    # Coalesced: consecutive threads access consecutive memory locations
    # Pattern: array[i] where i = blockIdx.x * blockDim.x + threadIdx.x
    
    # Check if indexing variable is used directly in array access
    indexing_var = None
    indexing_match = re.search(r"(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x", body)
    if indexing_match:
        indexing_var = indexing_match.group(1)
    
    if indexing_var:
        # Check if array accesses use the indexing variable directly
        direct_accesses = [
            acc for acc in memory_accesses 
            if indexing_var in acc or re.search(rf"{indexing_var}\s*[+\-]", acc)
        ]
        
        if len(direct_accesses) == len(memory_accesses):
            # All accesses use the indexing variable
            # Check for stride patterns
            stride_pattern = re.search(rf"{indexing_var}\s*\+\s*(\w+)", str(memory_accesses))
            if stride_pattern:
                stride_var = stride_pattern.group(1)
                # If stride is a constant or small, might still be OK
                if stride_var.isdigit() and int(stride_var) <= 32:
                    return "likely_coalesced_with_small_stride", memory_accesses
                return "potentially_non_coalesced_stride", memory_accesses
            
            return "likely_coalesced", memory_accesses
        else:
            return "mixed_access_pattern", memory_accesses
    
    # Check for constant offsets
    constant_offset_pattern = re.search(r"\[\s*(\w+)\s*\+\s*(\d+)\s*\]", body)
    if constant_offset_pattern:
        offset = constant_offset_pattern.group(2)
        if int(offset) <= 32:
            return "likely_coalesced_with_offset", memory_accesses
    
    return "unknown_access_pattern", memory_accesses


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python static_analyzer.py <path_to_cuda_file.cu>", file=sys.stderr)
        sys.exit(1)
    
    cuda_file_path = sys.argv[1]
    try:
        result = analyze_cuda_file(cuda_file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

