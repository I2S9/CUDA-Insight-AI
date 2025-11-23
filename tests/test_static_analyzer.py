"""Tests for the static analyzer."""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.static_analyzer import analyze_cuda_file


def test_analyze_saxpy():
    """Test analysis of saxpy kernel."""
    result = analyze_cuda_file("src/cuda/example_kernels/saxpy.cu")
    
    assert "file" in result
    assert "kernels" in result
    assert "summary" in result
    assert len(result["kernels"]) == 1
    
    kernel = result["kernels"][0]
    assert kernel["name"] == "saxpy"
    assert kernel["thread_indexing_pattern"] == "1D_grid"
    assert "likely_coalesced" in kernel["memory_access_pattern"]
    assert len(kernel["conditional_checks"]) > 0
    assert "i < n" in kernel["conditional_checks"]
    
    print("[OK] saxpy analysis passed")
    print(json.dumps(result, indent=2))


def test_analyze_vector_add():
    """Test analysis of vector_add kernel."""
    result = analyze_cuda_file("src/cuda/example_kernels/vector_add.cu")
    
    assert len(result["kernels"]) == 1
    kernel = result["kernels"][0]
    assert kernel["name"] == "vector_add"
    assert kernel["thread_indexing_pattern"] == "1D_grid"
    
    print("[OK] vector_add analysis passed")
    print(json.dumps(result, indent=2))


def test_json_structure():
    """Test that the JSON structure matches the expected format."""
    result = analyze_cuda_file("src/cuda/example_kernels/saxpy.cu")
    
    # Verify structure
    assert isinstance(result, dict)
    assert "file" in result
    assert "kernels" in result
    assert "summary" in result
    
    # Verify kernel structure
    kernel = result["kernels"][0]
    required_fields = [
        "name",
        "potential_warp_divergence",
        "memory_access_pattern",
        "thread_indexing_pattern",
        "indexing_formula",
        "conditional_checks",
        "memory_accesses",
    ]
    
    for field in required_fields:
        assert field in kernel, f"Missing field: {field}"
    
    # Verify types
    assert isinstance(kernel["name"], str)
    assert isinstance(kernel["potential_warp_divergence"], bool)
    assert isinstance(kernel["memory_access_pattern"], str)
    assert isinstance(kernel["thread_indexing_pattern"], str)
    assert isinstance(kernel["conditional_checks"], list)
    assert isinstance(kernel["memory_accesses"], list)
    
    print("[OK] JSON structure validation passed")


if __name__ == "__main__":
    test_analyze_saxpy()
    print()
    test_analyze_vector_add()
    print()
    test_json_structure()
    print("\nAll tests passed!")

