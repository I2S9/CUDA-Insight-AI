"""CUDA kernel runner with compilation and execution support."""

import os
import subprocess
import sys
import time
import ctypes
from pathlib import Path
from typing import Optional, Tuple
import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Try to import CUDA runtime
CUDA_AVAILABLE = False
try:
    if sys.platform == "win32":
        cudart = ctypes.CDLL("cudart64_12.dll")
    else:
        cudart = ctypes.CDLL("libcudart.so")
    CUDA_AVAILABLE = True
except OSError:
    pass


class CUDARunner:
    """Runner for CUDA kernels with compilation and execution."""

    def __init__(self, mock_mode: bool = False):
        """Initialize the CUDA runner.
        
        Args:
            mock_mode: If True, simulate execution without GPU.
        """
        self.mock_mode = mock_mode or not CUDA_AVAILABLE
        if self.mock_mode:
            console.print("[yellow]Running in mock mode (no GPU available)[/yellow]")

    def compile_kernel(self, cu_file: Path, output_dir: Optional[Path] = None) -> Path:
        """Compile a CUDA kernel file to a shared library.
        
        Args:
            cu_file: Path to the .cu source file.
            output_dir: Directory for output files. Defaults to same as cu_file.
            
        Returns:
            Path to the compiled shared library.
        """
        if output_dir is None:
            output_dir = cu_file.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = cu_file.stem
        output_file = output_dir / f"{output_name}.so"
        
        if sys.platform == "win32":
            output_file = output_dir / f"{output_name}.dll"
        
        # Check if nvcc is available
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"[green]Found nvcc:[/green] {result.stdout.split(chr(10))[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "nvcc not found. Please install CUDA toolkit and ensure nvcc is in PATH."
            )
        
        # Compile command
        compile_cmd = [
            "nvcc",
            str(cu_file),
            "-o", str(output_file),
            "--shared",
            "-Xcompiler", "-fPIC" if sys.platform != "win32" else "",
        ]
        
        if sys.platform == "win32":
            compile_cmd = [
                "nvcc",
                str(cu_file),
                "-o", str(output_file),
                "--shared",
                "-Xcompiler", "/MD",
            ]
        
        console.print(f"[blue]Compiling {cu_file.name}...[/blue]")
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"[green]Compilation successful:[/green] {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Compilation failed:[/red]")
            console.print(e.stderr)
            raise

    def load_kernel(self, so_file: Path):
        """Load a compiled CUDA kernel library.
        
        Args:
            so_file: Path to the compiled shared library.
            
        Returns:
            Loaded library object.
        """
        try:
            lib = ctypes.CDLL(str(so_file))
            return lib
        except OSError as e:
            raise RuntimeError(f"Failed to load kernel library: {e}")

    def run_kernel(
        self,
        cu_file: Path,
        kernel_name: Optional[str] = None,
        compile_kernel: bool = True,
        **kwargs
    ) -> Tuple[float, dict]:
        """Run a CUDA kernel and measure execution time.
        
        Args:
            cu_file: Path to the .cu source file.
            kernel_name: Name of the kernel launch function. If None, tries to find
                        a test function (test_*) or default launch function.
            compile_kernel: Whether to compile the kernel first.
            **kwargs: Arguments to pass to the kernel.
            
        Returns:
            Tuple of (execution_time_seconds, metadata_dict).
        """
        if self.mock_mode:
            return self._run_mock(cu_file, kernel_name or "test", **kwargs)
        
        # Compile if needed
        if compile_kernel:
            so_file = self.compile_kernel(cu_file)
        else:
            # Try to find existing compiled file
            so_file = cu_file.parent / f"{cu_file.stem}.so"
            if sys.platform == "win32":
                so_file = cu_file.parent / f"{cu_file.stem}.dll"
            if not so_file.exists():
                raise FileNotFoundError(f"Compiled kernel not found: {so_file}")
        
        # Load and run
        lib = self.load_kernel(so_file)
        
        # Auto-detect kernel function if not specified
        if kernel_name is None:
            # Try to find a test function first
            available_funcs = [name for name in dir(lib) if not name.startswith('_')]
            test_funcs = [f for f in available_funcs if f.startswith('test_')]
            if test_funcs:
                kernel_name = test_funcs[0]
                console.print(f"[blue]Auto-detected test function:[/blue] {kernel_name}")
            elif 'launch' in available_funcs:
                kernel_name = 'launch'
                console.print(f"[blue]Using default launch function[/blue]")
            else:
                raise AttributeError(
                    f"No suitable kernel function found. Available: {available_funcs}"
                )
        
        # Get the kernel function
        try:
            kernel_func = getattr(lib, kernel_name)
        except AttributeError:
            available_funcs = [name for name in dir(lib) if not name.startswith('_')]
            raise AttributeError(
                f"Kernel function '{kernel_name}' not found in {so_file}. "
                f"Available functions: {available_funcs}"
            )
        
        # Set return type if it's a test function (returns int)
        if kernel_name.startswith('test_'):
            kernel_func.restype = ctypes.c_int
        
        # Measure execution time
        start_time = time.perf_counter()
        if kernel_name.startswith('test_'):
            result = kernel_func()
            if result != 0:
                console.print(f"[yellow]Warning:[/yellow] Test function returned {result}")
        else:
            kernel_func(**kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        metadata = {
            "kernel_file": str(cu_file),
            "compiled_file": str(so_file),
            "kernel_name": kernel_name,
            "execution_time_ms": execution_time * 1000,
        }
        
        return execution_time, metadata

    def _run_mock(self, cu_file: Path, kernel_name: str, **kwargs) -> Tuple[float, dict]:
        """Simulate kernel execution in mock mode.
        
        Args:
            cu_file: Path to the .cu source file.
            kernel_name: Name of the kernel launch function.
            **kwargs: Arguments to pass to the kernel.
            
        Returns:
            Tuple of (simulated_execution_time_seconds, metadata_dict).
        """
        # Simulate some processing time
        time.sleep(0.001)  # 1ms simulation
        
        # Generate mock execution time (0.5-2.0 ms)
        import random
        mock_time = random.uniform(0.0005, 0.002)
        
        metadata = {
            "kernel_file": str(cu_file),
            "kernel_name": kernel_name,
            "execution_time_ms": mock_time * 1000,
            "mock_mode": True,
        }
        
        return mock_time, metadata


def main(
    kernel: str = typer.Option(..., "--kernel", "-k", help="Path to the CUDA .cu file"),
    mock: bool = typer.Option(False, "--mock", help="Run in mock mode (no GPU required)"),
    no_compile: bool = typer.Option(False, "--no-compile", help="Skip compilation (use existing .so/.dll)"),
    kernel_func: Optional[str] = typer.Option(None, "--kernel-func", help="Name of the kernel launch function (auto-detected if not specified)"),
):
    """Run a CUDA kernel from a .cu file.
    
    Examples:
        python src/cuda/runner.py --kernel src/cuda/example_kernels/saxpy.cu
        python src/cuda/runner.py --kernel src/cuda/example_kernels/vector_add.cu --mock
    """
    cu_file = Path(kernel)
    
    if not cu_file.exists():
        console.print(f"[red]Error:[/red] Kernel file not found: {cu_file}")
        raise typer.Exit(1)
    
    if not cu_file.suffix == ".cu":
        console.print(f"[red]Error:[/red] File must have .cu extension: {cu_file}")
        raise typer.Exit(1)
    
    console.print(Panel(f"[bold]CUDA Kernel Runner[/bold]\nFile: {cu_file}", title="Info"))
    
    runner = CUDARunner(mock_mode=mock)
    
    try:
        execution_time, metadata = runner.run_kernel(
            cu_file,
            kernel_name=kernel_func,
            compile_kernel=not no_compile,
        )
        
        console.print("\n[bold green]Execution successful![/bold green]")
        console.print(f"Execution time: {metadata['execution_time_ms']:.3f} ms")
        console.print(f"Kernel: {metadata['kernel_name']}")
        if metadata.get("mock_mode"):
            console.print("[yellow](Mock mode - simulated execution)[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)

