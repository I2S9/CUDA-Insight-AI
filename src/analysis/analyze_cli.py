"""Simple CLI for the static analyzer."""

import json
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.json import JSON

from src.analysis.static_analyzer import analyze_cuda_file

console = Console()
app = typer.Typer()


@app.command()
def analyze(
    file: str = typer.Argument(..., help="Path to the CUDA .cu file"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON"),
):
    """Analyze a CUDA kernel file statically.
    
    Examples:
        python -m src.analysis.analyze_cli src/cuda/example_kernels/saxpy.cu
        python -m src.analysis.analyze_cli src/cuda/example_kernels/saxpy.cu -o result.json
    """
    try:
        result = analyze_cuda_file(file)
        
        if output:
            output_path = Path(output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            console.print(f"[green]Results saved to:[/green] {output_path}")
        else:
            if pretty:
                console.print(JSON(json.dumps(result, indent=2)))
            else:
                print(json.dumps(result))
                
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()

