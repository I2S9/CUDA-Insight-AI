"""Main CLI interface for CUDA-Insight-AI."""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ai.llm_agent import analyze_cuda_file_path

app = typer.Typer(
    name="cuda-insight",
    help="CUDA-Insight-AI: AI-powered CUDA kernel analysis and optimization tool",
    add_completion=False
)
console = Console()


@app.command()
def main(
    kernel: str = typer.Option(..., "--kernel", "-k", help="Path to the CUDA .cu file to analyze"),
    profile: bool = typer.Option(False, "--profile", "-p", help="Enable runtime profiling"),
    save_report: Optional[str] = typer.Option(None, "--save-report", "-o", help="Save report to file (.txt or .md)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="OpenAI model to use"),
    mock: bool = typer.Option(False, "--mock", help="Use mock mode (no GPU/API required for testing)"),
) -> None:
    """Analyze a CUDA kernel and provide AI-powered optimization recommendations.
    
    This tool performs static analysis, optional runtime profiling, and uses an LLM
    to generate comprehensive optimization recommendations.
    
    Examples:
        python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu
        python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --profile
        python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --save-report report.txt
        python -m src.cli.main --kernel src/cuda/example_kernels/saxpy.cu --profile --save-report report.md
    """
    # Validate kernel file
    kernel_path = Path(kernel)
    
    if not kernel_path.exists():
        console.print(f"[red]Error:[/red] Kernel file not found: {kernel_path}")
        raise typer.Exit(1)
    
    if not kernel_path.suffix == ".cu":
        console.print(f"[red]Error:[/red] File must have .cu extension: {kernel_path}")
        raise typer.Exit(1)
    
    # Check API key if not in mock mode
    if not mock:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error:[/red] OpenAI API key required.")
            console.print("Set OPENAI_API_KEY environment variable or use --api-key option.")
            console.print("For testing without API, use --mock flag.")
            raise typer.Exit(1)
    
    # Display header
    console.print()
    console.print(Panel(
        f"[bold cyan]CUDA-Insight-AI[/bold cyan]\n"
        f"Analyzing: [bold]{kernel_path}[/bold]\n"
        f"Profiling: {'[green]Enabled[/green]' if profile else '[yellow]Disabled[/yellow]'}\n"
        f"Model: [bold]{model}[/bold]",
        title="Analysis Configuration",
        border_style="cyan"
    ))
    console.print()
    
    # Run analysis with progress indicator
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Analyzing CUDA kernel...", total=None)
            
            report = analyze_cuda_file_path(
                file_path=str(kernel_path),
                api_key=api_key,
                model=model,
                enable_profiling=profile,
                mock_mode=mock
            )
            
            progress.update(task, completed=True)
        
        console.print()
        
        # Display report
        console.print(Panel(
            Markdown(report),
            title="[bold green]Analysis Report[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Save report if requested
        if save_report:
            output_path = Path(save_report)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension
            if output_path.suffix.lower() == ".md":
                # Save as markdown
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
            else:
                # Save as plain text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
            
            console.print()
            console.print(f"[green]Report saved to:[/green] {output_path.absolute()}")
        
        console.print()
        console.print("[green]Analysis complete![/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during analysis:[/red] {e}")
        if not mock:
            console.print("\n[yellow]Tip:[/yellow] Use --mock flag to test without API/GPU")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

