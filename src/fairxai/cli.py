"""Command-line interface for FairXAI pipeline."""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer(
    name="fairxai",
    help="Fair and Explainable AI for Healthcare - CLI Tools",
    add_completion=False
)


@app.command()
def data_load(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., 'cardiac')"),
    output_dir: Path = typer.Option("data/raw", help="Output directory for loaded data"),
):
    """Load and standardize raw datasets."""
    typer.echo(f"🔄 Loading {dataset} dataset...")
    typer.echo(f"📁 Output directory: {output_dir}")
    
    # TODO: Import and call appropriate loader based on dataset name
    typer.echo("⚠️  CLI integration in progress - use scripts/data/load_*.py for now")


@app.command()
def data_preprocess(
    dataset: str = typer.Argument(..., help="Dataset name to preprocess"),
    config: Optional[Path] = typer.Option(None, help="Path to preprocessing config YAML"),
    test_size: float = typer.Option(0.3, help="Test set proportion"),
):
    """Preprocess data with train/test splitting and scaling."""
    typer.echo(f"🔄 Preprocessing {dataset} dataset...")
    if config:
        typer.echo(f"📋 Using config: {config}")
    typer.echo(f"📊 Test size: {test_size}")
    
    # TODO: Import and call preprocessor with config
    typer.echo("⚠️  CLI integration in progress - use scripts/data/preprocess_*.py for now")


@app.command()
def train(
    config: Path = typer.Argument(..., help="Path to training config YAML"),
    output_dir: Path = typer.Option("experiments/", help="Output directory for results"),
):
    """Train a model using specified configuration."""
    typer.echo(f"🤖 Training model with config: {config}")
    typer.echo(f"📁 Output directory: {output_dir}")
    
    # TODO: Parse config and run training pipeline
    typer.echo("⚠️  CLI integration in progress - use scripts/models/train_*.py for now")


@app.command()
def evaluate(
    results_dir: Path = typer.Argument(..., help="Path to training results directory"),
    fairness: bool = typer.Option(True, help="Include fairness assessment"),
):
    """Evaluate trained model performance and fairness."""
    typer.echo(f"📊 Evaluating results from: {results_dir}")
    if fairness:
        typer.echo("⚖️  Including fairness metrics")
    
    # TODO: Load results and run evaluation
    typer.echo("⚠️  CLI integration in progress - use scripts/fairness/assess_*.py for now")


@app.command()
def report(
    results_dir: Path = typer.Argument(..., help="Path to results directory"),
    output: Path = typer.Option("reports/fairness_report.html", help="Output report path"),
    template: Optional[Path] = typer.Option(None, help="Custom notebook template"),
):
    """Generate comprehensive fairness report."""
    typer.echo(f"📝 Generating report from: {results_dir}")
    typer.echo(f"💾 Output: {output}")
    
    # TODO: Use Papermill to execute template notebook
    typer.echo("⚠️  CLI integration in progress - use notebooks for now")


@app.command()
def version():
    """Show FairXAI version information."""
    typer.echo("FairXAI v0.1.0-alpha")
    typer.echo("Fair and Explainable AI for Healthcare Decision Support")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
