"""Command-line interface for PersonaEval."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .config import Config
from .evaluator import Evaluator
from .metrics import MetricsCalculator


console = Console()


@click.group()
@click.version_option()
def main():
    """PersonaEval: A comprehensive benchmark for LLM judges of role-play."""
    pass


@main.command()
@click.option('--config', '-c', 'config_path', default='configs/default.yaml',
              help='Path to configuration file')
@click.option('--track', '-t', 'track_name', default='all',
              help='Track name to evaluate (or "all" for all tracks)')
@click.option('--model', '-m', 'model_name', 
              help='Model name to evaluate')
@click.option('--no-resume', is_flag=True, default=False,
              help='Do not resume from existing results')
@click.option('--list-tracks', is_flag=True, default=False,
              help='List available tracks and exit')
@click.option('--list-models', is_flag=True, default=False,
              help='List available models and exit')
def run(config_path: str, track_name: str, model_name: str, 
        no_resume: bool, list_tracks: bool, list_models: bool):
    """Run evaluation experiments."""
    
    try:
        # Load configuration
        config = Config.from_file(config_path)
        
        # Handle list commands
        if list_tracks:
            _list_tracks(config)
            return
        
        if list_models:
            _list_models(config)
            return
        
        # Validate model if provided
        if model_name:
            if model_name not in config.list_models():
                console.print(f"[red]Error: Model '{model_name}' not found in configuration[/red]")
                console.print(f"Available models: {', '.join(config.list_models())}")
                sys.exit(1)
        else:
            console.print("[red]Error: Model name is required for running experiments[/red]")
            console.print("Use --model to specify a model, or --list-models to see available models")
            sys.exit(1)
        
        # Create evaluator
        evaluator = Evaluator(config)
        
        # Run experiments
        if track_name.lower() == 'all':
            console.print(f"Running {model_name} on all tracks...")
            results = evaluator.run_all_tracks(model_name, resume=not no_resume)
            _display_all_results(results, model_name)
        else:
            # Validate track
            if track_name not in config.list_tracks():
                console.print(f"[red]Error: Track '{track_name}' not found in configuration[/red]")
                console.print(f"Available tracks: {', '.join(config.list_tracks())}")
                sys.exit(1)
            
            console.print(f"Running {model_name} on {track_name}...")
            results = evaluator.run_experiment(track_name, model_name, resume=not no_resume)
    
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Please check your configuration file path.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('result_file', type=click.Path(exists=True))
def analyze(result_file: str):
    """Analyze experiment results."""
    
    try:
        import pandas as pd
        from .utils import calculate_statistics
        
        # Load results
        res_df = pd.read_csv(result_file)
        console.print(f"Loaded results from {result_file}")
        
        # Calculate statistics
        stats = calculate_statistics(res_df)
        
        # Display results
        table = Table(title="Experiment Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in stats.items():
            if isinstance(value, float):
                if metric == "accuracy":
                    table.add_row(metric, f"{value:.3f}")
                elif metric == "total_cost":
                    table.add_row(metric, f"${value:.4f}")
                elif metric == "completion_rate":
                    table.add_row(metric, f"{value:.1%}")
                else:
                    table.add_row(metric, f"{value:.2f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
        
        # Show error analysis if any
        if stats["error_count"] > 0:
            console.print(f"\n[yellow]Warning: {stats['error_count']} samples had errors[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--results-dir', '-r', default='results', help='Results directory')
@click.option('--models', '-m', 'model_names', help='Comma-separated list of model names')
@click.option('--tracks', default='Literary,Drama,Expertise', help='Comma-separated list of track names')
@click.option('--output', '-o', default='metrics.csv', help='Output CSV file path')
@click.option('--plot', is_flag=True, help='Generate comparison plots')
@click.option('--plot-output', default='metrics_comparison.png', help='Plot output file path')
def metrics(results_dir: str, model_names: str, tracks: str, output: str, plot: bool, plot_output: str):
    """Calculate evaluation metrics for experiment results."""
    
    try:
        # Parse track names
        track_names = [t.strip() for t in tracks.split(',')]
        
        # Parse model names
        if model_names:
            model_list = [m.strip() for m in model_names.split(',')]
        else:
            console.print("[red]Error: --models must be specified[/red]")
            sys.exit(1)
        
        console.print(f"Calculating metrics for models: {model_list}")
        console.print(f"Tracks: {track_names}")
        console.print(f"Results directory: {results_dir}")
        
        # Calculate metrics
        calculator = MetricsCalculator()
        
        if len(model_list) == 1:
            metrics_df = calculator.calculate_track_metrics(results_dir, model_list[0], track_names)
        else:
            metrics_df = calculator.calculate_multiple_models_metrics(results_dir, model_list, track_names)
        
        if len(metrics_df) == 0:
            console.print("[yellow]No metrics calculated. Check if result files exist.[/yellow]")
            return
        
        # Save metrics
        formatted_df = calculator.save_metrics(metrics_df, output)

        # Display metrics
        calculator.display_metrics(formatted_df)
        
        # Create plots if requested
        if plot:
            calculator.create_comparison_plot(metrics_df, plot_output)
        
        console.print(f"\n[green]Metrics calculation completed![/green]")
        console.print(f"Results saved to: {output}")
        if plot:
            console.print(f"Plot saved to: {plot_output}")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _list_tracks(config: Config):
    """List available tracks."""
    table = Table(title="Available Tracks")
    table.add_column("Track Name", style="cyan")
    table.add_column("Data File", style="magenta")
    table.add_column("Output Directory", style="green")
    
    for track in config.tracks:
        table.add_row(track.name, track.data_file, track.output_dir)
    
    console.print(table)


def _list_models(config: Config):
    """List available models."""
    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("API URL", style="magenta")
    table.add_column("Input Cost", style="green")
    table.add_column("Output Cost", style="yellow")
    
    for model_name, model_config in config.models.items():
        table.add_row(
            model_name,
            model_config.url,
            f"${model_config.cost_input:.6f}",
            f"${model_config.cost_output:.6f}"
        )
    
    console.print(table)


def _display_all_results(results: dict, model_name: str):
    """Display results for all tracks."""
    table = Table(title=f"Results for {model_name} on All Tracks")
    table.add_column("Track", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Completed", style="green")
    table.add_column("Total Cost", style="yellow")
    table.add_column("Avg Tokens", style="blue")
    
    for track_name, track_results in results.items():
        if "error" in track_results:
            table.add_row(track_name, "ERROR", "-", "-", "-")
        else:
            table.add_row(
                track_name,
                f"{track_results['accuracy']:.3f}",
                f"{track_results['completed_samples']}/{track_results['total_samples']}",
                f"${track_results['total_cost']:.4f}",
                f"{track_results['avg_tokens']:.1f}"
            )
    
    console.print(table)


if __name__ == "__main__":
    main()