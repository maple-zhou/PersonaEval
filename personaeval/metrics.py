"""Metrics calculation for PersonaEval."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .utils import create_safe_model_name

class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def extract_options_probs(self, row: pd.Series) -> Tuple[List[str], List[float]]:
        """Extract options and probabilities from a row."""
        options, probs = [], []
        for i in range(1, 6):  # Support up to 5 options
            opt_key = f'option{i}'
            prob_key = f'prob{i}'
            if pd.notna(row.get(opt_key)) and pd.notna(row.get(prob_key)):
                options.append(row[opt_key])
                probs.append(row[prob_key])
        return options, probs
    
    def calculate_top1_accuracy(self, res_df: pd.DataFrame) -> float:
        """Calculate Top-1 accuracy."""
        correct = 0
        total = 0
        for _, row in res_df.iterrows():
            if pd.isna(row['gt']):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            pred = options[np.argmax(probs)]
            correct += int(pred == row['gt'])
            total += 1
        return correct / total if total > 0 else 0.0
    
    def calculate_top2_accuracy(self, res_df: pd.DataFrame) -> float:
        """Calculate Top-2 accuracy."""
        correct = 0
        total = 0
        for _, row in res_df.iterrows():
            if pd.isna(row['gt']):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            top2_idx = np.argsort(probs)[-2:][::-1]
            top2_preds = [options[i] for i in top2_idx]
            correct += int(row['gt'] in top2_preds)
            total += 1
        return correct / total if total > 0 else 0.0
    
    def calculate_mean_rank(self, res_df: pd.DataFrame) -> float:
        """Calculate Mean Rank (MR)."""
        ranks = []
        for _, row in res_df.iterrows():
            if pd.isna(row['gt']):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            sorted_indices = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_indices):
                if options[idx] == row['gt']:
                    ranks.append(rank + 1)
                    break
        return np.mean(ranks) if ranks else 0.0
    
    def calculate_brier_score(self, res_df: pd.DataFrame) -> float:
        """Calculate Brier Score."""
        scores = []
        for _, row in res_df.iterrows():
            gt = row['gt']
            if pd.isna(gt):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            for opt, prob in zip(options, probs):
                scores.append((prob - int(opt == gt)) ** 2)
        return np.mean(scores) if scores else 0.0
    
    def calculate_ece(self, res_df: pd.DataFrame, n_bins: int = 20) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        preds_conf = []
        for _, row in res_df.iterrows():
            if pd.isna(row['gt']):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            max_idx = np.argmax(probs)
            conf = probs[max_idx]
            correct = int(options[max_idx] == row['gt'])
            preds_conf.append((conf, correct))
        
        if not preds_conf:
            return 0.0
        
        preds_conf = np.array(preds_conf)
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            in_bin = (preds_conf[:,0] > low) & (preds_conf[:,0] <= high)
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(preds_conf[in_bin][:, 0])
                accuracy = np.mean(preds_conf[in_bin][:, 1])
                ece += (np.sum(in_bin) / len(preds_conf)) * abs(avg_conf - accuracy)
        
        return ece
    
    def calculate_f1_score(self, res_df: pd.DataFrame) -> float:
        """Calculate F1 Score."""
        preds, gts = [], []
        for _, row in res_df.iterrows():
            if pd.isna(row['gt']):
                continue
            options, probs = self.extract_options_probs(row)
            if not options or not probs:
                continue
            pred = options[np.argmax(probs)]
            preds.append(1 if pred == row['gt'] else 0)
            gts.append(1)  # All samples are positive class
        return f1_score(gts, preds) if gts else 0.0
    
    def calculate_all_metrics(self, res_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all metrics for a given dataframe."""
        # Filter out rows with missing data
        valid_df = res_df.dropna(subset=['gt', 'success'])
        
        if len(valid_df) == 0:
            return {
                'num_samples': 0,
                'top1_accuracy': 0.0,
                'top2_accuracy': 0.0,
                'mean_rank': 0.0,
                'brier_score': 0.0,
                'ece': 0.0,
                'f1_score': 0.0
            }
        
        metrics = {
            'num_samples': len(valid_df),
            'top1_accuracy': self.calculate_top1_accuracy(valid_df),
            'top2_accuracy': self.calculate_top2_accuracy(valid_df),
            'mean_rank': self.calculate_mean_rank(valid_df),
            'brier_score': self.calculate_brier_score(valid_df),
            'ece': self.calculate_ece(valid_df),
            'f1_score': self.calculate_f1_score(valid_df)
        }
        
        return metrics
    
    def load_results(self, results_dir: str, model_name: str, track_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Load results for all tracks."""
        results = {}
        for track in track_names:
            # Handle model names with slashes
            safe_model_name = create_safe_model_name(model_name)
            file_path = Path(results_dir) / track / f"{track}_{safe_model_name}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                results[track] = df
                console = Console()
                console.print(f"[green]Loaded {len(df)} samples from {track}[/green]")
            else:
                console = Console()
                console.print(f"[yellow]Warning: Results file not found for {track}: {file_path}[/yellow]")
                results[track] = pd.DataFrame()
        
        return results
    
    def calculate_track_metrics(self, results_dir: str, model_name: str, track_names: List[str]) -> pd.DataFrame:
        """Calculate metrics for individual tracks and combined results."""
        results = self.load_results(results_dir, model_name, track_names)
        
        # Calculate metrics for each track
        track_metrics = []
        for track_name, df in results.items():
            if len(df) == 0:
                continue
            
            metrics = self.calculate_all_metrics(df)
            metrics['track'] = track_name
            metrics['model'] = model_name
            track_metrics.append(metrics)
        
        # Calculate combined metrics
        all_dfs = [df for df in results.values() if len(df) > 0]
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_metrics = self.calculate_all_metrics(combined_df)
            combined_metrics['track'] = 'Combined'
            combined_metrics['model'] = model_name
            track_metrics.append(combined_metrics)
        
        return pd.DataFrame(track_metrics)
    
    def calculate_multiple_models_metrics(self, results_dir: str, model_names: List[str], track_names: List[str]) -> pd.DataFrame:
        """Calculate metrics for multiple models."""
        all_metrics = []
        
        for model in model_names:
            try:
                model_metrics = self.calculate_track_metrics(results_dir, model, track_names)
                all_metrics.append(model_metrics)
            except Exception as e:
                console = Console()
                console.print(f"[red]Error calculating metrics for {model}: {e}[/red]")
                continue
        
        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_metrics(self, metrics_df: pd.DataFrame, output_path: str):
        """Save metrics to CSV file."""
        # Format metrics for better readability
        formatted_df = metrics_df.copy()
        
        # Convert percentages for display
        percentage_columns = ['top1_accuracy', 'top2_accuracy', 'f1_score']
        for col in percentage_columns:
            if col in formatted_df.columns:
                formatted_df[col] = (formatted_df[col] * 100).round(1)
        
        # Round other metrics
        formatted_df['mean_rank'] = formatted_df['mean_rank'].round(2)
        formatted_df['brier_score'] = (formatted_df['brier_score'] * 100).round(1)
        formatted_df['ece'] = (formatted_df['ece'] * 100).round(1)
        
        formatted_df.to_csv(output_path, index=False)
        console = Console()
        console.print(f"[green]Metrics saved to {output_path}[/green]")
        
        return formatted_df
    
    def display_metrics(self, metrics_df: pd.DataFrame):
        """Display metrics in a formatted table."""
        if len(metrics_df) == 0:
            console = Console()
            console.print("[red]No metrics to display[/red]")
            return
        
        console = Console()
        
        # Create table
        table = Table(title="Evaluation Metrics")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Track", style="blue", no_wrap=True)
        table.add_column("Samples", style="green", justify="right")
        table.add_column("Top-1 Acc (%)", style="magenta", justify="right")
        table.add_column("Top-2 Acc (%)", style="magenta", justify="right")
        table.add_column("Mean Rank", style="yellow", justify="right")
        table.add_column("Brier Score (%)", style="red", justify="right")
        table.add_column("ECE (%)", style="red", justify="right")
        table.add_column("F1 Score (%)", style="green", justify="right")
        
        for _, row in metrics_df.iterrows():
            table.add_row(
                str(row['model']),
                str(row['track']),
                str(row['num_samples']),
                f"{row['top1_accuracy']:.1f}",
                f"{row['top2_accuracy']:.1f}",
                f"{row['mean_rank']:.2f}",
                f"{row['brier_score']:.1f}",
                f"{row['ece']:.1f}",
                f"{row['f1_score']:.1f}"
            )
        
        console.print(table)
    
    def create_comparison_plot(self, metrics_df: pd.DataFrame, output_path: str = None):
        """Create comparison plots for metrics."""
        if len(metrics_df) == 0:
            console = Console()
            console.print("[red]No data for plotting[/red]")
            return
        
        # Filter to only include combined results for model comparison
        combined_df = metrics_df[metrics_df['track'] == 'Combined']
        
        if len(combined_df) == 0:
            console = Console()
            console.print("[red]No combined results found for plotting[/red]")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics_to_plot = [
            ('top1_accuracy', 'Top-1 Accuracy (%)'),
            ('top2_accuracy', 'Top-2 Accuracy (%)'),
            ('mean_rank', 'Mean Rank'),
            ('brier_score', 'Brier Score (%)'),
            ('ece', 'ECE (%)'),
            ('f1_score', 'F1 Score (%)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            # Sort by metric value
            sorted_df = combined_df.sort_values(metric, ascending=False)
            
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
            ax.set_title(title)
            ax.set_xlabel('Models')
            ax.set_ylabel(title)
            
            # Rotate x-axis labels
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, sorted_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            console = Console()
            console.print(f"[green]Plot saved to {output_path}[/green]")
        
        plt.show()


def calculate_metrics_for_model(results_dir: str, model_name: str, track_names: List[str] = None) -> pd.DataFrame:
    """Convenience function to calculate metrics for a single model."""
    if track_names is None:
        track_names = ['Literary', 'Drama', 'Expertise']
    
    calculator = MetricsCalculator()
    return calculator.calculate_track_metrics(results_dir, model_name, track_names)


def calculate_metrics_for_multiple_models(results_dir: str, model_names: List[str], track_names: List[str] = None) -> pd.DataFrame:
    """Convenience function to calculate metrics for multiple models."""
    if track_names is None:
        track_names = ['Literary', 'Drama', 'Expertise']
    
    calculator = MetricsCalculator()
    return calculator.calculate_multiple_models_metrics(results_dir, model_names, track_names) 