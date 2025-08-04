"""Main evaluation logic for PersonaEval."""

import datetime
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from tqdm import tqdm

from .config import Config, TrackConfig
from .models import ModelManager, APIResponse
from .utils import fix_res_df, create_safe_model_name, calculate_statistics


class Evaluator:
    """Main evaluator class for running experiments."""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.stop_event = threading.Event()
        self.save_queue = Queue()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals for graceful shutdown."""
        self.console.print(f"\n[{datetime.datetime.now()}] Received interrupt signal, shutting down gracefully...")
        self.stop_event.set()
        self.save_queue.put(True)
        sys.exit(0)
    
    def run_experiment(
        self, 
        track_name: str, 
        model_name: str,
        resume: bool = True
    ) -> Dict[str, float]:
        """
        Run experiment for a specific track and model.
        
        Args:
            track_name: Name of the track to evaluate
            model_name: Name of the model to test
            resume: Whether to resume from existing results
            
        Returns:
            Dictionary with experiment results
        """
        track_config = self.config.get_track(track_name)
        model_config = self.config.get_model(model_name)
        
        if not track_config:
            raise ValueError(f"Track '{track_name}' not found in configuration")
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        # Load data
        df_bench = pd.read_csv(track_config.data_file)
        self.console.print(f"Loaded {len(df_bench)} samples from {track_config.data_file}")
        
        # Prepare result file path
        result_file = self._get_result_file_path(track_config, model_name)
        
        # Load or create result dataframe
        res_df = self._load_or_create_results(df_bench, result_file, resume)
        
        # Create model manager
        model_manager = ModelManager(model_config, self.config.experiment)
        
        # Run evaluation
        self.console.print(f"Starting evaluation for {model_name} on {track_name}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Evaluating {len(res_df)} samples on {track_name} with {model_name}...", total=len(res_df))
            
            # Start save thread
            save_thread = threading.Thread(
                target=self._save_worker, 
                args=(res_df, result_file),
                daemon=True
            )
            save_thread.start()
            
            try:
                # Process samples
                with ThreadPoolExecutor(max_workers=self.config.experiment.max_workers) as executor:
                    futures = []
                    
                    for idx, row in res_df.iterrows():
                        if pd.notna(row.get('gt')) and pd.notna(row.get('success')) and pd.notna(row.get('res')) and pd.notna(row.get('response')) and pd.notna(row.get('prob1')) and pd.notna(row.get('prob2')) and pd.notna(row.get('prob3')) and pd.notna(row.get('prob4')):
                            progress.advance(task)
                            continue
                        
                        future = executor.submit(
                            self._process_sample,
                            track_name, row, model_manager, model_name, res_df, idx
                        )
                        futures.append(future)
                    
                    # Wait for completion
                    for future in as_completed(futures):
                        try:
                            future.result()
                            progress.advance(task)
                        except Exception as e:
                            self.console.print(f"Error in sample processing: {e}")
                            progress.advance(task)  # Still advance progress even on error
                
                # Final save and wait for save thread to complete
                self.save_queue.put(True)
                # Give save thread a moment to process the final save
                time.sleep(0.5)
                self.stop_event.set()
                self.console.print("Waiting for save thread to complete...")
                save_thread.join(timeout=5)  # Wait up to 5 seconds for thread to exit
                if save_thread.is_alive():
                    self.console.print("Warning: Save thread did not exit within timeout")
                else:
                    self.console.print("Save thread completed successfully")
                
            except KeyboardInterrupt:
                self.console.print("Interrupted by user")
                self.save_queue.put(True)
                self.stop_event.set()
                save_thread.join(timeout=5)
                raise
        
        # Calculate and display results
        results = calculate_statistics(res_df)
        self._display_results(results)
        
        return results
    
    def run_all_tracks(self, model_name: str, resume: bool = True) -> Dict[str, Dict[str, float]]:
        """Run experiments for all tracks with a given model."""
        all_results = {}
        
        for track_name in self.config.list_tracks():
            self.console.print(f"\n{'='*50}")
            self.console.print(f"Running {model_name} on {track_name}")
            self.console.print(f"{'='*50}")
            
            # Reset stop event and clear save queue for each track
            self.stop_event.clear()
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                except:
                    pass
            
            try:
                results = self.run_experiment(track_name, model_name, resume)
                all_results[track_name] = results
            except Exception as e:
                self.console.print(f"Error running {track_name}: {e}")
                all_results[track_name] = {"error": str(e)}
        
        return all_results
    
    def _get_result_file_path(self, track_config: TrackConfig, model_name: str) -> str:
        """Generate result file path for a track and model."""
        safe_model_name = create_safe_model_name(model_name)
        filename = f"{track_config.name}_{safe_model_name}.csv"
        return os.path.join(track_config.output_dir, filename)
    
    def _load_or_create_results(
        self, 
        df_bench: pd.DataFrame, 
        result_file: str, 
        resume: bool
    ) -> pd.DataFrame:
        """Load existing results or create new result dataframe."""
        
        if resume and os.path.exists(result_file):
            res_df = pd.read_csv(result_file)
            
            # Check if dataframe structure matches
            if len(res_df) != len(df_bench):
                self.console.print("Result file structure mismatch, fixing...")
                res_df = fix_res_df(res_df, df_bench)
            
            # Display current progress
            success = len(res_df[res_df['success'] == 1])
            fail = len(res_df[res_df['success'] == 0])
            total_completed = success + fail
            
            if total_completed > 0:
                self.console.print(f"Resuming from existing results:")
                print(f"Success: {success}, Fail: {fail}, Total: {total_completed}")
                print(f"Accuracy: {success / total_completed:.3f}")
                self.console.print(f"  Cost: ${res_df['cost'].sum():.4f}")
        else:
            # Create new result dataframe
            res_df = df_bench.copy()
            
            # Add result columns
            result_columns = {
                'res': 'string',
                'response': 'string', 
                'success': 'float64',
                'cost': 'float64',
                'tokens': 'int64',
                'prob1': 'float64',
                'prob2': 'float64',
                'prob3': 'float64',
                'prob4': 'float64'
            }
            
            for col, dtype in result_columns.items():
                res_df[col] = pd.Series(dtype=dtype)
        
        return res_df
    
    def _process_sample(
        self,
        track_name: str,
        row: pd.Series,
        model_manager: ModelManager,
        model_name: str,
        res_df: pd.DataFrame,
        idx: int
    ) -> None:
        """Process a single sample."""
        
        prompt = row['prompt']
        options = [row['option1'], row['option2'], row['option3'], row['option4']]
        if track_name == 'Expertise':
            options.append(row['option5'])
        ground_truth = row['gt']
        
        try:
            predicted_answer, api_response, probabilities, tokens = model_manager.call_api(
                prompt=prompt,
                model_name=model_name,
                options=options,
                ground_truth=ground_truth,
                temperature=self.config.experiment.temperature,
                max_retries=self.config.experiment.max_retries,
                sleep_interval=self.config.experiment.sleep_interval
            )
            
            # Update results
            res_df.loc[idx, 'res'] = str(predicted_answer)
            res_df.loc[idx, 'response'] = str(api_response.content)
            res_df.loc[idx, 'success'] = int(predicted_answer == ground_truth)
            res_df.loc[idx, 'cost'] = api_response.cost
            res_df.loc[idx, 'tokens'] = tokens
            res_df.loc[idx, 'prob1'] = probabilities[options[0]]
            res_df.loc[idx, 'prob2'] = probabilities[options[1]]
            res_df.loc[idx, 'prob3'] = probabilities[options[2]]
            res_df.loc[idx, 'prob4'] = probabilities[options[3]]
            
            # Trigger save
            self.save_queue.put(True)
            
        except Exception as e:
            self.console.print(f"Error processing sample {idx}: {e}")
            # Mark as failed but don't stop the experiment
            res_df.loc[idx, 'success'] = -1  # -1 indicates error
    
    def _save_worker(self, res_df: pd.DataFrame, result_file: str) -> None:
        """Background worker for saving results periodically."""
        last_save_time = time.time()
        pending_save = False
        
        while not self.stop_event.is_set() or pending_save:
            try:
                # Wait for save request with timeout
                self.save_queue.get(block=True, timeout=1)
                pending_save = True
                
                # Clear all pending save requests
                while not self.save_queue.empty():
                    try:
                        self.save_queue.get_nowait()
                    except:
                        pass
                
                # Save if enough time has passed or shutting down
                current_time = time.time()
                if (current_time - last_save_time > self.config.experiment.save_interval or 
                    self.stop_event.is_set()):
                    
                    res_df.to_csv(result_file, index=False)
                    last_save_time = current_time
                    pending_save = False
                    
            except:
                # Timeout, check if we should exit
                if self.stop_event.is_set() and not pending_save:
                    break
                # Continue loop
    
    def _calculate_results(
        self, 
        res_df: pd.DataFrame, 
        model_name: str, 
        track_name: str
    ) -> Dict[str, float]:
        """Calculate and return experiment results."""
        
        # Filter out error cases
        valid_results = res_df[res_df['success'] >= 0]
        
        if len(valid_results) == 0:
            return {
                "accuracy": 0.0,
                "total_samples": len(res_df),
                "completed_samples": 0,
                "total_cost": 0.0,
                "avg_tokens": 0.0
            }
        
        success_count = len(valid_results[valid_results['success'] == 1])
        accuracy = success_count / len(valid_results)
        total_cost = valid_results['cost'].sum()
        avg_tokens = valid_results['tokens'].mean()
        
        return {
            "accuracy": accuracy,
            "total_samples": len(res_df),
            "completed_samples": len(valid_results),
            "success_count": success_count,
            "total_cost": total_cost,
            "avg_tokens": avg_tokens
        }
    
    def _display_results(self, results: Dict[str, float]) -> None:
        """Display experiment results in a formatted table."""
        
        table = Table(title="Experiment Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in results.items():
            if isinstance(value, float):
                if metric == "accuracy":
                    table.add_row(metric, f"{value:.3f}")
                elif metric == "total_cost":
                    table.add_row(metric, f"${value:.4f}")
                else:
                    table.add_row(metric, f"{value:.2f}")
            else:
                table.add_row(metric, str(value))
        
        self.console.print(table)