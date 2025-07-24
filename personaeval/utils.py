"""Utility functions for PersonaEval."""

import pandas as pd
from typing import Dict, List, Optional


def fix_res_df(res_df: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    """
    Fix result dataframe structure to match benchmark dataframe.
    
    This function is used when the result file structure doesn't match
    the current benchmark structure (e.g., when new samples are added).
    """
    new_res_df = df_bench.copy()
    
    # Initialize result columns
    result_columns = {
        'res': None,
        'response': None,
        'success': None,
        'cost': None,
        'tokens': None,
        'prob1': None,
        'prob2': None,
        'prob3': None,
        'prob4': None
    }
    
    for col, default_value in result_columns.items():
        new_res_df[col] = default_value
    
    # Copy existing results
    for index, row in res_df.iterrows():
        if pd.isna(row['res']) or pd.isna(row['success']):
            continue
        
        # Find matching row in new dataframe
        matching_rows = new_res_df[new_res_df['prompt'] == row['prompt']]
        if len(matching_rows) != 1:
            continue
        
        # Update the matching row
        idx = matching_rows.index[0]
        for col in result_columns.keys():
            if col in row and not pd.isna(row[col]):
                new_res_df.loc[idx, col] = row[col]
    
    return new_res_df


def calculate_statistics(res_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive statistics from result dataframe.
    
    Args:
        res_df: Result dataframe with evaluation results
        
    Returns:
        Dictionary with various statistics
    """
    # Filter out error cases
    valid_results = res_df.dropna(subset=['gt', 'success'])
    
    if len(valid_results) == 0:
        return {
            "accuracy": 0.0,
            "total_samples": len(res_df),
            "completed_samples": 0,
            "success_count": 0,
            "failure_count": 0,
            "error_count": len(res_df),
            "total_cost": 0.0,
            "avg_tokens": 0.0,
            "completion_rate": 0.0
        }
    
    success_count = len(valid_results[valid_results['success'] == 1])
    failure_count = len(valid_results[valid_results['success'] == 0])
    error_count = len(res_df) - len(valid_results)
    
    accuracy = success_count / len(valid_results)
    completion_rate = len(valid_results) / len(res_df)
    total_cost = valid_results['cost'].sum()
    avg_tokens = valid_results['tokens'].mean()
    
    return {
        "accuracy": accuracy,
        "total_samples": len(res_df),
        "completed_samples": len(valid_results),
        "success_count": success_count,
        "failure_count": failure_count,
        "error_count": error_count,
        "total_cost": total_cost,
        "avg_tokens": avg_tokens,
        "completion_rate": completion_rate
    }


def create_safe_model_name(model_name: str) -> str:
    """Convert model name to safe filename."""
    return model_name.replace('/', '-').replace(':', '-').replace(' ', '_') 