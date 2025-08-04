"""Dataset preparation module for PersonaEval."""

import os
from pathlib import Path
from typing import List

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configuration
REPO_ID = "lingfengzhou/PersonaEval"
TARGET_DIR = "data"
REQUIRED_FILES = [
    "Literary.csv",
    "Drama.csv", 
    "Expertise.csv"
]


def check_dataset_files(target_dir: str = TARGET_DIR, 
                       required_files: List[str] = None) -> bool:
    """
    Check if all required dataset files exist locally.
    
    Args:
        target_dir: Local directory to check for files
        required_files: List of required filenames
        
    Returns:
        True if all files exist, False otherwise
    """
    if required_files is None:
        required_files = REQUIRED_FILES
    
    target_path = Path(target_dir)
    missing_files = []
    
    for filename in required_files:
        file_path = target_path / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files


def prepare_dataset(target_dir: str = TARGET_DIR,
                   repo_id: str = REPO_ID,
                   required_files: List[str] = None) -> bool:
    """
    Prepare dataset by checking local files and downloading missing ones from Hugging Face Hub.
    
    Args:
        target_dir: Local directory to store dataset files
        repo_id: Hugging Face repository ID
        required_files: List of required filenames
        
    Returns:
        True if all files are ready, False if any download failed
    """
    if required_files is None:
        required_files = REQUIRED_FILES
    
    if not HF_HUB_AVAILABLE:
        print("❌ huggingface_hub package not available. Please install with: pip install huggingface_hub")
        return False
    
    print("--- Checking local dataset files ---")
    
    # Ensure target directory exists
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    all_ready = True
    
    for filename in required_files:
        local_path = target_path / filename
        
        if local_path.exists():
            print(f"✅ File already exists: {local_path}")
        else:
            print(f"⏳ File missing: {local_path}. Downloading from Hugging Face Hub...")
            try:
                # Download file from Hugging Face Hub
                # hf_hub_download handles caching automatically
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(target_path),
                    local_dir_use_symlinks=False  # Recommended: copy files directly
                )
                print(f"✅ Download completed: {local_path}")
            except Exception as e:
                print(f"❌ Download failed: {filename}. Error: {e}")
                print("--- Please check repository ID and filenames, and verify network connection. ---")
                all_ready = False
    
    if all_ready:
        print("\n--- All dataset files are ready! ---")
    else:
        print("\n--- Some dataset files failed to download. ---")
    
    return all_ready


def ensure_dataset_ready() -> bool:
    """
    Ensure dataset is ready before running experiments.
    This is a convenience function for CLI usage.
    
    Returns:
        True if dataset is ready, False otherwise
    """
    files_exist, missing_files = check_dataset_files()
    
    if files_exist:
        print("✅ All required dataset files are present locally.")
        return True
    
    print(f"❌ Missing files: {missing_files}")
    print("Attempting to download missing files...")
    
    return prepare_dataset()