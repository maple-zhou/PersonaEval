"""Configuration management for PersonaEval."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class TrackConfig(BaseModel):
    """Configuration for a single evaluation track."""
    
    name: str = Field(..., description="Track name")
    data_file: str = Field(..., description="Path to the CSV data file")
    output_dir: str = Field(..., description="Directory to save results")
    
    @validator('data_file')
    def validate_data_file(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Data file does not exist: {v}")
        return v
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    
    url: str = Field(..., description="API endpoint URL")
    api_key: str = Field(..., description="API key")
    cost_input: float = Field(..., description="Cost per input token")
    cost_output: float = Field(..., description="Cost per output token")
    # Simple proxy configuration
    proxy_url: Optional[str] = Field(
        default=None, 
        description="Proxy URL for API requests (e.g., http://proxy:port)"
    )
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("Please provide a valid API key")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for experiment settings."""
    
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    max_retries: int = Field(default=5, description="Maximum number of retries per request")
    timeout: int = Field(default=600, description="Request timeout in seconds")
    temperature: float = Field(default=0.0, description="Model temperature")
    save_interval: int = Field(default=60, description="Save interval for result file in seconds")
    sleep_interval: int = Field(default=60, description="Sleep interval during retry in seconds")
    reasoning_models: Optional[List[str]] = Field(
        default=None, 
        description="List of models that support reasoning"
    )


class Config(BaseModel):
    """Main configuration class."""
    
    tracks: List[TrackConfig] = Field(..., description="List of evaluation tracks")
    models: Dict[str, ModelConfig] = Field(..., description="Model configurations")
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def get_track(self, track_name: str) -> Optional[TrackConfig]:
        """Get a specific track configuration by name."""
        for track in self.tracks:
            if track.name == track_name:
                return track
        return None
    
    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get a specific model configuration by name."""
        return self.models.get(model_name)
    
    def list_tracks(self) -> List[str]:
        """List all available track names."""
        return [track.name for track in self.tracks]
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys()) 