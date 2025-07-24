"""PersonaEval: A comprehensive benchmark for persona evaluation in language models."""

__version__ = "0.1.0"
__author__ = "Lingfeng Zhou"
__email__ = "zhoulingfeng@sjtu.edu.cn"

from .config import Config, TrackConfig, ModelConfig, ExperimentConfig
from .evaluator import Evaluator
from .models import ModelManager

__all__ = [
    "Config",
    "TrackConfig", 
    "ModelConfig",
    "ExperimentConfig",
    "Evaluator",
    "ModelManager",
] 