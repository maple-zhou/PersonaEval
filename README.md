# PersonaEval

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://github.com/maple-zhou/PersonaEval)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](https://huggingface.co/datasets/lingfengzhou/PersonaEval)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is the official codebase for the paper **PersonaEval: Are LLM Evaluators Human Enough to Judge Role-Play?**.

## Installation

We use [`uv`](https://github.com/astral-sh/uv) for environment management. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

## Usage

### Step 0: Dataset Preparation

**The evaluation datasets will be automatically downloaded when you run any command.** The system will:

1. Check if required data files exist in the `data/` directory
2. If files are missing, automatically download them from [Hugging Face Hub](https://huggingface.co/datasets/lingfengzhou/PersonaEval)
3. Required files: `Literary.csv`, `Drama.csv`, `Expertise.csv`

You can also manually prepare the dataset by running:

```bash
python -c "from personaeval.dataset_prepare import ensure_dataset_ready; ensure_dataset_ready()"
```

### Step 1: Configuration Setup

Copy the example configuration file and modify it:

```bash
cp configs/default.yaml.example configs/default.yaml
```

Edit `configs/default.yaml` to configure your models, including base_url, api_key, etc.

### Step 2: Run Experiments

The simplest way to start is:

```bash
personaeval run --model gpt-4.1
```

This will run the evaluation on all tracks with the specified model.

#### Command Line Arguments

- `--model, -m`: Model name to evaluate (required)
- `--track, -t`: Track name to evaluate (default: "all")
- `--config, -c`: Path to configuration file (default: "configs/default.yaml")
- `--no-resume`: Do not resume from existing results
- `--list-tracks`: List available tracks and exit
- `--list-models`: List available models and exit

#### Examples

```bash
# Run on a specific track
personaeval run --track Literary --model gpt-4.1

# Run on all tracks
personaeval run --track all --model claude-sonnet-4-20250514

# Use custom configuration
personaeval run --config configs/my_config.yaml --model gpt-4.1

# List available tracks and models
personaeval run list-tracks
personaeval run list-models
```

### Step 3: Calculate Metrics

After running experiments, calculate evaluation metrics:

```bash
# Calculate metrics for a single model
personaeval metrics --models gpt-4.1

# Calculate metrics for multiple models
personaeval metrics --models "gpt-4.1,claude-sonnet-4-20250514"

# Generate comparison plots
personaeval metrics --models "gpt-4.1,claude-sonnet-4-20250514" --plot

# Specify custom output
personaeval metrics --models gpt-4.1 --output my_metrics.csv --plot --plot-output comparison.png
```

## Configuration Parameters

### Model Configuration

Each model in the configuration file supports:

- `url`: API endpoint URL
- `api_key`: Your API key
- `cost_input`: Cost per input token
- `cost_output`: Cost per output token
- `proxy_url`: Optional proxy URL for API requests

### Experiment Configuration

- `max_workers`: Number of concurrent workers (default: 4)
- `max_retries`: Maximum retries per request (default: 5)
- `timeout`: Request timeout in seconds (default: 600)
- `temperature`: Model temperature (default: 0.0)
- `save_interval`: Save interval for result file in seconds (default: 60)
- `sleep_interval`: Sleep interval during retry (default: 60)
- `reasoning_models`: List of models that require stream mode for API call

### Track Configuration

Each track defines:

- `name`: Track name (Literary, Drama, Expertise)
- `data_file`: Path to CSV data file
- `output_dir`: Directory to save results

## Available Commands

### `personaeval run`

Run evaluation experiments on specified tracks and models.

**Options:**
- `--config, -c`: Configuration file path
- `--track, -t`: Track name or "all"
- `--model, -m`: Model name (required)
- `--no-resume`: Disable resume from existing results
- `--list-tracks`: List available tracks
- `--list-models`: List available models

### `personaeval metrics`

Calculate evaluation metrics from experiment results.

**Options:**
- `--results-dir, -r`: Results directory (default: "results")
- `--models, -m`: Comma-separated list of model names
- `--tracks`: Comma-separated list of track names (default: "Literary,Drama,Expertise")
- `--output, -o`: Output CSV file path (default: "metrics.csv")
- `--plot`: Generate comparison plots
- `--plot-output`: Plot output file path (default: "metrics_comparison.png")

### `personaeval analyze`

Analyze a single result file.

**Arguments:**
- `result_file`: Path to the result CSV file

## Project Structure

```
personaeval/
├── personaeval/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── models.py           # Model definitions and API calls
│   ├── evaluator.py        # Main evaluation logic
│   ├── metrics.py          # Metrics calculation
│   ├── dataset_prepare.py  # Dataset preparation utilities
│   └── utils.py            # Utility functions
├── configs/
│   ├── default.yaml        # Main configuration
│   └── default.yaml.example # Example configuration
├── data/                   # Evaluation datasets (auto-downloaded)
├── results/                # Experiment results
├── pyproject.toml
└── README.md
```

## Citation

If you find this code useful, please cite our paper:

<!-- ```bibtex
@article{zhou2024personaeval,
  title={PersonaEval: Are LLM Evaluators Human Enough to Judge Role-Play?},
  author={Zhou, Lingfeng and others},
  journal={arXiv preprint},
  year={2024}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.