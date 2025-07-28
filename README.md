# Flow Planner 👾
Guided trajectory synthesis via flow matching.

## Setup
Create the conda environment and activate it:

```bash
make setup
conda activate ./.fp
```

## Makefile Targets
The Makefile in `src` exposes several commands:
- `make setup` – create a conda environment with the required dependencies.
- `make clean` – remove models, runs and temporary files.

### Redacted Commands
- `make fm` – run the flow-matching experiment (`experiments/fm.sh`).
- `make train` – train a PPO agent using `experiments/lunarlander/train.sh`.
- `make eval` – evaluate a trained model via `pipelines/lunarlander/eval.py`.
- `make collect_dataset` – generate a Minari dataset with `pipelines/lunarlander/collect_dataset.py`.
- `make baseline_bc` – train a simple behavioural cloning policy.
- `make fm_bc` – behavioural cloning using the flow-matching pipeline.

Run these commands from the `src` directory.

## Pipeline Scripts
The `src/pipelines/lunarlander` folder contains helper scripts:

- `collect_dataset.py` – loads a trained PPO agent and records episodes to build a Minari dataset.
- `baseline_bc.py` – trains a basic MLP policy on the collected dataset.
- `fm_bc.py` – behavioural cloning with flow matching.
- `eval.py` – evaluates a saved model on LunarLander and prints statistics.
- `preprocessing.py` – utilities for normalising datasets and creating training chunks.
