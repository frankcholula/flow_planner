# Flow Planner 👾
Guided trajectory synthesis via flow matching.

## Environments
This repository runs on all the Box2D environments.

[](LunarLander)
[](BipedalWalker)
[](CarRacing)


## Repository Layout
```
flow_planner
├── LICENSE
├── Makefile
├── paper
├── README.md
├── setup
│   ├── environment_arm.yml
│   └── environment_x86.yml
├── src
│   ├── checkpoints
│   ├── conf
│   ├── experiments
│   ├── models
│   ├── notebooks
│   ├── pipelines
│   ├── run.py
│   └── utils
└── wandb
```

## Setup
Create the conda environment and activate it:

```bash
make setup
conda activate ./.fp
```

## Generating Datasets
Datasets are stored locally in [Minari](https://minari.farama.org/index.html) and generated with different RL algorithms based on the environment.
There is an ongoing ticket to implement a GCS bucket to store all these datasets. For now, you'll have to run the generate script in order to create the dataset.
- LunarLander
- BipedalWalker
- CarRacing

## Makefile Targets
The Makefile in `src` exposes several commands:
- `make setup` – create a conda environment with the required dependencies.
- `make clean` – remove models, runs and temporary files.
- `make datasets` – download and prepare all datasets.
Run these commands from the `src` directory.


