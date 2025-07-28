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
- `make datasets` – download and prepare all datasets.
Run these commands from the `src` directory.
