
ARCH := $(shell arch)
ifeq ($(ARCH), x86_64)
	SETUP_FILE := setup/environment_x86.yml
else ifeq ($(ARCH), arm64)
	SETUP_FILE := setup/environment_arm.yml
endif

.PHONY: setup
setup:
	@echo "Setting up ${ARCH} environment..."
	conda env create -f $(SETUP_FILE) --prefix ./.fp

.PHONY: lunarlander
lunarlander:
	@echo "Running LunarLander Experiments..."
	@echo "Conditional Flow Matching on Starting Observations..."
	@src/experiments/lunarlander/cfm_start_obs.sh

.PHONY: train
train:
	@echo "Running training script..."
	@src/experiments/lunarlander/train.sh

.PHONY: eval
eval:
	@echo "Running evaluation script..."
	@python src/pipelines/eval.py

.PHONY: collect_dataset
collect_dataset:
	python src/pipelines/collect_dataset.py

.PHONY: baseline_bc
baseline_bc:
	python src/pipelines/baseline_bc.py

.PHONY: fm_bc
fm_bc:
	python src/pipelines/fm_bc.py

.PHONY: mujoco_data
mujoco_data:
	@echo "Downloading MuJoCo hopper and walker2d..."
	minari download mujoco/hopper/expert-v0
	minari download mujoco/walker2d/expert-v0



.PHONY: kitchen_data
kitchen_data:
	@echo "Downloading kitchen dataset..."
	minari download D4RL/kitchen/partial-v2
	minari download D4RL/kitchen/complete-v2
	minari download D4RL/kitchen/mixed-v2


.PHONY: clean
clean:
	rm -rf videos/*
	rm -rf runs/*
	rm -rf wandb/*
