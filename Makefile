
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

.PHONY: train_zoo
train_zoo:
	@echo "Running RL Zoo Training..."
	@src/experiments/lunarlander/train_zoo.sh
	@src/experiments/bipedalwalker/train_zoo.sh
	@src/experiments/carracing/train_zoo.sh

.PHONY: load_pretrained
load_pretrained:
	@echo "Downloading pretrained models from Hugging Face..."
# 	python -m rl_zoo3.load_from_hub --algo ppo --env BipedalWalker-v3 -orga frankcholula -f logs/
	python -m rl_zoo3.load_from_hub --algo ppo --env CarRacing-v3 -orga frankcholula -f logs/
# 	python -m rl_zoo3.load_from_hub --algo ppo --env LunarLanderContinuous-v3 -orga frankcholula -f logs/

.PHONY: enjoy
enjoy:
# 	python -m rl_zoo3.enjoy --algo ppo --env BipedalWalker-v3 -f logs/
	python -m rl_zoo3.enjoy --algo ppo --env CarRacing-v3 -f logs/
# 	python -m rl_zoo3.enjoy --algo ppo --env LunarLanderContinuous-v3 -f logs/

.PHONY: push_to_hub
push_to_hub:
	@echo "Pushing models to Hugging Face Hub..."
	python -m rl_zoo3.push_to_hub --algo ppo --env BipedalWalker-v3 -f logs/ -orga frankcholula --repo-name ppo-BipedalWalker-v3
	python -m rl_zoo3.push_to_hub --algo ppo --env CarRacing-v3 -f logs/ -orga frankcholula --repo-name ppo-CarRacing-v3
	python -m rl_zoo3.push_to_hub --algo ppo --env LunarLanderContinuous-v3 -f logs/ -orga frankcholula --repo-name ppo-LunarLanderContinuous-v3

.PHONY: lunarlander
lunarlander:
	@echo "Running LunarLander Experiments..."
	@echo "Conditional Flow Matching on Starting Observations..."
	@src/experiments/lunarlander/cfm_start_obs.sh

# Dataset targets
.PHONY: datasets
datasets: box2d_data

.PHONY: box2d_data
box2d_data:
	@echo "Generating Box2D dataset..."

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
	@echo "Cleaning up video files..."
	rm -rf videos/*
	@echo "Cleaning up experiment runs..."
	rm -rf runs/*
	@echo "Cleaning up wandb files..."
	rm -rf wandb/*
	@echo "Cleaning up pretrained models..."
	rm -rf logs/*
