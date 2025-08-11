SHELL := /bin/bash

# --- Variables ---
ARCH         := $(shell uname -m)
HF_ORG       := frankcholula
WANDB_PROJECT := "Flow Planner"
LUNAR_ENV     := LunarLanderContinuous-v3
BIPEDAL_ENV   := BipedalWalker-v3
CAR_ENV       := CarRacing-v2

ifeq ($(ARCH), x86_64)
	SETUP_FILE := setup/environment_x86.yml
else ifeq ($(ARCH), arm64)
	SETUP_FILE := setup/environment_arm.yml
endif

# Default target when running just "make" is to show help
.DEFAULT_GOAL := help

.PHONY: help setup clean \
	train-lunar train-bipedal train-car \
	enjoy-lunar enjoy-bipedal enjoy-car \
	push-lunar push-bipedal push-car \
	download-datasets download-mujoco download-kitchen


help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Common Targets:"
	@echo "  setup              Create the Conda environment"
	@echo "  train-lunar        Train the LunarLander agent"
	@echo "  train-bipedal      Train the BipedalWalker agent"
	@echo "  train-car          Train the CarRacing agent"
	@echo "  enjoy-bipedal      Watch the trained BipedalWalker agent"
	@echo "  push-car           Push the CarRacing model to the Hub"
	@echo "  download-datasets  Download all external datasets"
	@echo "  clean              Remove generated files and directories"

# --- Environment Management ---
setup:
	@echo "Setting up environment for $(ARCH)..."
	conda env create -f $(SETUP_FILE) --prefix ./.fp

clean:
	@echo "Cleaning up generated files..."
	rm -rf videos/ runs/ wandb/* logs/*

# --- Training ---
train-lunar:
	@echo "Training LunarLander..."
	python -m rl_zoo3.train --algo ppo --env $(LUNAR_ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG)

train-bipedal:
	@echo "Training BipedalWalker..."
	python -m rl_zoo3.train --algo ppo --env $(BIPEDAL_ENV) --device cpu --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG)

train-car:
	@echo "Training CarRacing..."
	python -m rl_zoo3.train --algo ppo --env $(CAR_ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG)

# --- Evaluation ---
enjoy-lunar:
	python -m rl_zoo3.enjoy --algo ppo --env $(LUNAR_ENV) -f logs/

enjoy-bipedal:
	python -m rl_zoo3.enjoy --algo ppo --env $(BIPEDAL_ENV) -f logs/

enjoy-car:
	python -m rl_zoo3.enjoy --algo ppo --env $(CAR_ENV) -f logs/

# --- Hugging Face Hub ---
push-lunar:
	@echo "--> Pushing $(LUNAR_ENV) to Hub..."
	python -m rl_zoo3.push_to_hub --algo ppo --env $(LUNAR_ENV) -f logs/ -orga $(HF_ORG) --repo-name ppo-$(LUNAR_ENV)

push-bipedal:
	@echo "--> Pushing $(BIPEDAL_ENV) to Hub..."
	python -m rl_zoo3.push_to_hub --algo ppo --env $(BIPEDAL_ENV) -f logs/ -orga $(HF_ORG) --repo-name ppo-$(BIPEDAL_ENV)

push-car:
	@echo "--> Pushing $(CAR_ENV) to Hub..."
	python -m rl_zoo3.push_to_hub --algo ppo --env $(CAR_ENV) -f logs/ -orga $(HF_ORG) --repo-name ppo-$(CAR_ENV)

# --- Dataset Download ---
download-mujoco:
	@echo "Downloading MuJoCo datasets..."
	minari download mujoco/hopper/expert-v0
	minari download mujoco/walker2d/expert-v0

download-kitchen:
	@echo "Downloading kitchen datasets..."
	minari download D4RL/kitchen/partial-v2
	minari download D4RL/kitchen/complete-v2
	minari download D4RL/kitchen/mixed-v2

download-datasets: download-mujoco download-kitchen
