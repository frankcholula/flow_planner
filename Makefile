-include .env
export HF_TOKEN
export MINARI_REMOTE

SHELL := /bin/bash
ALGO ?= ppo
HF_ORG ?= frankcholula
WANDB_PROJECT ?= "Flow Planner"
LUNAR_ENV   := LunarLanderContinuous-v3
BIPEDAL_ENV := BipedalWalker-v3
CAR_ENV     := CarRacing-v3

# Dataset specific params
LEVEL ?= expert


ARCH := $(shell uname -m)
ifeq ($(ARCH), x86_64)
	SETUP_FILE := setup/environment_x86.yml
else ifeq ($(ARCH), arm64)
	SETUP_FILE := setup/environment_arm.yml
endif

	
.PHONY: help setup clean \
	train train-all train-lunar train-bipedal train-car \
	enjoy enjoy-lunar enjoy-bipedal enjoy-car \
	push-model push-all-models push-car-model push-bipedal-model push-lunar-model \
	download-datasets download-mujoco download-kitchen

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target> [VAR=value]"
	@echo "Example: make train-car ALGO=a2c"
	@echo ""
	@echo "------------------ Environment Management ------------------"
	@echo "  setup              Create the Conda environment from file"
	@echo "  clean              Remove generated logs, models, and data"
	@echo ""
	@echo "------------------ Training & Evaluation ------------------"
	@echo "  train-all          Train all agents sequentially"
	@echo "  train-lunar        Train the LunarLander agent"
	@echo "  train-bipedal      Train the BipedalWalker agent"
	@echo "  train-car          Train the CarRacing agent"
	@echo "  enjoy-[lunar|bipedal|car] Watch a trained agent"
	@echo ""
	@echo "------------------ Hugging Face Hub -----------------------"
	@echo "  push-all           Push all models to the Hub"
	@echo "  push-[lunar|bipedal|car]  Push a specific model to the Hub"
	@echo ""
	@echo "------------------ Dataset Management ---------------------"
	@echo "  download-datasets  Download all external datasets"

train:
	@echo "Training $(ENV) with $(ALGO)..."
	python -m rl_zoo3.train --algo $(ALGO) --env $(ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG)

enjoy:
	@echo "Watching $(ENV) agent..."
	python -m rl_zoo3.enjoy --algo $(ALGO) --env $(ENV) -f logs/ --load-best

push-model:
	@echo "--> Pushing $(ALGO)-$(ENV) to $(HF_ORG)..."
	python -m rl_zoo3.push_to_hub --algo $(ALGO) --env $(ENV) -f logs/ -orga $(HF_ORG) --repo-name $(ALGO)-$(ENV)

push-dataset:
	@echo "--> Collecting $(LEVEL) dataset for $(ENV)..."
	python -m src.pipelines.collect_dataset \
		   --env $(ENV) \
		   --total_episodes 1_000 \
		   --seed 42 \
		   --algo $(ALGO) \
		   --version 0 \
		   --level $(LEVEL)
# 	@echo "--> Pushing dataset $(ENV) to the Hub..."
# 	@minari upload Box2D/$(ENV)/$(LEVEL)-v0 --key-path $(HF_TOKEN)

# --- Environment Management ---
setup:
	@echo "Setting up environment for $(ARCH)..."
	conda env create -f $(SETUP_FILE) --prefix ./.fp

clean:
	@echo "Cleaning up generated files..."
	rm -rf videos/ runs/ wandb/* logs/*

# --- Training ---
train-lunar:   ENV=$(LUNAR_ENV)
train-lunar:   train

train-bipedal: ENV=$(BIPEDAL_ENV)
train-bipedal: train

train-car:     ENV=$(CAR_ENV)
train-car:     train

train-all: train-lunar train-bipedal train-car

# --- Evaluation ---
enjoy-lunar:   ENV=$(LUNAR_ENV)
enjoy-lunar:   enjoy

enjoy-bipedal: ENV=$(BIPEDAL_ENV)
enjoy-bipedal: enjoy

enjoy-car:     ENV=$(CAR_ENV)
enjoy-car:     enjoy

# --- Model pushing ---
push-lunar-model:    ENV=$(LUNAR_ENV)
push-lunar-model:    push-model


push-bipedal-model:  ENV=$(BIPEDAL_ENV)
push-bipedal-model:  push-model

push-car-model:      ENV=$(CAR_ENV)
push-car-model:      push-model

push-all-models: push-lunar-model push-bipedal-model push-car-model


# --- Dataset pushing ---
push-lunar-dataset:    ENV=$(LUNAR_ENV)
push-lunar-dataset:    push-dataset

push-bipedal-dataset:  ENV=$(BIPEDAL_ENV)
push-bipedal-dataset:  push-dataset

push-car-dataset:      ENV=$(CAR_ENV)
push-car-dataset:      push-dataset

push-all-datasets: push-lunar-dataset push-bipedal-dataset push-car-dataset



# --- Dataset Download ---
list:
	minari list remote
	
download-box2d:
	@echo "Downloading Box2D datasets..."
	minari download Box2D/$(ENV)/$(LEVEL)-v0

download-agents:
	@echo "Downloading trained-agents..."
# 	python -m rl_zoo3.load_from_hub --algo ppo --env LunarLanderContinuous-v3 -orga frankcholula -f logs/
# 	python -m rl_zoo3.load_from_hub --algo ppo --env CarRacing-v3 -orga frankcholula -f logs/
# 	python -m rl_zoo3.load_from_hub --algo ppo_lstm --env CarRacing-v3 -orga frankcholula -f logs/
# 	python -m rl_zoo3.load_from_hub --algo ppo --env BipedalWalker-v3 -orga frankcholula -f logs/
