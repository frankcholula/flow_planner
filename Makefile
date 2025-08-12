SHELL := /bin/bash
ALGO ?= ppo
HF_ORG ?= frankcholula
WANDB_PROJECT ?= "Flow Planner"
LUNAR_ENV   := LunarLanderContinuous-v3
BIPEDAL_ENV := BipedalWalker-v3
CAR_ENV     := CarRacing-v3

ARCH := $(shell uname -m)
ifeq ($(ARCH), x86_64)
	SETUP_FILE := setup/environment_x86.yml
else ifeq ($(ARCH), arm64)
	SETUP_FILE := setup/environment_arm.yml
endif

.PHONY: help setup clean \
	train train-all train-lunar train-bipedal train-car \
	enjoy enjoy-lunar enjoy-bipedal enjoy-car \
	push-hub push-all push-lunar push-bipedal push-car \
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

push-hub:
	@echo "--> Pushing $(ALGO)-$(ENV) to $(HF_ORG)..."
	python -m rl_zoo3.push_to_hub --algo $(ALGO) --env $(ENV) -f logs/ -orga $(HF_ORG) --repo-name $(ALGO)-$(ENV)

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

# --- Hugging Face Hub ---
push-lunar:    ENV=$(LUNAR_ENV)
push-lunar:    push-hub

push-bipedal:  ENV=$(BIPEDAL_ENV)
push-bipedal:  push-hub

push-car:      ENV=$(CAR_ENV)
push-car:      push-hub

push-all: push-lunar push-bipedal push-car


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
