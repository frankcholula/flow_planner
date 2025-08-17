-include .env

export HF_TOKEN
export MINARI_REMOTE

SHELL := /bin/bash
ALGO ?= ppo
HF_ORG ?= frankcholula
WANDB_PROJECT ?= "Flow Planner"
CATEGORY ?= Box2D
LUNAR_ENV := LunarLanderContinuous-v3
BIPEDAL_ENV := BipedalWalker-v3
CAR_ENV := CarRacing-v3
MOUNTAIN_ENV := MountainCarContinuous-v0
TOTAL_EPISODES := 500
LEVEL ?= expert
ARCH := $(shell uname -m)

ifeq ($(ARCH), x86_64)
	SETUP_FILE := setup/environment_x86.yml
else ifeq ($(ARCH), arm64)
	SETUP_FILE := setup/environment_arm.yml
endif

.PHONY: clean help setup \
	train-all train-lunar train-bipedal train-car \
	enjoy enjoy-lunar enjoy-bipedal enjoy-car enjoy-mountain \
	push-model push-all-models push-car-model push-bipedal-model push-lunar-model push-mountain-model \
	generate-dataset generate-all-datasets \
	push-datasets push-all-datasets push-lunar-dataset push-bipedal-dataset push-car-dataset push-mountain-dataset \
	download-datasets download-Box2D download-ClassicControl

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
	@echo "------------------ Dataset & Agent Management ---------------------"
	@echo "  generate_dataset   Generate a dataset for the specified environment"
	@echo "  download-Box2D  Download all Box2D datasets"
	@echo "  download-agents    Download all trained agents"
	@echo "  list-datasets	    List datasets from the Minari remote"
# --- Environment Management ---
setup:
	@echo "Setting up environment for $(ARCH)..."
	conda env create -f $(SETUP_FILE) --prefix ./.fp

clean:
	@echo "Cleaning up generated files..."
	rm -rf videos/ runs/ wandb/* logs/*

# --- Training ---
train-lunar:   ENV=$(LUNAR_ENV)
train-lunar:
	@echo "Training $(ENV) with $(ALGO)..."
	python -m rl_zoo3.train --algo $(ALGO) --env $(ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG) --device cpu

train-bipedal: ENV=$(BIPEDAL_ENV)
train-bipedal:
	@echo "Training $(ENV) with $(ALGO)..."
	python -m rl_zoo3.train --algo $(ALGO) --env $(ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG) --device cpu

train-car:     ENV=$(CAR_ENV)
train-car:
	@echo "Training $(ENV) with $(ALGO)..."
	python -m rl_zoo3.train --algo $(ALGO) --env $(ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG)

train-mountain: ENV=$(MOUNTAIN_ENV)
train-mountain:
	@echo "Training $(ENV) with $(ALGO)..."
	python -m rl_zoo3.train --algo $(ALGO) --env $(ENV) --track --wandb-project-name $(WANDB_PROJECT) --wandb-entity $(HF_ORG) --device cpu

train-all: train-lunar train-bipedal train-car train-mountain

# --- Evaluation ---
enjoy:
	@echo "Watching $(ENV) agent..."
	python -m rl_zoo3.enjoy --algo $(ALGO) --env $(ENV) -f logs/

enjoy-lunar:   ENV=$(LUNAR_ENV)
enjoy-lunar: enjoy

enjoy-bipedal: ENV=$(BIPEDAL_ENV)
enjoy-bipedal: enjoy

enjoy-car:     ENV=$(CAR_ENV)
enjoy-car: enjoy

enjoy-mountain: ENV=$(MOUNTAIN_ENV)
enjoy-mountain: enjoy

# --- Model pushing ---
push-model:
	@echo "--> Pushing $(ALGO)-$(ENV) to $(HF_ORG)..."
	python -m rl_zoo3.push_to_hub --algo $(ALGO) --env $(ENV) -f logs/ -orga $(HF_ORG) --repo-name $(ALGO)-$(ENV)

push-lunar-model:    ENV=$(LUNAR_ENV)
push-lunar-model: push-model

push-bipedal-model:  ENV=$(BIPEDAL_ENV)
push-bipedal-model: push-model

push-car-model:      ENV=$(CAR_ENV)
push-car-model: push-model

push-mountain-model: ENV = $(MOUNTAIN_ENV)
push-mountain-model: push-model

push-all-models: push-lunar-model push-bipedal-model push-car-model push-mountain-model

# --- Dataset generation ---
generate-dataset:
	@echo "--> Generating $(TOTAL_EPISODES) of $(LEVEL) dataset for $(ENV)..."
	python -m src.pipelines.collect_dataset \
		   --env $(ENV) \
		   --total_episodes $(TOTAL_EPISODES) \
		   --seed 42 \
		   --algo $(ALGO) \
		   --version 0 \
		   --level $(LEVEL)

generate-lunar-dataset: ENV=$(LUNAR_ENV)
generate-lunar-dataset: generate-dataset

generate-bipedal-dataset: ENV=$(BIPEDAL_ENV)
generate-bipedal-dataset: generate-dataset

generate-vae-training-dataset:
	@echo "--> Generating VAE training dataset for CarRacing using random policy..."
	python -m src.pipelines.carracing.random_policy
	@echo "--> Mixing random dataset with expert dataset..."
	python -m src.pipelines.carracing.mix_datasets

mix-car-dataset:
	@echo "--> Mixing CarRacing datasets..."
	minari combine Box2D/CarRacing-v3/expert-v0 Box2D/CarRacing-v3/simple-v0 Box2D/CarRacing-v3/mixed-v0

generate-car-dataset: ENV=$(CAR_ENV)
generate-car-dataset: generate-dataset mix-car-dataset generate-vae-training-dataset

generate-mountain-dataset: ENV=$(MOUNTAIN_ENV)
generate-mountain-dataset: generate-dataset

generate-all-datasets: generate-lunar-dataset generate-bipedal-dataset generate-car-dataset generate-mountain-dataset
# --- Dataset pushing ---
push-dataset:
	@echo "--> Pushing $(ENV) to the $(CATEGORY) category..."
	@minari upload $(CATEGORY)/$(ENV)/$(LEVEL)-v0 --key-path $(HF_TOKEN)

push-lunar-dataset:
	@$(MAKE) push-dataset ENV=$(LUNAR_ENV)

push-bipedal-dataset:
	@$(MAKE) push-dataset ENV=$(BIPEDAL_ENV)

push-car-dataset:
	@$(MAKE) push-dataset ENV=$(CAR_ENV)

push-mountain-dataset:
	@$(MAKE) push-dataset ENV=$(MOUNTAIN_ENV) CATEGORY=ClassicControl

push-all-datasets: push-lunar-dataset push-bipedal-dataset push-car-dataset push-mountain-dataset

# --- Dataset Download ---
list-datasets:
	@echo "Listing dataset from $(MINARI_REMOTE)..."
	@minari list remote

download-ClassicControl: CATEGORY=ClassicControl
download-ClassicControl:
	@echo "Downloading all $(CATEGORY) datasets..."
	minari download $(CATEGORY)/$(MOUNTAIN_ENV)/$(LEVEL)-v0

download-Box2D: CATEGORY=Box2D
download-Box2D:
	@echo "Downloading all $(CATEGORY) datasets..."
	minari download $(CATEGORY)/$(LUNAR_ENV)/expert-v0
	minari download $(CATEGORY)/$(BIPEDAL_ENV)/expert-v0
	minari download $(CATEGORY)/$(CAR_ENV)/expert-v0
	minari download $(CATEGORY)/$(CAR_ENV)/mixed-v0
	minari download $(CATEGORY)/$(CAR_ENV)/simple-v0

download-agents:
	@echo "Downloading trained-agents..."
	python -m rl_zoo3.load_from_hub --algo ${ALGO} --env ${LUNAR_ENV} -orga frankcholula -f logs/
	python -m rl_zoo3.load_from_hub --algo ${ALGO} --env ${BIPEDAL_ENV} -orga frankcholula -f logs/
	python -m rl_zoo3.load_from_hub --algo ${ALGO} --env ${CAR_ENV} -orga frankcholula -f logs/
	python -m rl_zoo3.load_from_hub --algo ${ALGO} --env ${MOUNTAIN_ENV} -orga frankcholula -f logs/
