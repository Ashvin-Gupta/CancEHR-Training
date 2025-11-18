#!/bin/bash

# Shell script to run LLM classification fine-tuning
# Usage: bash run_text_llm_classifier.sh

# Activate conda environment if needed
# conda activate your_env_name

# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0

# Path to config file
CONFIG_PATH="src/pipelines/text_based/configs/llm_finetune_classifier.yaml"

# Run the fine-tuning script
python -m src.pipelines.text_based.finetune_llm_classifier \
    --config_filepath "$CONFIG_PATH"

echo "Classification fine-tuning complete!"


