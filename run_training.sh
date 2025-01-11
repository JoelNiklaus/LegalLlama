#!/bin/bash

# Make sure the ephemeral disk is writable
sudo chmod 777 /ephemeral

# Load environment variables
set -a
source .env
set +a

# See available models at https://huggingface.co/unsloth

# For some reason, gemma-2-27b-it yields NaN in the grad_norm, so we don't use it.
# MODELS=("google/madlad400-10b-mt" "facebook/seamless-m4t-v2-large") # These are not yet supported by unsloth

# command-r-plus does not work so far: NameError: name 'CohereLayerNorm' is not defined
# The other model trains, but then at inference has errors
# Cohere is not supported by unsloth yet: https://huggingface.co/unsloth/c4ai-command-r-plus-08-2024-bnb-4bit/discussions/1
# For some reason, the loss is extremely high. Check again and abort if it is not fixed.
MODELS=("c4ai-command-r-08-2024" "c4ai-command-r-plus-08-2024" "gemma-2-27b-it")

MODELS_DONE=("unsloth/phi-4" "Unbabel/TowerInstruct-13B-v0.1" "Unbabel/TowerInstruct-7B-v0.2" "gemma-2-2b-it" "gemma-2-9b-it" "Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Meta-Llama-3.1-8B-Instruct" "Meta-Llama-3.1-70B-Instruct" "Llama-3.3-70B-Instruct" "Phi-3.5-mini-instruct" "Phi-3-medium-4k-instruct" "mistral-7b-instruct-v0.3" "Qwen2.5-72B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-14B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-3B-Instruct" "Qwen2.5-1.5B-Instruct" "Qwen2.5-0.5B-Instruct")

for model in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name $model --batch_size 128 --learning_rate 1e-4 --lora_rank 16 --num_epochs 5 --cpus 30 --push_to_hub
done




