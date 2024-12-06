#!/bin/bash

# Make sure the ephemeral disk is writable
sudo chmod 777 /ephemeral

# Load environment variables
set -a
source .env
set +a

# See available models at https://huggingface.co/unsloth

# Batch sizes on an 80GB H100 with max_seq_length=2048:
# unsloth/Phi-3.5-mini-instruct-bnb-4bit: 128 uses 59GB VRAM
# unsloth/Qwen2.5-7B-Instruct-bnb-4bit: 32 uses 66GB VRAM
# unsloth/Qwen2.5-14B-Instruct-bnb-4bit: 32 uses 72GB VRAM
# unsloth/gemma-2-27b-bnb-4bit: 16 uses 67GB VRAM

# Batch sizes on an 80GB H100 with max_seq_length=512:
# Llama-3.2-1B-Instruct: 128 uses 51GB VRAM
# Qwen2.5-7B-Instruct: 128 uses 66GB VRAM
# Qwen2.5-14B-Instruct: 128 uses 74GB VRAM,
# Qwen2.5-0.5B-Instruct: 128 uses 59GB VRAM, 64 uses 30GB VRAM

# GPU 0: running 32B Qwen2.5-Instruct then gemma-2-27b-it then gemma-2-2b-it then gemma-2-9b-it
# GPU 1: running 70B Llama-3.1-Instruct

MODELS=("Qwen2.5-32B-Instruct" "gemma-2-27b-it" "gemma-2-2b-it" "gemma-2-9b-it")
MODELS=("Meta-Llama-3.1-70B-Instruct")
    
MODELS_DONE=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Meta-Llama-3.1-8B-Instruct" "Phi-3.5-mini-instruct" "Phi-3-medium-4k-instruct" "mistral-7b-instruct-v0.3" "Qwen2.5-72B-Instruct" "Qwen2.5-14B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-3B-Instruct" "Qwen2.5-1.5B-Instruct" "Qwen2.5-0.5B-Instruct")

for model in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name $model --batch_size 128 --learning_rate 1e-4 --lora_rank 16 --num_epochs 5 --cpus 30 --push_to_hub
done




