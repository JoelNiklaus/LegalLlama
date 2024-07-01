#!/bin/bash

activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
mkdir -p ~/llm_eval
cd ~/llm_eval
python3 -m venv .venv
activate
pip install --upgrade pip setuptools wheel cuda-python torch transformers datasets
pip install pytest triton einops tiktoken sentencepiece huggingface_hub[cli] accelerate protobuf flash-attn
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
mkdir output
