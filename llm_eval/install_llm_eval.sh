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
# pip install --upgrade cuda-python && sudo reboot now # Do this if the current cuda version is outdated
pip install --upgrade pip setuptools wheel torch transformers datasets accelerate huggingface_hub[cli]
pip install pytest triton einops tiktoken sentencepiece protobuf flash-attn
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
mkdir output
