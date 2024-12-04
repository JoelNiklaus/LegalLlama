# LegalLlama
This repository is used to fine-tune open LLMs on legal data

## Set Up

```bash
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
bash /opt/miniconda-installer.sh

conda create -n legalllama python=3.12
conda activate legalllama
pip install torch
pip install -r requirements.txt
```