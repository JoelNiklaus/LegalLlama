activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}

#sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
mkdir ~/llm_eval
cd ~/llm_eval
python3 -m venv .venv
activate
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
mkdir output
# Probably loop through all the tasks and models
lm_eval --model hf --model_args pretrained=tiiuae/falcon-11B --tasks hellaswag --batch_size 1 --output_path ./output --num_fewshot 10
