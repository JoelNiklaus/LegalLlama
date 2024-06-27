activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
sudo apt install ninja-build # for Flash attention
mkdir ~/llm_eval
cd ~/llm_eval
python3 -m venv .venv
activate
pip install --upgrade pip setuptools wheel
pip install packaging
pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
mkdir output
