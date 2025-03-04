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

## Run

```bash
bash run_training.sh
```


## Citation

If you find this repository helpful, feel free to cite our publication [SwiLTra-Bench: The Swiss Legal Translation Benchmark](https://arxiv.org/abs/2503.01372):
```
@misc{niklaus2025swiltrabenchswisslegaltranslation,
      title={SwiLTra-Bench: The Swiss Legal Translation Benchmark}, 
      author={Joel Niklaus and Jakob Merane and Luka Nenadic and Sina Ahmadi and Yingqiang Gao and Cyrill A. H. Chevalley and Claude Humbel and Christophe Gösken and Lorenzo Tanzi and Thomas Lüthi and Stefan Palombo and Spencer Poff and Boling Yang and Nan Wu and Matthew Guillod and Robin Mamié and Daniel Brunner and Julio Pereyra and Niko Grupen},
      year={2025},
      eprint={2503.01372},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01372}, 
}
```
