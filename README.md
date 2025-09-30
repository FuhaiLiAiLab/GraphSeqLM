# GraphSeqLM: A Unified Graph–Language Framework for Omic Graph Learning

<p align="center">
  <a href="https://dl.acm.org/doi/pdf/10.1145/3701716.3715503">
    <img src="https://img.shields.io/badge/Paper-ACM%20DL-0A7BBB" alt="Paper PDF">
  </a>
  <a href="https://github.com/FuhaiLiAiLab/GraphSeqLM">
    <img src="https://img.shields.io/badge/GitHub-GraphSeqLM-181717?logo=github" alt="GitHub Repo">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  </a>
</p>

![Figure 1](./figures/F1.png)

---

## Overview

**GraphSeqLM** is a unified framework that fuses **graph neural networks (GNNs)** with **biological sequence embeddings** to learn from multi‑omic graphs at scale. It augments topological signals with **LLM‑derived embeddings of DNA, RNA, and proteins**, enabling richer node/edge semantics for **sample‑specific** analyses of signaling pathways and protein–protein interaction networks.

---

## Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [License](#license)

---

## Environment Setup

> Tested with Python 3.10 and PyTorch CUDA 12.1 wheels.

```bash
# Create environment
conda create --name mkg python=3.10
conda activate mkg

# PyTorch (CUDA 12.1)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric core
pip install torch_geometric

# Optional optimizations (match your torch/cu version)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv   -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Transformer backbones for sequence embeddings
pip install transformers
```

---

## Data Preprocessing

```bash
cd data
python processed_data_gen.py
```

This script prepares graph structures and attaches sequence‑derived features used by the GraphSeqLM encoder.

---

## Training

```bash
python main-graphseqlm-gpt.py
```

Key flags (see the script for full options):

- `--dataset`: dataset identifier  
- `--task`: task name (e.g., classification/regression)  
- `--epochs`, `--lr`, `--batch_size`: training hyperparameters

---

## Citation

If this repository is useful in your research, please consider citing the following related work:

```bibtex
@inproceedings{zhang2025graphseqlm,
  title={GraphSeqLM: A Unified Graph Language Framework for Omic Graph Learning},
  author={Zhang, Heming and Huang, Di and Chen, Yixin and Li, Fuhai},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={1510--1513},
  year={2025}
}
```

> You may also wish to cite the GraphSeqLM paper (see the PDF linked above).

## License

MIT License. See `LICENSE` for details.
