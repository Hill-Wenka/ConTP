# ConTP: Transporter Function Prediction via Contrastive Learning

## Introduction

This repository contains the official implementation of the ConTP inference workflow, designed for high-resolution functional annotation of membrane transporters.
ConTP leverages contrastive learning to disentangle functional determinants from overall sequence similarity, enabling:

- Fine-grained substrate specificity prediction, and
- TC (Transporter Classification) family assignment

The provided pipeline allows users to reproduce the results reported in the manuscript and apply ConTP to annotate novel transporter sequences.

This work is supported by
[Structural and Functional Bioinformatics Research Group (SFB) group in KAUST](https://sfb.kaust.edu.sa/),

Any questions or suggestions are welcome. You can:

- Report issues in the GitHub repository [ConTP](https://github.com/Hill-Wenka/ConTP/issues).
- Email
    - Co-First
      author [Wenjia He](https://orcid.org/0000-0001-8161-4642) ([wenjia.he@kaust.edu.sa](mailto:wenjia.he@kaust.edu.sa)).
    - Co-First
      author [Chenjie Feng](https://scholar.google.com/citations?user=Lwexn88AAAAJ&hl=en) ([chenjie.feng@kaust.edu.sa](mailto:chenjie.feng@kaust.edu.sa)).
    - Corresponding
      author [Xin Gao](https://orcid.org/0000-0002-7108-3574) ([xin.gao@kaust.edu.sa](mailto:xin.gao@kaust.edu.sa)).

## Table of Contents

<details open><summary><b>Outline</b></summary>

- [Introduction](#Introduction)
- [Environment Installation](#Environment-Installation)
- [Usage](#Inference)
  - [Inference](#Inference)
  - [Reproduction](#Reproduction)
  - [Dataset](#Dataset)
- [News](#News)

</details>

## Environment Installation

### Create conda environment

Install the required packages using conda and pip.

```commandline
conda create -n contp python=3.10 -y
conda activate contp
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ipywidgets jupyterlab tqdm numpy pandas lightning omegaconf biopython fair-esm scikit-learn h5py aaindex tensorboard
```

### Inference

Predict substrate specificity of given transporter proteins:

```commandline
python ./script/predict.py --query_fasta ./temp/example.fasta --task substrate
```

Predict TC family of given transporter proteins:

```commandline
python ./script/predict.py --query_fasta ./temp/example.fasta --task tc
```

### Reproduction

- Download the preprocessed dataset
  from [Google Drive](https://drive.usercontent.google.com/open?id=1VAekBVKyqqYjy6qofbl4TEzibWck1Vx6&authuser=1).
- Move the downloaded archive into the current project directory and extract it into the ```./dataset/``` folder.
- The Jupyter notebook ```./script/debug.ipynb``` provides a minimal example for reproducing the results reported in the
  paper.

### Dataset

- The TP-Substrate dataset is located in  ```./data/TP_Substrate.csv```
- The TP-TC dataset is located in  ```./data/TP_TC.csv```
- ```./data/substrate_mapping.csv``` provides detailed information on 70 fine-grained substrate types.
- ```./data/tc_mapping.csv``` provides detailed information on 1352 fine-grained TC family.

## News

- **2025/11/22**: Upload the TP-Substrate and TP-TC benchmark.
- **2025/11/23**: Update inference codes.
- **2025/11/24**: Update a jupyter notebook to reproduce the result.
