# ConTP: Transporter Function Prediction via Contrastive Learning

## Introduction

This repository contains the official implementation of the ConTP inference workflow, designed for high-resolution
functional annotation of membrane transporters.
ConTP leverages contrastive learning to disentangle functional determinants from overall sequence similarity, enabling:

- Fine-grained substrate specificity prediction, and
- TC (Transporter Classification) family assignment

The provided pipeline allows users to reproduce the results reported in the manuscript and apply ConTP to annotate novel
transporter sequences.

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
    - [Options](#Options)
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

## Inference

Predict substrate specificity of given transporter proteins:

```commandline
python ./script/predict.py --query_fasta ./temp/example.fasta --task substrate --out_dir ./temp/output/ --save_dist
```

The results are located in ./temp

Predict TC family of given transporter proteins:

```commandline
python ./script/predict.py --query_fasta ./temp/example.fasta --task tc --out_dir ./temp/output/ --save_dist
```

The case study of CLIC6 (TCDB identifier: 1.A.12.1.4)

```commandline
python ./script/predict.py --query_fasta ./temp/CLIC6.fasta --task substrate --out_dir ./temp/output/ --save_dist
```

A Jupyter notebook (```./script/predict.ipynb```) is provided for running inference in an interactive Python environment.
This allows you to customize the prediction workflow, inspect intermediate representations (e.g., ESM embeddings,
distance matrices), and perform advanced analyses tailored to your specific use cases.

## Options

Below is a full description of all available flags (adapted and expanded from ```-h```):

### Required

| Argument             | Description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| `--query_fasta PATH` | Path to the input FASTA file containing one or more query protein sequences. |

### Prediction Task

| Argument                | Description                                                                                                      |
|-------------------------|------------------------------------------------------------------------------------------------------------------|
| `--task {substrate,tc}` | Select prediction task: `substrate` for multi-label substrate classification; `tc` for TC family classification. |

### Thresholding (for substrate prediction)

| Argument            | Description                                                                                                             |
|---------------------|-------------------------------------------------------------------------------------------------------------------------|
| `--threshold FLOAT` | Decision threshold for multi-label substrate classification. Default: `0.034`, which is determined in the training set. |

### Device & Execution Settings

| Argument           | Description                                     |
|--------------------|-------------------------------------------------|
| `--device DEVICE`  | Computing device, e.g., `cuda:0` or `cpu`.      |
| `--batch_size INT` | Batch size used for ESM-2 embedding extraction. |

### Output Options

| Argument           | Description                                                                                                                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--out_dir PATH`   | Directory to save final prediction results.                                                                                                                                                              |
| `--esm_cache PATH` | Optional directory to store or load cached ESM embeddings.                                                                                                                                               |
| `--save_dist`      | If set, saves the full distance matrix for all query sequences as a CSV file named `{basename}_dist.csv` in the output directory. Useful for advanced analysis, custom decision logic, or visualization. |

### Model Checkpoints

| Argument                | Description                                     |
|-------------------------|-------------------------------------------------|
| `--ckpt_substrate PATH` | Pretrained checkpoint for substrate prediction. |
| `--ckpt_tc PATH`        | Pretrained checkpoint for TC classification.    |

### Label Mapping Files

| Argument               | Description                                                                      |
|------------------------|----------------------------------------------------------------------------------|
| `--substrate_map PATH` | JSON file mapping substrate indices to substrate names (70 fine-grained labels). |
| `--tc_map PATH`        | JSON file mapping TC family indices to TC numbers (>1,000 classes).              |

### Reproduction

- Download the preprocessed dataset
  from [Google Drive](https://drive.google.com/file/d/1VAekBVKyqqYjy6qofbl4TEzibWck1Vx6/view?usp=drive_link).
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
- **2025/11/25**: Refine the inference output logic and README.