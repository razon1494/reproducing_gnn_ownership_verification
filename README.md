# Revisiting Black-box Ownership Verification for Graph Neural Networks

***Paper title: Revisiting Black-box Ownership Verification for Graph Neural Networks***

## Overview

This repository presents my independent reproduction and analysis of the paper  
**"Revisiting Black-box Ownership Verification for Graph Neural Networks" (IEEE S&P 2024)**.

My goal is not only to reproduce the reported results, but also to:
- understand the robustness and limitations of black-box ownership verification for GNNs,
- analyze how verification performance changes under different architectures and attack settings,
- and explore directions toward more structure-aware and robust fingerprinting methods.

This project is part of my preparation for PhD-level research in applied machine learning security.

## What Was Reproduced

I reproduced the following experiments from the paper:

- Base black-box ownership verification (Section 5.2)
- Multiple verification settings (Settings 1–4)
- Robustness evaluations under:
  - model fine-tuning
  - weight pruning
  - double extraction
- Adaptive attack scenarios (Section 6.4)

Experiments were run on:
- Datasets: Cora, Citeseer (extendable)
- Architectures: GCN, GAT, GraphSAGE
- Learning paradigm: Transductive

## My Contributions

Beyond reproducing the experimental pipeline provided by the original authors, my contributions include:

- Successfully reproduced core experimental results reported in the IEEE S&P 2024 paper.
- Implementing and reproducing the robustness evaluation corresponding to **Table 8 (Impact of Double Extraction)**.
- Aggregating experimental results across multiple datasets, verification settings, and learning paradigms into a unified analysis script.
- Verifying consistency between reproduced results and those reported in the paper, while identifying scenarios where verification performance degrades under repeated extraction.
- Structuring the codebase and analysis artifacts to support further extensions toward more robust and structure-aware fingerprinting methods.

All analysis scripts and result summaries were developed independently as part of this reproduction study.

## Key Observations and Insights

From the reproduced experiments, I observed the following patterns:

- **Double extraction significantly weakens ownership verification performance**, particularly under transductive settings, indicating that repeated model extraction poses a serious threat to current black-box verification methods.
- **Verification robustness varies across datasets and architectures**, suggesting that existing methods may implicitly rely on dataset-specific structural properties.
- In several settings, verification accuracy remains high despite non-zero false positive rates, highlighting a potential trade-off between detection sensitivity and reliability.

These observations suggest that future ownership verification methods should explicitly account for adaptive adversaries and structural graph properties, motivating exploration of structure-aware fingerprinting strategies.

The reproduced results for the impact of double extraction are summarized in the folder `CSV Results` and the file name is: `table8_impact_of_double_extraction.csv`.

## Implementation and Reproducibility Details (Optional)

## Environment Setup

Opearting system: Ubuntu 22.04.4 LTS

CPU: Intel i9-12900K

Graphics card: RTX 4090

RAM: 64GB

CUDA version: 11.8

You need to install some third-party libraries with the following commands:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy
pip install scikit-learn
pip install tqdm
pip install pyyaml
pip install argparse
```

## File Illustration

## Code Structure (High-Level)

- `main/`: experiment orchestration, extraction, robustness evaluation, and verification logic
- `model/`: GNN architectures, extraction models, and verification classifiers
- `config/`: experiment configurations for datasets, architectures, and verification settings
- `utils/`: data loading, graph processing, and shared utilities

Detailed configuration and execution instructions are provided below for reproducibility.


## Run Experiments

### Base Ownership Verification Experiments (See Section 5.2)

Everytime after setting the corresponding parameters in configurations, you can directly run the "main.py" file under the "main" folder.

```
python main.py
```

For example, with the default settings, you will run four verification settings (see Section 4.2), with no masking magnitude, in the Cora dataset, under transductive learning.

### Extended Studies (See Section 6.3)

1. extended study I: impact of independent models which are trained on randomly picked data

Inside "train_models_by_setting" function in "verification_cfg.py": set "process" parameter to "test".

```
python main.py
```

2. extended study II: impact of local model numbers

Inside "train_setting1.yaml": set the "num_model_per_arch" parameter to the local model numbers you want to test. In our paper, valid values: [10, 20, 30, 40, 50].

```
python main.py
```

### Robustness Experiments (See Section 6.4)

Fine-tune, prune and double extraction are all implemented in the "robustness.py" file. 

To run all three robustness techniques, you need to pass the saving path of real test models to the functions.

Besides, for the prune robustness experiment, you need to set the magnitude of prune.

After setting corresponding parameters in configurations, you can directly run the "robustness.py" file under the "main" folder to get real test models after robustness techniques.

```
python robustness.py
```

And then you should change the "test_save_root" parameter to the path where you saved the real test models after robustness techniques and run the ownership verification experiment again.

```
python main.py
```

### Adaptive Attacks (See Section 6.4)

Inside "train_models_by_setting" function in "verification_cfg.py": set "classifier" parameter to "classifier_model".

```
python main.py
```

## Results Viewing

All models and results will be saved in the path you set in the "global_cfg.yaml" file.

The name of each json file is the target model architecture.

Inside the file:

- TN: true negative number.
- TP: true positive number.
- FN: false negative number.
- FP: false positive number.
- FPR: false positive rate.
- FNR: false negative rate.
- Accuracy: accuracy of ownership verification.
- original_model_acc: target model accuracy of downstream task.
- mask_model_acc: masked target model accuracy of downstream task.
- train_inde_acc_list: local independent models accuracy of downstream task.
- train_surr_acc_list: local extraction models accuracy of downstream task.
- train_surr_fidelity_list: local extraction models fidelity of downstream task.
- test_inde_acc_list: real test independent models accuracy of downstream task.
- test_surr_acc_list: real test extraction models accuracy of downstream task.
- test_surr_fidelity_list: real test extraction models fidelity of downstream task.
- total_time: total running time.
- mask_run_time: masking running time.
## Citation
If you find several components of this work useful or want to use this code in your research, please cite the following paper:
@inproceedings{zhou2024revisiting,\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={Revisiting Black-box Ownership Verification for Graph Neural Networks},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Zhou, Ruikai and Yang, Kang and Wang, Xiuling and Wang, Wendy Hui and Xu, Jun},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;booktitle={2024 IEEE Symposium on Security and Privacy (SP)},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pages={210--210},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2024},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;organization={IEEE Computer Society}\
}
