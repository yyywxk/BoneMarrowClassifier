# BoneMarrowClassifier

<p align="center">
    <!-- <a href="https://github.com/yyywxk/BoneMarrowClassifier/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/yyywxk/BoneMarrowClassifier/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/yyywxk/BoneMarrowClassifier">
    </a>
    <a href="https://github.com/yyywxk/BoneMarrowClassifier/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/yyywxk/BoneMarrowClassifier">
    </a>
    <a href="https://github.com/yyywxk/BoneMarrowClassifier/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/yyywxk/BoneMarrowClassifier?color=pink">
    </a>
    <a href="https://github.com/yyywxk/BoneMarrowClassifier">
        <img alt="issues" src="https://img.shields.io/github/stars/yyywxk/BoneMarrowClassifier">
    </a>
    <a href="mailto: qiulinwei@buaa.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

## Overview

**BoneMarrowClassifier**  is an open-source project dedicated to the fine-grained classification of bone marrow cells. This repository serves as a comprehensive resource for researchers and practitioners in the field of hematology and medical imaging, providing a suite of baseline models and tools for accurate cell classification.

## Use Cases

- **Research**: Ideal for researchers looking to benchmark new algorithms against established baselines.
- **Clinical Applications**: Useful for developing clinical decision support systems that require precise cell classification.
- **Educational Purposes**: Suitable for educational institutions to teach students about advanced machine learning techniques in medical imaging.

## Requirements

- Please clone this repository and navigate to it in your terminal.
- Then prepare an environment with python=3.10, and then use the command `pip install -r requirements.txt` for the dependencies.

## Datasets

- We support **ALL-IDB** dataset [[Baidu Cloud](https://pan.baidu.com/s/1Lpxudlx_8PMmXCCK2dnRJg?pwd=ynqt)] and **PBC** dataset [[Baidu Cloud](https://pan.baidu.com/s/1GZ0NmogWtfX2UocvvdJ1Nw?pwd=w5m3)] . More details can be found in [SCKansformer: Fine-Grained Classification of Bone Marrow Cells via Kansformer Backbone and Hierarchical Attention Mechanisms](https://ieeexplore.ieee.org/document/10713291).
- All data are set in `data/` directory.

## Getting Started

To get started with ​**BoneMarrowClassifier**​, follow these steps:

1. Clone the repository:
   
   ```bash
   git clone https://github.com/yyywxk/BoneMarrowClassifier.git
   ```
2. Run the baseline models on the **ALL-IDB** dataset:
   
   ```bash
   python train_ALL_IDB2.py --model <model_name>
   ```
3. Or run the baseline models on the **PBC** dataset:
   
   ```bash
   python train_PBC.py --model <model_name>
   ```

## Questions

For any questions or feedback, feel free to contact us at [qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn).

## Acknowledgement

[SCKansformer](https://github.com/JustlfC03/SCKansformer)

