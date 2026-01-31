# A Comparative Study of CNN and Vision Transformer for Histopathology Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains the complete experimental pipeline for classifying lung and colon cancer histopathology images. The study performs a head-to-head comparison between **Convolutional Neural Networks (ResNet50)** and **Vision Transformers (ViT)**, while also evaluating the efficacy of **Macenko Stain Normalization** in the preprocessing pipeline.

### Key Research Questions:
1. Does the global attention mechanism of Vision Transformers outperform the local feature extraction of CNNs in medical imaging?
2. Does Macenko Stain Normalization improve model robustness and accuracy on the LC25000 dataset?

---

## 📊 Performance Summary
After 20 epochs of training using transfer learning, the results were as follows:

| Model ID | Architecture | Stain Normalization | Test Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | ResNet50 (Baseline) | None | 97.07% | 0.9699 |
| **Model 2** | ResNet50 | Macenko | 96.53% | 0.9645 |
| **Model 3** | **Vision Transformer** | **None** | **97.36%** | **0.9729** |
| **Model 4** | **Vision Transformer** | **Macenko** | **97.36%** | **0.9729** |

**Key Finding:** The Vision Transformer achieved the highest accuracy. Interestingly, stain normalization did not provide a performance boost, suggesting the LC25000 dataset possesses significant inherent color consistency.

---

## 📁 Repository Structure
```text
.
├── scripts/
│   ├── dataset.py      # Custom Dataset class with Macenko normalization
│   ├── model.py        # Model definitions (ResNet50 & ViT-B/16)
│   ├── train.py        # Training & validation loop logic
│   └── eval.py         # Comprehensive evaluation & plotting
├── results/
│   ├── plots/          # Confusion matrices and training curves
│   ├── models/         # Best saved weights (.pth)
│   └── metrics.json    # Final recorded performance metrics
├── requirements.txt    # Necessary Python packages
└── .gitignore          # Prevents uploading large data/envs

