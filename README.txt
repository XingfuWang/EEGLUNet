# EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space

This repository contains the official implementation for the paper:

> **"EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space"**
> *(Accepted to ICASSP 2026)*

---

### **1. Environment Setup**

Required environment:

* Python 3.10
* PyTorch 2.2.2 + CUDA 12.1
* numpy==1.24.4
* pandas==2.2.3
* matplotlib==3.9.0
* scikit-learn==1.5.0
* moabb==1.2.0
* braindecode==0.8.1

We recommend installing PyTorch manually before installing the other dependencies:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
