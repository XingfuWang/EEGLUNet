This repository contains the official implementation for the paper:

    "EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space"
    (Accepted to ICASSP 2026)

========================
1. Environment Setup
========================

Required environment:

- Python 3.10
- PyTorch 2.2.2 + CUDA 12.1
- numpy==1.24.4
- pandas==2.2.3
- matplotlib==3.9.0
- scikit-learn==1.5.0
- moabb==1.2.0
- braindecode==0.8.1

We recommend installing PyTorch manually before installing the other dependencies:

    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

Then install all other packages via:

    pip install -r requirements.txt

========================
2. Datasets
========================

This project supports the following EEG datasets:
- BNCI 2014-001 (~743MB)
- BNCI 2014-002 (~179MB)
- Zhou2016 (~176MB)

Important:  
You do NOT need to download any dataset manually.  
On the first run, the script will automatically download the raw data from the internet.

========================
3. How to Run
========================

To train models on all supported datasets using default settings, simply run:

    python main.py

This will:
- Automatically download and preprocess the data
- Train all models
- Save all results under the `output_folder/` directory

Customizing datasets or models:
To evaluate specific datasets or models, open `main.py` and edit the corresponding lists by simply uncommenting the desired lines.

Example:

dataset_list = [
    'BNCI2014_001',
    # 'BNCI2014_002',
    # 'Zhou2016',
]

model_list = [
    'EEGLUNet',
    # 'CTNet',
    # 'DeepNet',
    ...
]

========================
4. Citation & Contact
========================

Thank you for your interest in our work! If you find this codebase or the EEGLUNet framework helpful in your research, please consider citing our ICASSP 2026 paper. 

If you have any questions, encounter issues, or would like to discuss potential collaborations, please feel free to open an issue in this repository or contact the authors directly. 