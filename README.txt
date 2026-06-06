# EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space
This repository provides the official PyTorch implementation of **EEGLUNet**, a novel network for efficient and discriminative EEG feature representation learning proposed in our ICASSP 2026 paper.

> **EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space**
> *Accepted by IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)*

---

## 1. Environment Setup
### 1.1 Dependencies Requirements
All tested software versions are listed below for reproducibility:
- Python == 3.10
- PyTorch == 2.2.2 + CUDA 12.1
- numpy == 1.24.4
- pandas == 2.2.3
- matplotlib == 3.9.0
- scikit-learn == 1.5.0
- moabb == 1.2.0
- braindecode == 0.8.1

### 1.2 Installation Instructions
It is recommended to install PyTorch first to match the designated CUDA runtime version, followed by other auxiliary dependencies.
```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

## 2. Supported Datasets
Our framework supports three public motor imagery EEG datasets automatically managed via the MOABB toolkit:
| Dataset Name | Approximate Storage Size |
| :----------- | :----------------------- |
| BNCI 2014-001 | 743 MB |
| BNCI 2014-002 | 179 MB |
| Zhou2016 | 176 MB |

> **Important Note**: No manual dataset downloading is required. Raw EEG data will be automatically downloaded, verified and preprocessed at the first execution of the program.

## 3. How to Run the Code
### 3.1 Default Full Execution
Run the following command to train and evaluate all pre-defined models on all supported datasets under default hyperparameter configurations:
```bash
python main.py
```
The execution pipeline automatically includes three stages:
1. Automatic dataset downloading and preprocessing
2. Model training and validation
3. Experimental result storage under the `output_folder/` directory

### 3.2 Customized Dataset & Model Configuration
To execute experiments on partial datasets or specified models, modify the `dataset_list` and `model_list` variables in `main.py` by uncommenting target entries and commenting unused items.
```python
# Example for customized dataset selection
dataset_list = [
    'BNCI2014_001',
    # 'BNCI2014_002',
    # 'Zhou2016',
]

# Example for customized model selection
model_list = [
    'EEGLUNet',
    # 'CTNet',
    # 'DeepNet',
    # Other baseline models
]
```

## 4. Citation
If this repository and our work benefit your research, please star ⭐ this repository and cite our published paper as follows:
```bibtex
@INPROCEEDINGS{11462965,
  author={Wang, Xingfu and Qi, Wenxia and Yang, Wenjie and Hu, Boshang and Liu, Chuman and Wang, Wei},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space},
  year={2026},
  volume={},
  number={},
  pages={7216-7220},
  keywords={Filtering;Filters;Band-pass filters;Filter banks;Active filters;Circuits and systems;Protocols;Radio access networks;Regional area networks;Communication systems;EEG;Signal processing;Deep learning;Neural network},
  doi={10.1109/ICASSP55912.2026.11462965}
}
```

## 5. Contact
For technical issues, implementation bugs, academic discussion or potential research cooperation, feel free to submit an Issue in this repository or contact the corresponding authors via email.

## 6. Acknowledgements
We thank the authors of [MATT](https://github.com/cecnl/matt), [Tensor-CSPNet](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet), and [Braindecode](https://github.com/braindecode/braindecode), as parts of our implementation are adapted from their open-source codes.
