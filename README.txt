# EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space



This repository contains the official implementation for the paper:  

> "EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space"  

> *(Accepted to ICASSP 2026)*



---



## 1. Environment Setup



### 1.1 Requirements

The following dependencies are required to run the code:

- Python 3.10

- PyTorch 2.2.2 + CUDA 12.1

- numpy == 1.24.4

- pandas == 2.2.3

- matplotlib == 3.9.0

- scikit-learn == 1.5.0

- moabb == 1.2.0

- braindecode == 0.8.1



### 1.2 Installation Steps

We recommend installing **PyTorch first manually** (to ensure CUDA compatibility) before other dependencies:



```bash

# Install PyTorch with CUDA 12.1 support

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Then install the remaining dependencies via requirements.txt:

Bash



pip install -r requirements.txt

2. Datasets

This project supports the following EEG datasets:

BNCI 2014-001 (~743MB)

BNCI 2014-002 (~179MB)

Zhou2016 (~176MB)

Important Note

✅ No manual download required! On the first run, the script will automatically download the raw data from the internet.

3. How to Run

3.1 Default Execution

To train models on all supported datasets using default settings, simply execute:

Bash



python main.py

This command will:

Automatically download and preprocess the data.

Train all models.

Save all results under the output_folder/ directory.

3.2 Customizing Datasets or Models

To evaluate specific datasets or models, open main.py and edit the corresponding lists by simply uncommenting the desired lines. 😄

Example:

Python



dataset_list = [

    'BNCI2014_001',

    # 'BNCI2014_002',

    # 'Zhou2016',

]



model_list = [

    'EEGLUNet',

    # 'CTNet',

    # 'DeepNet',

    # ...

]

4. Citation & Contact

If you find our research or this codebase helpful in your work, please consider giving this repository a ⭐ Star and citing our ICASSP 2026 paper:

代码段



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

If you have any questions, encounter issues, or would like to discuss potential collaborations, please feel free to open an issue in this repository or contact the authors directly.

5. Acknowledgements

We sincerely thank the open-source community, specifically the authors of matt(https://github.com/cecnl/matt), Tensor-CSPNet(https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet), and the Braindecode framework for their excellent research and implementations.
