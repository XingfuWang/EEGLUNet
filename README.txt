# EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space

This repository contains the official implementation for the paper:
**"EEGLUNet: Fast and Discriminative EEG Representation Learning in LU-Decomposed Space"**
(Accepted to ICASSP 2026)

---

## **1. Environment Setup**

**Required environment:**

- Python 3.10
- PyTorch 2.2.2 + CUDA 12.1
- numpy==1.24.4
- pandas==2.2.3
- matplotlib==3.9.0
- scikit-learn==1.5.0
- moabb==1.2.0
- braindecode==0.8.1

**Installation Steps:**
We recommend installing PyTorch manually before installing the other dependencies:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

Then install all other packages via:

```bash
pip install -r requirements.txt

```

---

## **2. Datasets**

This project supports the following EEG datasets:

* **BNCI 2014-001** (~743MB)
* **BNCI 2014-002** (~179MB)
* **Zhou2016** (~176MB)

**Important:** You do **NOT** need to download any dataset manually.

On the first run, the script will automatically download the raw data from the internet.

---

## **3. How to Run**

**Default Execution:**
To train models on all supported datasets using default settings, simply run:

```bash
python main.py

```

This will:

* Automatically download and preprocess the data
* Train all models
* Save all results under the `output_folder/` directory

**Customizing datasets or models:**
To evaluate specific datasets or models, open `main.py` and edit the corresponding lists by simply uncommenting the desired lines. 😄

*Example:*

```python
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

```

---

## **4. Citation & Contact**

If you find our research or this codebase helpful in your work, please consider giving this repository a ⭐ **star** and citing our ICASSP 2026 paper:

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

If you have any questions, encounter issues, or would like to discuss potential collaborations, please feel free to open an issue in this repository or contact the authors directly.

---

## **5. Acknowledgements**

We would like to express our sincere gratitude to the open-source community for their invaluable contributions to this field. Specifically, parts of our code are based on the excellent research and implementations from:

* The authors of **[matt](https://github.com/cecnl/matt)**
* The authors of **[Tensor-CSPNet](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet)**
* The developers of the **[Braindecode](https://www.google.com/search?q=https://braindecode.com/)** framework

Happy coding! 🌸

```
