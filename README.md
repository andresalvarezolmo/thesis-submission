# thesis-submission

# Benchmarking Biosignal Foundation Models in Out-of-Distribution Data

This repository contains the code and supporting files for the MPhil project **Benchmarking Biosignal Foundation Models in Out-of-Distribution Data**, submitted to the Department of Computer Science and Technology, University of Cambridge.

The project evaluates the generalisation performance of ECG and EEG foundation models under a variety of real-world distribution shifts. It benchmarks zero-shot and fine-tuned performance across public datasets to assess robustness, highlight limitations, and identify opportunities for improvement in biosignal foundation model research.

Each experiment extends the setup from its original model in a different way. Whereas the ECG and EEGNetv4 experiments contain more newly written code, the WildECG and EEGPT ones, are largely based on the original paper code. All experiments extend the evaluation to use a unified approach, as described in Section 4.4 of the Thesis.

---

## Folder Structure

```
thesis-submission
├── ECG-FM
│   ├── existing-implementation
│   │   └── implementation-link.txt
│   ├── results
│   │   ├── experiment-results.txt
│   │   └── valid.csv
│   └── scripts
│       ├── fine-tuned.py
│       ├── fine-tuning-CODE-15.py
│       ├── raw-model.py
│       └── validation-results.py
├── EEGNetv4
│   ├── existing-implementation
│   │   └── implementation-link.txt
│   ├── modified-library-files
│   │   ├── evaluations.py
│   │   └── results.py
│   ├── results
│   │   └── experiment-results.txt
│   └── scripts
│       ├── Cho2017
│       │   ├── knn-pretrained-Lee.py
│       │   └── logistic-pretrained-Lee.py
│       ├── Weibo2014
│       │   ├── knn-pretrained-Lee.py
│       │   └── logistic-pretrained-Lee.py
│       └── Zhou2016
│           ├── knn-pretrained-Lee.py
│           └── logistic-pretrained-Lee.py
├── EEGPT
│   ├── existing-implementation
│   │   └── implementation-link.txt
│   ├── results
│   │   └── experiment-results.txt
│   └── scripts
│       ├── fine-tuned-metrics-diagram.ipynb
│       ├── finetune_TUAB_EEGPT.sh
│       ├── finetune_TUEV_EEGPT.sh
│       ├── run_class_finetuning_EEGPT_change_tuev.py
│       └── run_class_finetuning_EEGPT_change.py
├── WildECG
│   ├── existing-implementation
│   ├── results
│   │   └── experiment-results.txt
│   └── scripts
│       ├── supervised_cv.py
│       └── supervised_ext.py
└── README.md
```

Each model folder contains:
- `scripts/`: Scripts for fine-tuning, evaluation, and reproducibility
- `results/`: Evaluation results
- `existing-implementation/`: Link to the source model repository

---

## Datasets

This repository does **not** include any datasets. Below are included the links to view and download them.

### ECG
- [PTB-XL](https://physionet.org/content/ptb-xl/)
- [LUDB](https://physionet.org/content/ludb/)
- [CODE-15](https://zenodo.org/records/4916206)
- [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)

### EEG
- [Cho2017](https://neurotechx.github.io/moabb/generated/moabb.datasets.Cho2017.html#moabb.datasets.Cho2017)
- [Weibo2014](https://neurotechx.github.io/moabb/generated/moabb.datasets.Weibo2014.html#moabb.datasets.Weibo2014)
- [Zhou2016](https://neurotechx.github.io/moabb/generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016)
- [TUAB](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
- [TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

---

## System Configuration

Although different versions of Python and PyTorch were used to match the requirements of each foundation model’s original setup, all experiments were run using **CUDA 12.8.93** and **Ubuntu 24.04.2 LTS** to ensure comparability and reproducibility.

| Model     | Python  | PyTorch | scikit-learn |
|-----------|---------|---------|--------------|
| ECG-FM    | 3.9.22  | 2.6.0   | 1.6.1        |
| WildECG   | 3.10.17 | 2.6.0   | 1.6.1        |
| EEGPT     | 3.10.17 | 2.0.0   | 1.4.1        |
| EEGNetv4  | 3.10.17 | 2.6.0   | 1.5.2        |

GPU used: NVIDIA L4 Tensor Core 24 GB

---

## Acknowledgements

This project benefited from open-source resources provided by:
- PhysioNet
- UC Irvine
- Lobachevsky University
- TUH EEG Corpus
- MOABB
- HuggingFace
- Telehealth Network of Minas Gerais
- The original authors of ECG-FM, WildECG, EEGPT and EEGNetv4

All cited works are listed in the bibliography of the submitted thesis.

---
---