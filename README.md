# FIGT2P: Adaptive fuzzy incidence graph learning and tensor three-mode projection for multi-source weak multi-label classification

> A robust framework for multi-source multi-label learning under noisy, missing, and inconsistent annotations.

This repository contains the official implementation of the **FIGT2P** model, designed to handle weakly supervised multi-label learning from multiple heterogeneous sources with label noise and missingness.

---

## 💡 Usage

We provide a small synthetic multi-source multi-label dataset in `datasets/` for quick testing.
> For real-world experiments, please download full datasets from their official sources (see below).

> After downloading, place your dataset in `datasets/`, and use `trans_to_weaklabels.py` to convert the original multi-source multi-label data into a weakly supervised version with missing and noisy labels. The transformed dataset is automatically saved in the `weaklabel_datasets/` folder.

> Run `main.py`, which contains the full implementation of the FIGT2P model, its core algorithm, and evaluation code.

---


## 🔗 Datasets Used in the Paper

> **Please download the datasets from their official sources listed below.**

All datasets used in our experiments are publicly available. Please download them directly from the original providers:

| Dataset Name       | Description                                      | Official Download Link                                                                 |
|--------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **NOIZEUS**        | Speech corpus with noise-augmented utterances    | https://ecs.utdallas.edu/loizou/speech/noizeus/                                        |
| **3Sources**       | Multi-view news articles (BBC, Reuters, Guardian)| http://mlg.ucd.ie/datasets/3sources.html                                               |
| **Rugby**          | Event labeling in rugby match transcripts        | https://github.com/transientlunatic/rugby-data                                         |
| **KDIS MLL**       | Collection of multi-label learning resources     | https://www.uco.es/kdis/mllresources/                                                  |
| **ESC-50**         | Environmental sound classification (50 classes)  | https://github.com/karolpiczak/ESC-50                                                  |
| **Mulan MLC**      | Standard multi-label classification benchmarks   | http://mulan.sourceforge.net/datasets-mlc.html                                         |
| **NUS-WIDE**       | Web image dataset with user tags                 | https://hyper.ai/datasets/16124                                                        |
| **Pascal VOC**     | Image classification & object detection          | https://pjreddie.com/projects/pascal-voc-dataset-mirror/                               |

> 💡 After downloading, organize the data as needed and specify the path via command-line arguments or config files (see usage below).

---

## ⚙️ Installation

Clone the repository:
   ```bash
   git clone https://github.com/EalynChan/FIGT2P.git
