# DOLPHIN

DOLPHIN: Deep Exon-level Graph Neural Network for Single-cell Representation Learning and Alternative Splicing

Full documentation and tutorials are available at [DOLPHIN Docs](https://dolphin-sc.readthedocs.io/en/latest/).

<img title="DOLPHIN Logo" alt="Alt text" src="DOLPHIN_logo.png">

## Overview
<img title="DOLPHIN Overview" alt="Alt text" src="Overview_DOLPHIN.png">
The advent of single-cell sequencing has revolutionized the study of cellular dynamics, providing unprecedented resolution into the molecular states and heterogeneity of individual cells. However, the rich potential of exon-level information and junction reads within single cells remains underutilized. Conventional gene-count methods overlook critical exon and junction data, limiting the quality of cell representation and downstream analyses such as subpopulation identification and alternative splicing detection. To address this, we introduce DOLPHIN, a deep learning method that integrates exon-level and junction read data, representing genes as graph structures. These graphs are processed by a variational autoencoder to improve cell embeddings. Compared to conventional gene-based methods, DOLPHIN shows superior performance in cell clustering, biomarker discovery, and alternative splicing detection, providing deeper insights into cellular processes. By examining cellular dynamics with enhanced resolution, DOLPHIN detects subtle differences often missed at the gene level, offering new insights into disease mechanisms and potential therapeutic targets.

## Key Capabilities of DOLPHIN:

- Exon-Level Quantification: It represents genes as graphs, where nodes are exons and edges are junction reads, capturing detailed transcriptomic information at the exon level.
- Better Cell Embedding: DOLPHIN leverages exon and junction read data to significantly improve the accuracy of cell embeddings, providing better resolution and resulting in more precise, biologically meaningful cell clusters compared to conventional gene-count based approaches.
- Enhanced Alternative Splicing Detection: By aggregating exon and junction reads from neighboring cells, DOLPHIN significantly enhances the detection of alternative splicing events, providing deeper insights into cell-specific splicing patterns.
- Superior Performance in Downstream Analysis: DOLPHIN consistently outperforms conventional gene-count methods in multiple downstream tasks, including the identification of differential exon markers and alternative splicing events. This high-resolution approach allows DOLPHIN to uncover biologically significant exon markers that are often missed by traditional methods.

## Installation

Installing scASprofiler directly from GitHub ensures you have the latest version. 

### 🧠 Platform Notes

**Note:** This tool has been primarily tested on Linux-based systems, specifically Ubuntu 22.04.4 LTS. While it may run on other platforms, we recommend using a **Linux environment** for best compatibility and performance, especially for memory-intensive preprocessing steps such as **STAR** or **Cell Ranger** alignment.

---

📥 Step 1: Clone the Repository
```bash
git clone https://github.com/1027victor/scASprofiler.git
cd scASprofiler
```

🛠 Step 2: Create and Activate the Conda Environment
```bash
conda env create -f environment.yaml
conda activate scasp
```

📦 Step 3: Install torch and the scASprofiler Python Package
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install .
```

🧑‍💻 (Optional) Developer Mode Installation
```bash
pip install -e .
```
---

