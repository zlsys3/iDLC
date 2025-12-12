# iDLC: Interpretable Dual-Level Correction for Single-Cell Data Integration

An interpretable deep learning framework for batch correction in single-cell RNA sequencing (scRNA-seq) data that explicitly disentangles biological signals from technical noise through a structured dual-level approach.

## Overview

iDLC (Interpretable Dual-Level Correction) provides a novel solution for integrating scRNA-seq datasets across different experimental batches. The framework employs:

1. **Explicit Feature Disentanglement**: A structured residual autoencoder that explicitly separates biological content from batch-specific technical noise
2. **Interpretable Correction**: Mutual Nearest Neighbor (MNN) pairs as interpretable bridges between biological feature learning and distribution alignment
3. **Dual-Level Integration**: Sequential autoencoder-based feature disentanglement followed by GAN-based distribution correction

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.7.1+
- Scanpy 1.11.5+
- Annoy 1.17.3+

### Install from source
```bash
git clone https://github.com/zlsys3/iDLC.git
cd iDLC
pip install -r requirements.txt
```
## Quick Start

```python
import scanpy as sc
from idlc import iDLCBatchCorrection

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize corrector
corrector = iDLCBatchCorrection(
    n_top_genes=2000,
    latent_dim=128,
    n_epochs_ae=400,
    n_epochs_gan=300,
    ae_batch_size=256,
    gan_batch_size=256,
    lr=0.001,
    patience=15
)

# Perform batch correction
corrected_adata = corrector.correct(
    adata,
    batch_key='batch',  # Column name containing batch information
    k=20,  # Number of neighbors for MNN search
    save_dir="./corrected_results"
)
```
