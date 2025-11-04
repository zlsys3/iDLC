# iDLC: Interpretable Deep Learning-based batch effect Correction and integration for Single-Cell Transcriptomic Data

iDLC is a deep learning-based method for batch correction in single-cell RNA sequencing data, combining autoencoders with generative adversarial networks (GANs) for effective integration of multiple datasets.

## Features

- **Deep Autoencoder**: Learns meaningful representations while separating batch effects from biological signals
- **GAN-based Correction**: Uses generative adversarial networks for precise batch effect removal
- **Mutual Nearest Neighbors**: Identifies biologically similar cells across batches
- **Comprehensive Visualization**: Detailed loss tracking and training monitoring

## Installation

### From PyPI
```bash
pip install idlc
