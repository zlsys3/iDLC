"""
Example usage of iDLC batch correction based on AnnData
"""
import os
import torch
# Set PyTorch memory optimization environment variables

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Set to 0 for better performance

# Limit number of threads used by PyTorch
torch.set_num_threads(4)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import scanpy as sc
from idlc import iDLCBatchCorrection


def main():
    # Read merged h5ad file containing all batch data
    print("Loading merged h5ad file...")
    adata = sc.read_h5ad('data/Immune.h5ad')  # Replace with your file path

    print(f"Loaded data shape: {adata.shape}")

    # Create corrector instance
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

    # Execute batch correction - now directly using AnnData object
    corrected_adata = corrector.correct(
        adata,
        k=100,
        batch_key='Batch',  # Specify batch information column
        save_dir="./correct_result/iImmune"
    )

    print("Batch correction completed!")
    print(f"Corrected AnnData object: {corrected_adata}")


if __name__ == "__main__":
    main()