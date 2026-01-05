# main.py
"""
Example usage of iDLC batch correction based on AnnData
"""
import os
import torch


import scanpy as sc
from idlc import iDLCBatchCorrection


def main():
    # Read h5ad file containing all batch data
    print("Loading merged h5ad file...")
    adata = sc.read_h5ad('your_data.h5ad')
    print(f"Loaded data shape: {adata.shape}")


    # Create corrector instance
    corrector = iDLCBatchCorrection(
        n_top_genes=2000,
        latent_dim=128,
        n_epochs_ae=300,
        n_epochs_gan=200,
        ae_batch_size=256,
        gan_batch_size=256,
        lr=0.001,
        patience=15
       
    )


    corrected_adata = corrector.correct(
        adata,
        k=30,
        batch_key='Batch',
        save_dir="./correct_result",
        use_ot=True,           # Enable OT regularization
        ot_method='sinkhorn',       # Choose OT method: 'mmd', 'wasserstein', 'energy', 'sinkhorn'
        ot_weight=0.01          # OT loss weight
    )

    print("Batch correction completed!")
    print(f"Corrected AnnData object: {corrected_adata}")




if __name__ == "__main__":
    main()