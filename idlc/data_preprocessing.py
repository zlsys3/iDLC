import numpy as np
import scanpy as sc
import warnings
import gc
import scipy
warnings.filterwarnings('ignore')


def custom_data_preprocess(adata, key='batch', n_top_genes=2000, flavor='seurat_v3',
                           min_genes=200, min_cells=3):
    """Data preprocessing with memory optimization"""

    print(f"Starting preprocessing, initial data shape: {adata.shape}")

    # Save original cell names
    original_obs_names = adata.obs_names.copy()

    # Free unnecessary memory
    if hasattr(adata, 'raw'):
        del adata.raw
    gc.collect()

    # Quality control in chunks
    print("Performing quality control...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Data shape after quality control: {adata.shape}")

    # Check if any cells were filtered
    if len(adata.obs_names) != len(original_obs_names):
        print(f"Note: Cell count changed from {len(original_obs_names)} to {len(adata.obs_names)}")

    # Use sparse matrices to save memory
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    print("Performing normalization...")
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Garbage collection again
    gc.collect()

    print("Selecting highly variable genes...")
    # Select highly variable genes - using more memory-efficient method
    try:
        if flavor == 'seurat_v3':
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, batch_key=key)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    except Exception as e:
        print(f"Error selecting highly variable genes: {e}")
        # Fallback to simpler method
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, 1000), flavor='seurat')

    # Extract highly variable genes
    adata = adata[:, adata.var['highly_variable']]

    print('Preprocessing completed.')
    print(f"Final data shape: {adata.shape}")

    # Final garbage collection
    gc.collect()

    return adata