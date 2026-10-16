import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker
import scipy.sparse as sp
import colorcet as cc
# configuration
file_name = 'pdac_strong'          # dataset name used when saving
adata_path = f"./benchmark_results/adata_with_embeddings_{file_name}.h5ad"
output_dir = f"./benchmark_results/{file_name}"
os.makedirs(output_dir, exist_ok=True)

# 1. load AnnData
print(f"Loading adata from {adata_path}...")
adata = sc.read_h5ad(adata_path)
print(f"Loaded adata shape: {adata.shape}")
print(f"Available embeddings: {list(adata.obsm.keys())}")

# 2. color maps
def get_batch_colors(batch_list):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(batch_list))]
    return {batch: colors[i] for i, batch in enumerate(sorted(batch_list))}

def get_celltype_colors(celltype_list):
    n = len(celltype_list)
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(n)]
    else:
        palette = cc.glasbey
        colors = [palette[i] for i in range(n)]
    return {ct: colors[i] for i, ct in enumerate(sorted(celltype_list))}

# ensure Batch is categorical
if 'Batch' in adata.obs:
    adata.obs['Batch'] = adata.obs['Batch'].astype('category')
    batch_cats = adata.obs['Batch'].cat.categories
else:
    batch_cats = []
celltype_cats = adata.obs['CellType'].unique() if 'CellType' in adata.obs else []
batch_color_dict = get_batch_colors(batch_cats) if len(batch_cats) > 0 else {}
celltype_color_dict = get_celltype_colors(celltype_cats) if len(celltype_cats) > 0 else {}
color_dict = {**batch_color_dict, **celltype_color_dict}

# 3. MDE plotting
def mde_and_plot(adata, key, name, color_dict, output_dir, file_prefix):
    if key not in adata.obsm or adata.obsm[key] is None:
        print(f"skip {name}: embedding missing")
        return
    emb = adata.obsm[key]
    if sp.issparse(emb):
        emb = emb.toarray()
    mde_coords = ov.utils.mde(emb)
    adata.obsm[f"mde_{key}"] = mde_coords
    for col in ['Batch', 'CellType']:
        if col not in adata.obs.columns:
            print(f"skip column {col}: not present")
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax = ov.pl.embedding(adata, basis=f"mde_{key}", color=[col], palette=color_dict,
                             frameon='small',show=False, ax=ax, save=f"_{file_prefix}_{name}_{col}")
        fig = ax.get_figure()
        save_path = f"{output_dir}/{file_prefix}_{name}_{col}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {save_path}")

# 4. MDE visualization for all embeddings

methods_to_plot = ['Unintegrated', 'iDLC', 'iMAP', 'scDREAMER', 'scPoli',
                   'Combat', 'Harmony', 'fastMNN', 'Scanorama', 'scVI',
                   'scFoundation', 'scGPT']
for key in methods_to_plot:
    if key in adata.obsm:
        mde_and_plot(adata, key, key, color_dict, output_dir, file_name)
    else:
        print(f"warning: {key} not in adata.obsm, skipping")

# 5. benchmark evaluation
embedding_keys = [key for key in methods_to_plot if key in adata.obsm and adata.obsm[key] is not None]
print(f"\nevaluating methods: {embedding_keys}")

# densify all embeddings
for key in embedding_keys:
    if sp.issparse(adata.obsm[key]):
        adata.obsm[key] = adata.obsm[key].toarray()

if len(embedding_keys) > 0:
    bm = Benchmarker(adata, batch_key="Batch", label_key="CellType",
                     embedding_obsm_keys=embedding_keys, n_jobs=8)
    bm.benchmark()

    # save Benchmarker object
    save_path = f"benchmarker_{file_name}_post.pkl.gz"
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(bm, f)
    print(f"Benchmarker saved to {save_path}")

    # save results table (bm.plot_results_table handles the figure)
    bm.plot_results_table(min_max_scale=False, save_dir=output_dir)
    print(f"results table saved to {output_dir}")

    # print summary
    print("\n=== Benchmark Results ===")
    print(bm._results)
else:
    print("no embeddings available for evaluation; check adata.obsm keys")
