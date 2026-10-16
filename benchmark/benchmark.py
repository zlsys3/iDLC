import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
import mkl
mkl.set_num_threads(1)
import pickle
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
import scvi
import torch
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker
import scipy.sparse as sp
import imap.stage1
import imap.stage2
from joblib import parallel_backend

# dataset identifier (change for each dataset)
file_name = 'pdac_strong'

# ===== foundation-model (scFoundation / scGPT) configuration =====
# update the paths below; if a package or its weights are missing, that method is skipped automatically.
SCFOUNDATION_REPO_DIR  = '/path/to/iDLC/scFoundation_repo'   # cloned biomap-research/scFoundation repo (model/get_embedding.py + model/OS_scRNA_gene_index.19264.tsv)
SCFOUNDATION_MODEL_DIR = '/path/to/iDLC/scFoundation_model' # weights dir (contains the 01B-resolution checkpoint)
SCGPT_MODEL_DIR        = '/path/to/iDLC/scGPT_human'        # dir with vocab.json + best_model.pt (whole-human or continual-pretrained)
FOUNDATION_DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
# set to 'T' if adata.X is already normalize+log1p rather than raw counts
SCFOUNDATION_PRE_NORMALIZED = 'F'

print(f"omicverse version: {ov.__version__}")
print(f"scanpy version: {sc.__version__}")
print(f"scvi version: {scvi.__version__}")
ov.ov_plot_set()

# 1. load raw data and metadata
adata = sc.read_h5ad(f'/path/to/iDLC/data/{file_name}.h5ad')
metadata = pd.read_csv(f'/path/to/iDLC/data/{file_name}_meta.csv', index_col=0)

common_cells = adata.obs_names.intersection(metadata.index)
adata = adata[common_cells].copy()
metadata = metadata.loc[common_cells]
adata.obs = metadata
adata.obs['Batch'] = adata.obs['Batch'].astype('category')
print(f"raw data shape: {adata.shape}")

# keep a full-gene raw-counts copy for the foundation models: they are pretrained on the
# whole transcriptome, so feeding only the 2000 HVGs used below would unfairly hurt them.
adata_full = adata.copy()

# 2. basic preprocessing
ov.utils.store_layers(adata, layers='counts')
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000, batch_key='Batch')
adata.raw = adata
adata = adata[:, adata.var.highly_variable_features]
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
adata.obsm["Unintegrated"] = adata.obsm["scaled|original|X_pca"]

# 3. run / load each method
# iMAP
adata_imap = adata.copy()
print(adata_imap.shape)
#adata_imap = imap.stage1.data_preprocess(adata_imap, 'Batch', n_batch=len(adata.obs['Batch'].unique()))
EC, ec_data = imap.stage1.iMAP_fast(adata_imap, key="Batch", n_epochs=50, latent_dim=32)
output_results = imap.stage2.integrate_data(adata_imap, ec_data, inc=False, n_epochs=50,
                                            key='Batch', n_batch=len(adata.obs['Batch'].unique()),k1=2, k2=10)
print(output_results.shape)
imap_pca=sc.pp.pca(output_results, n_comps=50, random_state=42)
adata.obsm['iMAP'] = imap_pca

# scDREAMER
scdreamer_csv_path = f'/path/to/iDLC/scDREAMER/{file_name}latent_matrix_50.csv'
scdreamer_df = pd.read_csv(scdreamer_csv_path, header=None)
scdreamer_mat = scdreamer_df.values.astype(np.float32)
print(f"scDREAMER raw shape: {scdreamer_mat.shape}")

if scdreamer_mat.shape[0] != adata.n_obs:
    if scdreamer_mat.shape[1] == adata.n_obs:
        scdreamer_mat = scdreamer_mat.T
        print(f"transposed shape: {scdreamer_mat.shape}")
    else:
        if scdreamer_mat.shape[0] > adata.n_obs:
            scdreamer_mat = scdreamer_mat[:adata.n_obs, :]
        else:
            pad = np.zeros((adata.n_obs - scdreamer_mat.shape[0], scdreamer_mat.shape[1]))
            scdreamer_mat = np.vstack([scdreamer_mat, pad])
        print(f"adjusted shape: {scdreamer_mat.shape}")

adata.obsm['scDREAMER'] = scdreamer_mat

# iDLC
idlc_path = f'/path/to/iDLC/correct_result/{file_name}/corrected_data.h5ad'
corrected_idlc = sc.read_h5ad(idlc_path)
common = adata.obs_names.intersection(corrected_idlc.obs_names)
print(f"iDLC shared cells: {len(common)}")
adata = adata[common].copy()
corrected_idlc = corrected_idlc[adata.obs_names]
sc.pp.pca(corrected_idlc, n_comps=50, random_state=42)
adata.obsm['iDLC'] = corrected_idlc.obsm['X_pca']

# scPoli
scpoli_path = f'/path/to/iDLC/scpoli/{file_name}/adata.h5ad'
scpoli_adata = sc.read_h5ad(scpoli_path)
print(f"scPoli data shape: {scpoli_adata.shape}")
common = adata.obs_names.intersection(scpoli_adata.obs_names)
print(f"scPoli shared cells: {len(common)}")
scpoli_adata = scpoli_adata[adata.obs_names]
# if len(common) != adata.n_obs:
#     adata = adata[common].copy()
#     scpoli_adata = scpoli_adata[common].copy()
# scpoli_adata = scpoli_adata[adata.obs_names]
if 'X_emb' in scpoli_adata.obsm:
    adata.obsm['scPoli'] = scpoli_adata.obsm['X_emb']
else:
    sc.pp.pca(scpoli_adata, n_comps=50, random_state=42)
    adata.obsm['scPoli'] = scpoli_adata.obsm['X_pca']

# Combat / Harmony / Scanorama (omicverse)
adata_combat = ov.single.batch_correction(adata, batch_key='Batch', methods='combat', n_pcs=50)
adata.obsm['Combat'] = adata.obsm.pop('X_combat')

adata_harmony = ov.single.batch_correction(adata, batch_key='Batch', methods='harmony', n_pcs=50, random_state=42)
adata.obsm['Harmony'] = adata.obsm.pop('X_harmony')

adata_scanorama = ov.single.batch_correction(adata, batch_key='Batch', methods='scanorama', n_pcs=50)
adata.obsm['Scanorama'] = adata.obsm.pop('X_scanorama')

# fastMNN
from scanpy.external.pp import mnn_correct
print("running fastMNN...")
adata_mnn = adata.copy()
hvgs = adata_mnn.var_names[adata_mnn.var.highly_variable_features].tolist() if 'highly_variable_features' in adata_mnn.var else None
try:
    corrected_result = mnn_correct(
        adata_mnn, batch_key="Batch", var_subset=hvgs,
        k=20, sigma=1.0, cos_norm_in=True, cos_norm_out=True,
        svd_dim=50, var_adj=True, do_concatenate=True, n_jobs=8
    )
    corrected_adata = corrected_result[0][0]
    sc.pp.pca(corrected_adata, n_comps=50, random_state=42)
    adata.obsm['fastMNN'] = corrected_adata.obsm['X_pca']
except Exception as e:
    print(f"fastMNN failed: {e}")
    adata.obsm['fastMNN'] = None

# scVI
print("running scVI...")
adata_scvi = adata.copy()
scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key="Batch")
vae = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=50, gene_likelihood="nb")
vae.train()
adata.obsm['scVI'] = vae.get_latent_representation()

# ===== foundation-model inference helpers =====
def run_scfoundation_cell_embedding(adata_in, repo_dir, model_dir, device='cuda',
                                    pool_type='all', tgthighres='t4', version='ce',
                                    pre_normalized='F'):
    """Generate scFoundation cell embeddings via the official model/get_embedding.py CLI.

    adata_in.X must be raw counts and adata_in.var_names gene symbols, aligned to the
    repo's OS_scRNA_gene_index.19264.tsv (missing genes padded with 0). Returns (n_obs, 128).
    """
    import subprocess, sys, tempfile
    gene_index = os.path.join(repo_dir, 'model', 'OS_scRNA_gene_index.19264.tsv')
    if not os.path.exists(gene_index):
        raise FileNotFoundError(f"scFoundation gene list not found: {gene_index}")
    gene_list = list(pd.read_csv(gene_index, sep='\t')['gene_name'])
    X = adata_in.X
    if sp.issparse(X):
        X = X.toarray()
    var_names = list(adata_in.var_names)
    g2i = {g: i for i, g in enumerate(gene_list)}
    mat = np.zeros((adata_in.n_obs, len(gene_list)), dtype=np.float32)
    for j, g in enumerate(var_names):
        idx = g2i.get(g)
        if idx is not None:
            mat[:, idx] = X[:, j]
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, 'sf_input.csv')
    pd.DataFrame(mat).to_csv(csv_path, index=False, header=False)
    out_dir = os.path.join(tmp, 'out')
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(repo_dir, 'model', 'get_embedding.py'),
           '--task_name', 'sf', '--input_type', 'singlecell', '--output_type', 'cell',
           '--pool_type', pool_type, '--tgthighres', tgthighres,
           '--data_path', csv_path, '--save_path', out_dir,
           '--pre_normalized', pre_normalized, '--version', version]
    if model_dir:
        cmd += ['--model_path', model_dir]
    env = os.environ.copy()
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run(cmd, check=True, cwd=os.path.join(repo_dir, 'model'), env=env)
    npy_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                 if 'cell_embedding' in f and f.endswith('.npy')]
    if not npy_files:
        raise FileNotFoundError(f"scFoundation produced no embedding file in {out_dir}")
    return np.load(npy_files[0]).astype(np.float32)


def run_scgpt_cell_embedding(adata_in, model_dir, device='cuda',
                             gene_col='index', batch_size=64):
    """Generate scGPT cell embeddings via scgpt.tasks.embed_data. Returns (n_obs, 512)."""
    from scgpt.tasks import embed_data
    adata_sg = adata_in.copy()
    sc.pp.normalize_total(adata_sg, target_sum=1e4)
    sc.pp.log1p(adata_sg)
    try:
        from scgpt.preprocess import binning
        adata_sg, _ = binning(adata_sg, n_bins=51)
    except Exception:
        pass  # newer scGPT versions bin internally
    try:
        adata_sg = embed_data(adata_sg, model_dir=model_dir, gene_col=gene_col,
                              batch_size=batch_size, device=device,
                              use_fast_transformer=False, return_new_adata=True)
        emb = adata_sg.obsm['X_scGPT']
    except TypeError:
        embed_data(adata_sg, model_dir=model_dir, gene_col=gene_col,
                   batch_size=batch_size, device=device, use_fast_transformer=False)
        emb = adata_sg.obsm['X_scGPT']
    return emb.astype(np.float32)


# scFoundation (direct inference)
print("running scFoundation...")
try:
    sf_raw = adata_full[adata.obs_names].copy()
    sf_emb = run_scfoundation_cell_embedding(
        sf_raw, SCFOUNDATION_REPO_DIR, SCFOUNDATION_MODEL_DIR,
        device=FOUNDATION_DEVICE, pre_normalized=SCFOUNDATION_PRE_NORMALIZED)
    if sf_emb.shape[0] != adata.n_obs:
        sf_emb = sf_emb[:adata.n_obs] if sf_emb.shape[0] > adata.n_obs else \
            np.vstack([sf_emb, np.zeros((adata.n_obs - sf_emb.shape[0], sf_emb.shape[1]))])
    sf_adata = sc.AnnData(sf_emb)
    sc.pp.pca(sf_adata, n_comps=50, random_state=42)
    adata.obsm['scFoundation'] = sf_adata.obsm['X_pca']
    print(f"scFoundation embedding shape: {adata.obsm['scFoundation'].shape}")
except Exception as e:
    print(f"scFoundation inference failed, skipping: {e}")
    adata.obsm['scFoundation'] = None

# scGPT (direct inference)
print("running scGPT...")
try:
    sg_raw = adata_full[adata.obs_names].copy()
    sg_emb = run_scgpt_cell_embedding(sg_raw, SCGPT_MODEL_DIR, device=FOUNDATION_DEVICE)
    if sg_emb.shape[0] != adata.n_obs:
        sg_emb = sg_emb[:adata.n_obs] if sg_emb.shape[0] > adata.n_obs else \
            np.vstack([sg_emb, np.zeros((adata.n_obs - sg_emb.shape[0], sg_emb.shape[1]))])
    sg_adata = sc.AnnData(sg_emb)
    sc.pp.pca(sg_adata, n_comps=50, random_state=42)
    adata.obsm['scGPT'] = sg_adata.obsm['X_pca']
    print(f"scGPT embedding shape: {adata.obsm['scGPT'].shape}")
except Exception as e:
    print(f"scGPT inference failed, skipping: {e}")
    adata.obsm['scGPT'] = None

adata.write(f"benchmark_results/adata_with_embeddings_{file_name}.h5ad",compression='gzip')
print(f"saved all embeddings to benchmark_results/adata_with_embeddings_{file_name}.h5ad")
