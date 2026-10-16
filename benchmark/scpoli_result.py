import os
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
from scarches.models.scpoli import scPoli

# input / output roots (update these for your environment)
DATA_ROOT = '/path/to/iDLC/data'
SCGPOLI_OUT_ROOT = '/path/to/iDLC/scpoli'

# dataset label -> (input h5ad file, batch/condition key)
# the label must match the `file_name` used in benchmark.py
datasets = {
    'pdac_mild':    ('pdac_mild.h5ad',    'Batch'),
    'pdac_strong':  ('pdac_strong.h5ad',  'Batch'),
    'immune':       ('immune.h5ad',       'Batch'),
    'hcl_mca_data': ('hcl_mca_data.h5ad', 'Batch'),
    'vc_data':      ('vc_data.h5ad',      'batch'),
}

early_stopping_kwargs = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

for name, (h5ad_file, batch_key) in datasets.items():
    adata = sc.read(os.path.join(DATA_ROOT, h5ad_file))
    scpoli_model = scPoli(
        adata=adata,
        condition_keys=batch_key,
        embedding_dims=50,   # matches the unified 50-dim evaluation protocol
        recon_loss='nb',
    )
    scpoli_model.train(
        n_epochs=100,
        pretraining_epochs=20,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=5,
    )
    out_dir = os.path.join(SCGPOLI_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    scpoli_model.save(out_dir, overwrite=True, save_anndata=True)
    print(f"scPoli for {name} saved to {out_dir}")
