import os
import shutil
import glob
import warnings
warnings.filterwarnings("ignore")
import random

import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
tf2.random.set_seed(0)
tf2.keras.utils.set_random_seed(0)

from scDREAMER import scDREAMER

# input / output roots (update these for your environment)
DATA_ROOT = '/path/to/iDLC/data'


# dataset label -> (input h5ad file, batch key, cell-type key)
# the label must match the `file_name` used in benchmark.py
datasets = {
    'pdac_mild':    ('pdac_mild.h5ad',    'Batch',     'CellType'),
    'pdac_strong':  ('pdac_strong.h5ad',  'Batch',     'CellType'),
    'immune':       ('immune.h5ad',       'Batch',     'CellType'),
    'hcl_mca_data': ('hcl_mca_data.h5ad', 'Batch',     'CellType'),
    'vc_data':      ('vc_data.h5ad',      'batch',     'drug_type'),
}

# training config (increase EPOCH for full training runs)
EPOCH = 500
Z_DIM = 50     # embedding dim; must be 50 to match the unified 50-dim protocol
X_DIM = 2000   # top highly variable genes

run_config = tf.ConfigProto()
run_config.gpu_options.per_process_gpu_memory_fraction = 0.333
run_config.gpu_options.allow_growth = True

for name, (h5ad_file, batch_key, cell_type_key) in datasets.items():
    dataset_name = os.path.join(DATA_ROOT, h5ad_file)
    with tf.Session(config=run_config) as sess:
        dreamer = scDREAMER(
            sess,
            epoch=EPOCH,
            dataset_name=dataset_name,
            batch=batch_key,
            cell_type=cell_type_key,
            name=name,
            z_dim=Z_DIM,
            X_dim=X_DIM,
        )
        dreamer.train_cluster()