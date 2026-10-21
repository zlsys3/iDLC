import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import gzip
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import omicverse as ov
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker
import scipy.sparse as sp
import colorcet as cc

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
file_name = 'pdac_strong'
adata_path = f"./benchmark_results/adata_with_embeddings_{file_name}.h5ad"
output_dir = f"./benchmark_results/{file_name}"
os.makedirs(output_dir, exist_ok=True)

BATCH_KEY = 'Batch'
LABEL_KEY = 'CellType'

# scGraph (Wang et al. 2026, Nat Biotechnol)
RUN_SCGRAPH = True
SCGRAPH_PARAMS = dict(
    trim_rate=0.05,
    thres_batch=100,
    thres_celltype=10,
)
SCGRAPH_SCORE_COL = 'corr-weighted'   # the scGraph score reported in the paper

# MNN pair accuracy. The paper reports it for MNN-based methods only;
# set MNN_METHODS = None to score every embedding.
RUN_MNN = True
MNN_METHODS = ['iDLC', 'iMAP', 'fastMNN', 'Scanorama']
MNN_K = 20                          # must match k_mnn in Supplementary Table S2
MNN_MAX_CELLS_PER_BATCH = None      # set an int to subsample very large batches

# DEG-overlap (Jaccard) accuracy
RUN_DEG = True
FEATURES_PATH = f"./benchmark_results/adata_with_features_{file_name}.h5ad"
# Layer names are matched against the method name automatically ('iDLC',
# 'iDLC_corrected', 'X_iDLC', ...). Map explicitly if your layers differ.
FEATURE_LAYER_OVERRIDE = {}
FEATURES_IS_LOG = True              # False: run normalize_total + log1p before DE

DEG_PVAL_KEY = 'pvals'              # or 'pvals_adj'
DEG_ALPHA = 0.05
DEG_MIN_LOGFOLD = 1.0               # |log2FC| > 1, i.e. at least two-fold
DEG_MIN_REF_MARKERS = 10            # cell types with fewer reference markers are skipped
DEG_MAX_CELLS_PER_TYPE = None       # set an int to subsample very large cell types

MAKE_BARPLOTS = True

# ---------------------------------------------------------------------
# 1. Load AnnData
# ---------------------------------------------------------------------
print(f"Loading adata from {adata_path}...")
adata = sc.read_h5ad(adata_path)
print(f"Loaded adata shape: {adata.shape}")
print(f"Available embeddings: {list(adata.obsm.keys())}")

# ---------------------------------------------------------------------
# 2. Color maps
# ---------------------------------------------------------------------
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

if BATCH_KEY in adata.obs:
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype('category')
    batch_cats = adata.obs[BATCH_KEY].cat.categories
else:
    batch_cats = []
celltype_cats = adata.obs[LABEL_KEY].unique() if LABEL_KEY in adata.obs else []
batch_color_dict = get_batch_colors(batch_cats) if len(batch_cats) > 0 else {}
celltype_color_dict = get_celltype_colors(celltype_cats) if len(celltype_cats) > 0 else {}
color_dict = {**batch_color_dict, **celltype_color_dict}

# ---------------------------------------------------------------------
# 3. MDE plotting
# ---------------------------------------------------------------------
def mde_and_plot(adata, key, name, color_dict, output_dir, file_prefix):
    if key not in adata.obsm or adata.obsm[key] is None:
        print(f"skip {name}: embedding missing")
        return
    emb = adata.obsm[key]
    if sp.issparse(emb):
        emb = emb.toarray()
    mde_coords = ov.utils.mde(emb)
    adata.obsm[f"mde_{key}"] = mde_coords
    for col in [BATCH_KEY, LABEL_KEY]:
        if col not in adata.obs.columns:
            print(f"skip column {col}: not present")
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax = ov.pl.embedding(adata, basis=f"mde_{key}", color=[col], palette=color_dict,
                             frameon='small', show=False, ax=ax, save=f"_{file_prefix}_{name}_{col}")
        fig = ax.get_figure()
        save_path = f"{output_dir}/{file_prefix}_{name}_{col}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {save_path}")

# ---------------------------------------------------------------------
# 4. MDE visualization for all embeddings
# ---------------------------------------------------------------------
methods_to_plot = ['Unintegrated', 'iDLC', 'iMAP', 'scDREAMER', 'scPoli',
                   'Combat', 'Harmony', 'fastMNN', 'Scanorama', 'scVI',
                   'scFoundation', 'scGPT']
for key in methods_to_plot:
    if key in adata.obsm:
        mde_and_plot(adata, key, key, color_dict, output_dir, file_name)
    else:
        print(f"warning: {key} not in adata.obsm, skipping")

# ---------------------------------------------------------------------
# 5. scIB benchmark
# ---------------------------------------------------------------------
embedding_keys = [key for key in methods_to_plot if key in adata.obsm and adata.obsm[key] is not None]
print(f"\nevaluating methods: {embedding_keys}")

for key in embedding_keys:
    if sp.issparse(adata.obsm[key]):
        adata.obsm[key] = adata.obsm[key].toarray()

if len(embedding_keys) > 0:
    bm = Benchmarker(adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                     embedding_obsm_keys=embedding_keys, n_jobs=8)
    bm.benchmark()

    save_path = f"benchmarker_{file_name}_post.pkl.gz"
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(bm, f)
    print(f"Benchmarker saved to {save_path}")

    bm.plot_results_table(min_max_scale=False, save_dir=output_dir)
    print(f"results table saved to {output_dir}")

    print("\n=== Benchmark Results ===")
    print(bm._results)
else:
    print("no embeddings available for evaluation; check adata.obsm keys")


# ---------------------------------------------------------------------
# 6. scGraph (Corr-Weighted)
# ---------------------------------------------------------------------
def run_scgraph(adata, keys, batch_key, label_key, params, cache_dir):
    """Run scgraph; newer versions accept AnnData directly, older ones need a path."""
    try:
        from scgraph import scGraph
    except ImportError:
        print("[scGraph] not installed, skipping. Install with: pip install scgraph-eval")
        return None

    # only_umap must be False: the benchmark embeddings are not 2D.
    kw = dict(batch_key=batch_key, label_key=label_key, only_umap=False, **params)

    try:
        sg = scGraph(adata=adata, obsm_keys=list(keys), **kw)
        res = sg.main()
        print("[scGraph] used in-memory AnnData API")
        return res
    except TypeError as e:
        print(f"[scGraph] this version rejects adata=/obsm_keys= ({e}); falling back to a temporary h5ad")

    tmp = os.path.join(cache_dir, f"_scgraph_tmp_{os.getpid()}.h5ad")
    sub = ad.AnnData(X=adata.X, obs=adata.obs[[batch_key, label_key]].copy())
    for k in keys:
        sub.obsm[k] = np.asarray(adata.obsm[k])
    sub.write_h5ad(tmp)
    try:
        sg = scGraph(adata_path=tmp, **kw)
        return sg.main()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def pick_score_col(df, wanted):
    """Locate the Corr-Weighted column; its spelling varies between releases."""
    norm = {c: str(c).lower().replace('_', '-').replace(' ', '-') for c in df.columns}
    for c, n in norm.items():
        if n == wanted:
            return c
    for c, n in norm.items():
        if 'corr' in n and 'weight' in n:
            return c
    raise KeyError(f'no Corr-Weighted column found; columns are {list(df.columns)}')


scgraph_df = None
if RUN_SCGRAPH and len(embedding_keys) > 0:
    print("\n=== scGraph ===")
    raw = run_scgraph(adata, embedding_keys, BATCH_KEY, LABEL_KEY,
                      SCGRAPH_PARAMS, output_dir)
    if raw is not None:
        raw.to_csv(f"{output_dir}/scgraph_full_results.csv")
        col = pick_score_col(raw, SCGRAPH_SCORE_COL)
        s = pd.Series(raw[col].values, index=raw.index, name='scGraph')
        scgraph_df = s.to_frame()
        scgraph_df.to_csv(f"{output_dir}/scgraph_scores.csv")
        print(f"[scGraph] score column used: {col}")
        print(scgraph_df.round(4))

        # Some versions prefix the returned keys; map them back to method names.
        if not set(scgraph_df.index) & set(embedding_keys):
            mapping = {}
            for m in embedding_keys:
                for idx in scgraph_df.index:
                    if str(idx).endswith(m) or str(idx) == m:
                        mapping[idx] = m
            if mapping:
                scgraph_df = scgraph_df.rename(index=mapping)
                print(f"[scGraph] index remapped: {mapping}")


# ---------------------------------------------------------------------
# 7. MNN pair accuracy
# ---------------------------------------------------------------------
try:                      # ships with umap-learn; much faster than brute force on large inputs
    from pynndescent import NNDescent
    _HAVE_PYNND = True
except ImportError:
    _HAVE_PYNND = False


def _knn_backend():
    return 'pynndescent' if _HAVE_PYNND else 'sklearn'


def _knn(X_ref, X_q, k, backend):
    """Indices of the k nearest neighbours in X_ref for every row of X_q."""
    k = int(min(k, len(X_ref)))
    if backend == 'pynndescent':
        idx = np.ascontiguousarray(X_ref, dtype=np.float32)
        q = np.ascontiguousarray(X_q, dtype=np.float32)
        index = NNDescent(idx, n_neighbors=k, random_state=0, n_jobs=-1)
        ind, _ = index.query(q, k=k)
        ind = np.asarray(ind)
        ind = np.where(ind < 0, 0, ind)      # approximate search occasionally returns -1
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1).fit(X_ref)
        _, ind = nn.kneighbors(X_q)          # sklearn returns (distances, indices)
        ind = np.asarray(ind)
    return ind.astype(np.int64, copy=False)  # some releases return floats


def mnn_pair_accuracy(adata, key, batch_key, label_key, k=20,
                      max_cells_per_batch=None, seed=0):
    """Fraction of mutually nearest cross-batch pairs that share a cell type.

    Pairs are pooled over all batch pairs and weighted by pair count.
    """
    emb = adata.obsm[key]
    if sp.issparse(emb):
        emb = emb.toarray()
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    emb = emb / norm                      # L2 normalisation, as in the paper

    batches = adata.obs[batch_key].astype('category')
    labels = adata.obs[label_key].astype(str).values
    cats = list(batches.cat.categories)
    idx_by = {b: np.where(batches.values == b)[0] for b in cats}

    rng = np.random.default_rng(seed)
    backend = _knn_backend()
    total_pairs = 0
    total_ok = 0
    per_pair = {}

    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            A, B = idx_by[cats[i]], idx_by[cats[j]]
            if len(A) < k + 1 or len(B) < k + 1:
                continue
            if max_cells_per_batch is not None:
                if len(A) > max_cells_per_batch:
                    A = rng.choice(A, max_cells_per_batch, replace=False)
                if len(B) > max_cells_per_batch:
                    B = rng.choice(B, max_cells_per_batch, replace=False)

            knn_ab = _knn(emb[B], emb[A], k, backend)   # each cell of A -> positions in B
            knn_ba = _knn(emb[A], emb[B], k, backend)   # each cell of B -> positions in A
            set_ab = [set(row.tolist()) for row in knn_ab]

            n_pair = 0
            n_ok = 0
            for b_pos in range(len(B)):
                for a_pos in knn_ba[b_pos]:
                    if b_pos in set_ab[a_pos]:
                        n_pair += 1
                        if labels[A[a_pos]] == labels[B[b_pos]]:
                            n_ok += 1
            per_pair[f"{cats[i]}|{cats[j]}"] = (n_ok / n_pair if n_pair else np.nan, n_pair)
            total_pairs += n_pair
            total_ok += n_ok

    return dict(accuracy=(total_ok / total_pairs if total_pairs else np.nan),
                n_pairs=total_pairs,
                n_correct=total_ok,
                backend=backend,
                per_pair=per_pair)


mnn_df = None
if RUN_MNN and len(embedding_keys) > 0:
    print("\n=== MNN pair accuracy ===")
    targets = embedding_keys if MNN_METHODS is None else \
        [m for m in MNN_METHODS if m in embedding_keys]
    print(f"methods scored: {targets}  (k={MNN_K})")
    if MNN_MAX_CELLS_PER_BATCH is not None:
        print(f"NOTE: at most {MNN_MAX_CELLS_PER_BATCH} cells per batch (subsampled estimate)")

    rows = []
    for m in targets:
        r = mnn_pair_accuracy(adata, m, BATCH_KEY, LABEL_KEY, k=MNN_K,
                              max_cells_per_batch=MNN_MAX_CELLS_PER_BATCH)
        rows.append(dict(Method=m, MNN_accuracy=r['accuracy'],
                         n_pairs=r['n_pairs'], n_correct=r['n_correct']))
        print(f"  {m:14s} acc={r['accuracy']:.4f}  pairs={r['n_pairs']}  "
              f"({r['n_correct']} correct, knn={r['backend']})")
        pd.Series({kk: vv for kk, (vv, _) in r['per_pair'].items()}).to_csv(
            f"{output_dir}/mnn_per_batchpair_{m}.csv")
    mnn_df = pd.DataFrame(rows).set_index('Method')
    mnn_df.to_csv(f"{output_dir}/mnn_accuracy.csv")
    print(mnn_df.round(4))


# ---------------------------------------------------------------------
# 8. DEG-overlap (Jaccard) accuracy
# ---------------------------------------------------------------------
def markers_from_expr(X, var_names, obs_labels, label_key, pval_key,
                      alpha, min_logfold, max_cells_per_type=None, seed=0):
    """One-vs-rest Wilcoxon markers per cell type; returns {cell type: set(genes)}."""
    a = ad.AnnData(X=X, obs=pd.DataFrame({label_key: obs_labels.values},
                                         index=obs_labels.index),
                   var=pd.DataFrame(index=var_names))
    a.obs[label_key] = a.obs[label_key].astype('category')

    groups = [g for g in a.obs[label_key].cat.categories
              if (a.obs[label_key] == g).sum() >= 3]
    if max_cells_per_type is not None and groups:
        rng = np.random.default_rng(seed)
        keep = []
        for g in groups:
            idx = np.where(a.obs[label_key].values == g)[0]
            if len(idx) > max_cells_per_type:
                idx = rng.choice(idx, max_cells_per_type, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        a = a[keep, :].copy()
        a.obs[label_key] = a.obs[label_key].astype('category')
        groups = [g for g in a.obs[label_key].cat.categories
                  if (a.obs[label_key] == g).sum() >= 3]

    if not groups:
        return {}

    sc.tl.rank_genes_groups(a, groupby=label_key, groups=groups,
                            reference='rest', method='wilcoxon')
    out = {}
    for g in groups:
        df = sc.get.rank_genes_groups_df(a, group=g)
        fc = 'logfoldchanges' if 'logfoldchanges' in df.columns else 'log2FoldChange'
        pv = pval_key if pval_key in df.columns else 'pvals'
        sel = df[(df[pv] < alpha) & (df[fc].abs() > min_logfold)]
        out[str(g)] = set(sel['names'].astype(str))
    return out


def find_feature_key(feat, method, n_vars):
    """Locate the gene-level matrix for one method; returns (container, key) or (None, None)."""
    if method in FEATURE_LAYER_OVERRIDE:
        k = FEATURE_LAYER_OVERRIDE[method]
        if k in feat.layers:
            return 'layers', k
        if k in feat.obsm:
            return 'obsm', k
        print(f"  {method:14s} warning: FEATURE_LAYER_OVERRIDE points to missing key '{k}'")
        return None, None

    cands = [method, f'{method}_corrected', f'X_{method}', f'X_{method}_corrected',
             f'corrected_{method}', f'{method}_correct']
    for k in cands:
        if k in feat.layers:
            return 'layers', k
    for k in cands:
        if k in feat.obsm and getattr(feat.obsm[k], 'shape', (0, 0))[1] == n_vars:
            return 'obsm', k

    low = method.lower()
    for k in feat.layers.keys():
        if low in str(k).lower():
            return 'layers', k
    return None, None


def align_features(feat, ref):
    """Reorder feat to the reference cell order; returns (feat, message or None)."""
    if feat.n_obs != ref.n_obs:
        return None, f'cell count differs (features={feat.n_obs}, reference={ref.n_obs})'
    if feat.obs_names.equals(ref.obs_names):
        return feat, None
    if set(map(str, feat.obs_names)) == set(map(str, ref.obs_names)):
        return feat[ref.obs_names].copy(), 'reordered to the reference cell order'
    return None, 'obs_names differ from the reference; cannot align cells'


deg_df = None
if RUN_DEG and LABEL_KEY in adata.obs:
    print("\n=== DEG-overlap (Jaccard) ===")
    feat = None
    if not os.path.exists(FEATURES_PATH):
        print(f"  {FEATURES_PATH} not found; skipping DEG-overlap")
    else:
        print(f"  reading {FEATURES_PATH} ...")
        feat = sc.read_h5ad(FEATURES_PATH)
        print(f"  shape={feat.shape}  layers={list(feat.layers.keys())}")
        print(f"  obsm={list(feat.obsm.keys())}")
        feat, msg = align_features(feat, adata)
        if feat is None:
            print(f"  skipping DEG-overlap: {msg}")
        elif msg:
            print(f"  note: {msg}")

if RUN_DEG and LABEL_KEY in adata.obs and feat is not None:
    ref_X = adata.X
    ref_labels = adata.obs[LABEL_KEY]

    print("computing reference markers on uncorrected data ...")
    ref_markers = markers_from_expr(ref_X, adata.var_names, ref_labels, LABEL_KEY,
                                    DEG_PVAL_KEY, DEG_ALPHA, DEG_MIN_LOGFOLD,
                                    DEG_MAX_CELLS_PER_TYPE)
    ref_kept = {g: s for g, s in ref_markers.items() if len(s) >= DEG_MIN_REF_MARKERS}
    print(f"  cell types with >= {DEG_MIN_REF_MARKERS} reference markers: {len(ref_kept)}")

    ref_genes = set(map(str, adata.var_names))
    rows = []
    for m in embedding_keys:
        container, key = find_feature_key(feat, m, feat.n_vars)
        if container is None:
            print(f"  {m:14s} skip: no gene-level matrix in the features file "
                  f"(set FEATURE_LAYER_OVERRIDE if the layer name differs)")
            continue
        try:
            Xm = feat.layers[key] if container == 'layers' else feat.obsm[key]
            genes = feat.var_names
            if not FEATURES_IS_LOG:
                tmp = ad.AnnData(X=Xm.copy(), var=pd.DataFrame(index=genes))
                sc.pp.normalize_total(tmp, target_sum=1e4)
                sc.pp.log1p(tmp)
                Xm = tmp.X
            print(f"  {m:14s} using {container}['{key}']  shape={Xm.shape}")
        except Exception as e:
            print(f"  {m:14s} skip: failed to read the matrix ({e})")
            continue

        # Restrict both marker sets to shared genes, ordered as in the reference.
        genes_str = list(map(str, genes))
        g2pos = {g: i for i, g in enumerate(genes_str)}
        common = [g for g in adata.var_names.astype(str) if g in g2pos and g in ref_genes]
        if not common:
            print(f"  {m:14s} skip: no shared gene names with the reference")
            continue
        pos_m = [g2pos[g] for g in common]
        Xm = Xm[:, pos_m]
        common_set = set(common)
        if len(common) != feat.n_vars or len(common) != adata.n_vars:
            print(f"                 genes used: {len(common)} "
                  f"(reference {adata.n_vars}, features {feat.n_vars}) - scored on the intersection")

        mk = markers_from_expr(Xm, np.array(common), ref_labels, LABEL_KEY,
                               DEG_PVAL_KEY, DEG_ALPHA, DEG_MIN_LOGFOLD,
                               DEG_MAX_CELLS_PER_TYPE)

        jac = []
        for g, ref_set in ref_kept.items():
            if g not in mk:
                continue
            a_set = ref_set & common_set
            b_set = mk[g] & common_set
            if not (a_set | b_set):
                continue
            jac.append(len(a_set & b_set) / len(a_set | b_set))
        if jac:
            rows.append(dict(Method=m, DEG_overlap=float(np.mean(jac)),
                             n_celltypes_scored=len(jac)))
            print(f"  {m:14s} DEG-overlap={np.mean(jac):.4f}  "
                  f"({len(jac)} cell types)")
        else:
            print(f"  {m:14s} skip: no scoreable cell type")

    if rows:
        deg_df = pd.DataFrame(rows).set_index('Method')
        deg_df.to_csv(f"{output_dir}/deg_overlap.csv")
        print(deg_df.round(4))
    else:
        print("  no method produced a scoreable gene-level matrix; "
              "check the layer names in FEATURES_PATH, or set FEATURE_LAYER_OVERRIDE")


# ---------------------------------------------------------------------
# 9. Summary and bar plots
# ---------------------------------------------------------------------
print("\n=== Summary ===")
parts = []
if scgraph_df is not None:
    parts.append(scgraph_df)
if mnn_df is not None:
    parts.append(mnn_df[['MNN_accuracy']])
if deg_df is not None:
    parts.append(deg_df[['DEG_overlap']])

if parts:
    summary = pd.concat(parts, axis=1)
    summary.to_csv(f"{output_dir}/summary_all_metrics.csv")
    print(summary.round(4))

    if MAKE_BARPLOTS:
        for col, fname, xlabel in [('scGraph', 'bar_scgraph.png', 'scGraph (Corr-Weighted)'),
                                   ('MNN_accuracy', 'bar_mnn_accuracy.png', 'MNN pair accuracy'),
                                   ('DEG_overlap', 'bar_deg_overlap.png', 'DEG-overlap (Jaccard)')]:
            if col not in summary.columns:
                continue
            s = summary[col].dropna().sort_values()
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(5.2, max(2.4, 0.32 * len(s) + 1.0)))
            ax.barh(s.index, s.values, color='#4C72B0')
            for y, v in enumerate(s.values):
                ax.text(v, y, f' {v:.3f}', va='center', fontsize=7)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.tick_params(labelsize=8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{fname}", dpi=300)
            plt.close(fig)
            print(f"saved {output_dir}/{fname}")
else:
    print("no supplementary metric was computed")
