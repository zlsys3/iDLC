import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import warnings
import scipy.sparse

from .data_preprocessing import custom_data_preprocess
from .models import iDLCAutoencoder
from .losses import ImprovedLossRecorder, SimpleEarlyStopping

from .training import train_gan, train_gan_with_ot, acquire_mnn_pairs

warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class iDLCBatchCorrection:
    """Stable DeepMNN batch correction pipeline - AnnData version"""

    def __init__(self, n_top_genes=2000, latent_dim=256, n_epochs_ae=100,
                 n_epochs_gan=100, ae_batch_size=256,gan_batch_size=64, lr=0.001, patience=15):
        self.n_top_genes = n_top_genes
        self.latent_dim = latent_dim
        self.n_epochs_ae = n_epochs_ae
        self.n_epochs_gan = n_epochs_gan
        self.ae_batch_size = ae_batch_size
        self.gan_batch_size = gan_batch_size
        self.lr = lr
        self.patience = patience
        self.autoencoder = None
        self.hvg_indices = None
        self.scaler_mean = None
        self.scaler_std = None
        self.batch_info = None
        self.is_trained = False
        self.loss_recorder = ImprovedLossRecorder()

    def preprocess_data(self, adata: sc.AnnData, batch_key: str = 'batch') -> sc.AnnData:
        """基于 AnnData 的数据预处理"""
        print("开始基于 AnnData 的数据预处理...")

        # 检查批次信息
        if batch_key not in adata.obs.columns:
            raise ValueError(f"批次键 '{batch_key}' 未在 adata.obs 中找到")

        print(f"输入数据形状: {adata.shape}")
        print(f"批次: {adata.obs[batch_key].unique().tolist()}")

        # 保存原始细胞名称
        original_obs_names = adata.obs_names.copy()

        # 使用自定义预处理 - 传入 batch_key
        adata = custom_data_preprocess(adata, key=batch_key, n_top_genes=self.n_top_genes)

        # 检查是否有细胞被过滤
        if len(adata.obs_names) != len(original_obs_names):
            print(f"注意: 细胞数量从 {len(original_obs_names)} 变为 {len(adata.obs_names)} 经过过滤")

        # 保存预处理信息
        self.hvg_indices = adata.var_names.tolist()

        # 正确处理稀疏矩阵的统计量计算
        if hasattr(adata.X, 'toarray'):
            print("检测到稀疏矩阵，转换为稠密矩阵计算统计量...")
            dense_matrix = adata.X.toarray()
            self.scaler_mean = dense_matrix.mean(axis=0)
            self.scaler_std = dense_matrix.std(axis=0)
            del dense_matrix
        else:
            self.scaler_mean = adata.X.mean(axis=0)
            self.scaler_std = adata.X.std(axis=0)

        # 避免除零错误
        self.scaler_std = np.where(self.scaler_std == 0, 1, self.scaler_std)

        # 获取批次信息 - 使用传入的 batch_key
        unique_batches = adata.obs[batch_key].unique()
        self.batch_info = {batch: idx for idx, batch in enumerate(unique_batches)}
        self.batch_num = len(self.batch_info)

        print(f"预处理完成，最终数据形状: {adata.shape}")
        print(f"批次数量: {self.batch_num}")

        return adata

    def train_autoencoder(self, adata, batch_key: str = 'batch'):
        """训练自动编码器（修复版：增加随机标签的分类损失，强制解码器使用批次信息）"""
        # 确保数据是稠密矩阵
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray().astype(np.float32)
        else:
            X = adata.X.astype(np.float32)

        n_cells, n_genes = X.shape

        # 获取批次标签并映射到数字
        batch_labels = adata.obs[batch_key].map(self.batch_info).values

        # 转换为 one-hot 编码
        batch_onehot = np.zeros((n_cells, self.batch_num))
        batch_onehot[np.arange(n_cells), batch_labels] = 1
        batch_onehot = torch.FloatTensor(batch_onehot)

        # 初始化自动编码器
        self.autoencoder = iDLCAutoencoder(n_genes, self.batch_num, self.latent_dim)

        if torch.cuda.is_available():
            self.autoencoder = self.autoencoder.cuda()
            batch_onehot = batch_onehot.cuda()

        # 优化器与调度器
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()                # 重建损失
        classification_criterion = nn.CrossEntropyLoss()  # 分类损失
        early_stopping = SimpleEarlyStopping(patience=self.patience)

        # 数据加载
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), batch_onehot)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.ae_batch_size, shuffle=True, drop_last=True
        )

        # 训练循环
        self.autoencoder.train()
        for epoch in range(self.n_epochs_ae):
            total_loss = 0
            total_recon_loss = 0
            total_content_loss = 0
            total_class_loss = 0
            total_rand_class_loss = 0   # 新增随机分类损失

            for batch_idx, (x_batch, batch_onehot_batch) in enumerate(dataloader):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    batch_onehot_batch = batch_onehot_batch.cuda()

                optimizer.zero_grad()

                # ---- 1. 真实批次标签路径 ----
                encoded, decoded, bio_part = self.autoencoder(x_batch)
                batch_part = encoded[:, :self.batch_num]          # 真实批次组分
                recon_loss = criterion(decoded, x_batch)
                class_loss = classification_criterion(batch_part, torch.argmax(batch_onehot_batch, dim=1))

                # ---- 2. 随机批次标签路径（核心修复） ----
                # 生成随机 one-hot 批次标签（与真实批次不同的随机标签）
                random_indices = torch.randint(0, self.batch_num, (x_batch.size(0),))
                if torch.cuda.is_available():
                    random_indices = random_indices.cuda()
                random_batch_noise = torch.zeros_like(batch_onehot_batch)
                random_batch_noise.scatter_(1, random_indices.unsqueeze(1), 1)

                # 使用随机批次标签 + 原始生物组分 重建
                random_reconstructed = self.autoencoder.decoder(
                    torch.cat([random_batch_noise, bio_part], dim=1)
                )


                # 再次编码随机重建结果，提取其批次组分和生物组分
                new_encoded, _, new_bio_part = self.autoencoder(random_reconstructed)
                new_batch_part = new_encoded[:, :self.batch_num]

                # 内容一致性损失：生物组分应保持不变
                content_loss = criterion(new_bio_part, bio_part)

                # ★★★ 核心修复：随机重建的批次组分必须与输入的随机标签一致 ★★★
                rand_class_loss = classification_criterion(new_batch_part, random_indices)

                # ---- 3. 总损失（权重可调整） ----
                loss = (recon_loss
                        + 0.5 * content_loss
                        + 0.5 * class_loss
                        + 0.5 * rand_class_loss)      # 关键项：强制使用随机标签

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()

                # 记录损失
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_content_loss += content_loss.item()
                total_class_loss += class_loss.item()
                total_rand_class_loss += rand_class_loss.item()

            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_content = total_content_loss / len(dataloader)
            avg_class = total_class_loss / len(dataloader)
            avg_rand_class = total_rand_class_loss / len(dataloader)

            # 记录损失（扩展记录器以包含 rand_class_loss）
            self.loss_recorder.record_ae_loss(avg_loss, avg_recon, avg_content, avg_class)
            # 可选：额外存储 rand_class_loss，您可以在 ImprovedLossRecorder 中添加相应记录
            # 这里为简洁，不扩展记录器

            scheduler.step(avg_loss)
            if early_stopping(avg_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"AE Epoch {epoch + 1}/{self.n_epochs_ae}, Loss: {avg_loss:.4f}, "
                    f"Recon: {avg_recon:.4f}, Content: {avg_content:.4f}, "
                    f"Class: {avg_class:.4f}, RandClass: {avg_rand_class:.4f}, LR: {current_lr:.6f}")

        self.is_trained = True

    def extract_features(self, X):
        """提取特征"""
        self.autoencoder.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                X = torch.FloatTensor(X).cuda()
            else:
                X = torch.FloatTensor(X)

            _, _, bio_part = self.autoencoder(X)
            return bio_part.cpu().numpy()

    def visualize_components(self, adata_processed, batch_key='batch', celltype_key='celltype', save_dir=None):
        """
        可视化解耦的生物学组分 (c) 和 批次噪声组分 (n)
        - 按批次着色：生物学组分 UMAP、批次噪声组分 PCA（默认配色）
        - 按细胞类型着色：生物学组分 UMAP（根据类型数量自适应色系）
        - 反事实重建
        图例放在底部并换行，标记点放大。
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import umap
        import numpy as np
        import os
        import torch

        if not self.is_trained:
            raise ValueError("Model must be trained before visualization.")

        # 获取表达矩阵（稠密）
        if hasattr(adata_processed.X, 'toarray'):
            X_all = adata_processed.X.toarray().astype(np.float32)
        else:
            X_all = adata_processed.X.astype(np.float32)

        # 提取所有细胞的生物学组分 (c) 和 批次组分 (n)
        self.autoencoder.eval()
        all_bio_parts = []
        all_batch_parts = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(X_all), batch_size):
                end = min(i + batch_size, len(X_all))
                X_batch = torch.FloatTensor(X_all[i:end])
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                encoded, _, bio_part = self.autoencoder(X_batch)
                batch_part = encoded[:, :self.batch_num]
                all_bio_parts.append(bio_part.cpu().numpy())
                all_batch_parts.append(batch_part.cpu().numpy())
        bio_mat = np.vstack(all_bio_parts)
        batch_mat = np.vstack(all_batch_parts)

        # 获取批次标签和细胞类型标签
        batch_labels = adata_processed.obs[batch_key].values
        unique_batches = np.unique(batch_labels)

        has_celltype = celltype_key in adata_processed.obs.columns
        if has_celltype:
            celltype_labels = adata_processed.obs[celltype_key].values
            unique_celltypes = np.unique(celltype_labels)
            n_celltypes = len(unique_celltypes)

        # --- 生物学组分的 UMAP（全数据降维一次，避免重复计算）---
        pca = PCA(n_components=50, random_state=42)
        bio_pca = pca.fit_transform(bio_mat)
        reducer = umap.UMAP(n_components=2, random_state=42)
        bio_umap = reducer.fit_transform(bio_pca[:, :30])

        # ===== 辅助函数：底部多列图例 =====
        def add_bottom_legend(ax, n_items, title='', ncol=5, y_offset=-0.12, markerscale=6):
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                return
            ncol = min(ncol, n_items)
            ax.legend(handles, labels, title=title, loc='upper center',
                    bbox_to_anchor=(0.5, y_offset), ncol=ncol, frameon=False,
                    markerscale=markerscale, handletextpad=0.5)
            plt.subplots_adjust(bottom=0.15)

        # ========== 1. 按批次着色的生物学组分 UMAP ==========
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            ax.scatter(bio_umap[mask, 0], bio_umap[mask, 1],
                    label=batch, s=2, alpha=0.6, rasterized=True)
        ax.set_title('Biological Component (c) colored by batch')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        add_bottom_legend(ax, len(unique_batches), title='Batch', ncol=5, y_offset=-0.12, markerscale=6)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'bio_component_umap_by_batch.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ========== 2. 按细胞类型着色的生物学组分 UMAP ==========
        if has_celltype:
            # 自适应色系
            if n_celltypes <= 10:
                cmap = plt.get_cmap('tab10')
                colors = [cmap(i) for i in range(n_celltypes)]
            elif n_celltypes <= 20:
                cmap = plt.get_cmap('tab20')
                colors = [cmap(i) for i in range(n_celltypes)]
            else:
                try:
                    cmap = plt.get_cmap('turbo')
                except AttributeError:
                    cmap = plt.get_cmap('gist_ncar')
                colors = [cmap(i / n_celltypes) for i in range(n_celltypes)]
            celltype_color_map = {ct: colors[i] for i, ct in enumerate(sorted(unique_celltypes))}

            fig, ax = plt.subplots(figsize=(7, 6))
            for ct in sorted(unique_celltypes):
                mask = celltype_labels == ct
                ax.scatter(bio_umap[mask, 0], bio_umap[mask, 1],
                        label=ct, s=2, alpha=0.6, rasterized=True,
                        color=celltype_color_map[ct])
            ax.set_title('Biological Component (c) colored by cell type')
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ncol_celltype = min(6, n_celltypes)
            add_bottom_legend(ax, n_celltypes, title='Cell type', ncol=ncol_celltype, y_offset=-0.12, markerscale=6)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'bio_component_umap_by_celltype.png'), dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"Warning: '{celltype_key}' not found in obs. Skipping celltype-colored UMAP.")

        # ========== 3. 批次噪声组分的 PCA（按批次着色）==========
        if self.batch_num > 2:
            pca_n = PCA(n_components=2, random_state=42)
            batch_pca = pca_n.fit_transform(batch_mat)
        else:
            batch_pca = batch_mat

        fig, ax = plt.subplots(figsize=(7, 6))
        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            ax.scatter(batch_pca[mask, 0], batch_pca[mask, 1],
                    label=batch, s=2, alpha=0.6, rasterized=True)
        ax.set_title('Batch Noise Component (n) colored by batch')
        ax.set_xlabel('PC1' if self.batch_num > 2 else 'dim1')
        ax.set_ylabel('PC2' if self.batch_num > 2 else 'dim2')
        add_bottom_legend(ax, len(unique_batches), title='Batch', ncol=5, y_offset=-0.12, markerscale=6)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'batch_component_pca.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ========== 4. 反事实重建（只显示真实批次重建 vs 随机批次重建，不显示原始数据） ==========
        sample_idx = 0
        sample_bio = bio_mat[sample_idx:sample_idx+1]
        true_batch = batch_labels[sample_idx]
        true_batch_idx = self.batch_info[true_batch]

        real_noise = np.zeros((1, self.batch_num))
        real_noise[0, true_batch_idx] = 1
        real_noise_t = torch.FloatTensor(real_noise)
        other_batch_idx = (true_batch_idx + 1) % self.batch_num
        random_noise = np.zeros((1, self.batch_num))
        random_noise[0, other_batch_idx] = 1
        random_noise_t = torch.FloatTensor(random_noise)

        sample_bio_t = torch.FloatTensor(sample_bio)
        if torch.cuda.is_available():
            real_noise_t = real_noise_t.cuda()
            random_noise_t = random_noise_t.cuda()
            sample_bio_t = sample_bio_t.cuda()
            self.autoencoder.cuda()

        with torch.no_grad():
            combined_real = torch.cat([real_noise_t, sample_bio_t], dim=1)
            recon_real = self.autoencoder.decoder(combined_real).cpu().numpy()
            combined_random = torch.cat([random_noise_t, sample_bio_t], dim=1)
            recon_random = self.autoencoder.decoder(combined_random).cpu().numpy()

        # 去标准化
        if self.scaler_mean is not None:
            recon_real_denorm = recon_real * self.scaler_std + self.scaler_mean
            recon_random_denorm = recon_random * self.scaler_std + self.scaler_mean
        else:
            recon_real_denorm = recon_real
            recon_random_denorm = recon_random

        n_genes_show = min(20, len(self.hvg_indices))
        gene_names = self.hvg_indices[:n_genes_show]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(n_genes_show)
        width = 0.35
        # 仅绘制两组bar，并排居中
        ax.bar(x - width/2, recon_real_denorm[0, :n_genes_show], width,
            label='Reconstructed (true batch)', alpha=0.7, color='#ff7f0e')
        ax.bar(x + width/2, recon_random_denorm[0, :n_genes_show], width,
            label='Reconstructed (random batch)', alpha=0.7, color='#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels(gene_names, rotation=45, ha='right')
        ax.set_ylabel('Expression (denormalized)')
        ax.set_title('Counterfactual reconstruction: fixed biological component, changed batch noise')
        ax.legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'counterfactual_reconstruction.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print("Component visualization completed.")

    def save_model(self, filepath):
        """保存模型和预处理参数"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_state = {
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'hvg_indices': self.hvg_indices,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'batch_info': self.batch_info,
            'batch_num': self.batch_num,
            'n_top_genes': self.n_top_genes,
            'latent_dim': self.latent_dim
        }

        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """加载模型和预处理参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")

        model_state = torch.load(filepath, map_location='cpu')

        self.hvg_indices = model_state['hvg_indices']
        self.scaler_mean = model_state['scaler_mean']
        self.scaler_std = model_state['scaler_std']
        self.batch_info = model_state['batch_info']
        self.batch_num = model_state['batch_num']
        self.n_top_genes = model_state['n_top_genes']
        self.latent_dim = model_state['latent_dim']

        n_genes = len(self.hvg_indices)
        self.autoencoder = iDLCAutoencoder(n_genes, self.batch_num, self.latent_dim)
        self.autoencoder.load_state_dict(model_state['autoencoder_state_dict'])

        if torch.cuda.is_available():
            self.autoencoder = self.autoencoder.cuda()

        self.is_trained = True
        print(f"Model loaded from {filepath}")

    # 在batch_correction.py的correct方法中修改以下部分：

    def correct(self, adata: sc.AnnData, batch_key: str = 'batch', save_dir: str = None,
            model_path: str = None, plot_loss: bool = True, k: int = 50,
            use_ot: bool = False, ot_method='mmd', ot_weight=0.1,
            vis_components: bool = False) -> sc.AnnData:
        """主校正函数 - 完全基于 AnnData"""
        
        print("开始批次校正流程...")
        
        # 如果提供了模型路径，加载模型
        if model_path and os.path.exists(model_path):
            print("加载预训练模型...")
            self.load_model(model_path)
            # 对输入数据应用预处理
            adata_processed = self.apply_preprocessing_to_adata(adata, batch_key)
        else:
            # 数据预处理
            print("预处理数据...")
            adata_processed = self.preprocess_data(adata, batch_key)
            
            # 训练自动编码器
            print("训练自动编码器...")
            self.train_autoencoder(adata_processed, batch_key)
        
        print(f"预处理后数据形状: {adata_processed.shape}")
        
        # 确保数据是稠密矩阵
        if hasattr(adata_processed.X, 'toarray'):
            X_all = adata_processed.X.toarray().astype(np.float32)
        else:
            X_all = adata_processed.X.astype(np.float32)
        
        # 提取所有细胞的潜在表示
        print("提取生物学潜在表示...")
        all_latent = self.extract_features(X_all)
        print(f"潜在表示形状: {all_latent.shape}")
        
        # 按批次分组原始数据和潜在表示
        batch_groups = {}
        latent_groups = {}
        unique_batches = adata_processed.obs[batch_key].unique()
        
        for batch in unique_batches:
            batch_mask = adata_processed.obs[batch_key] == batch
            # 原始数据
            batch_groups[batch] = X_all[batch_mask]
            # 潜在表示
            latent_groups[batch] = all_latent[batch_mask]
        
        # 选择参考批次
        ref_batch = max(batch_groups.keys(), key=lambda x: len(batch_groups[x]))
        print(f"参考批次: {ref_batch}")
        
        # 校正每个批次
        corrected_data = {}
        reference_set = batch_groups[ref_batch].copy()
        reference_latent = latent_groups[ref_batch].copy()
        
        # 处理参考批次
        corrected_data[ref_batch] = batch_groups[ref_batch]
        
        # 校正其他批次
        batches_to_correct = [batch for batch in unique_batches if batch != ref_batch]
        
        for batch in batches_to_correct:
            print(f"校正批次: {batch} 相对于参考批次 {ref_batch}")
            
            # 使用潜在表示搜索 MNN 对
            print("使用潜在表示搜索MNN对...")
            pairs = acquire_mnn_pairs(reference_latent, latent_groups[batch], k=k)
            
            if len(pairs) < 10:
                print(f"MNN 对不足 ({len(pairs)}), 使用直接校正")
                corrected_data[batch] = batch_groups[batch]
            else:
                # 准备GAN训练数据
                datasetA = batch_groups[batch][[y for x, y in pairs]]
                datasetB = reference_set[[x for x, y in pairs]]
                
                # 确保数据是 float32 类型
                datasetA = datasetA.astype(np.float32)
                datasetB = datasetB.astype(np.float32)
                batch_groups_batch_float32 = batch_groups[batch].astype(np.float32)
                
                # 为OT损失准备潜在表示
                latent_A = latent_groups[batch][[y for x, y in pairs]]
                latent_B = reference_latent[[x for x, y in pairs]]
                
                # 选择是否使用OT正则化
                if use_ot:
                    print(f"使用OT正则化GAN进行批次校正 (方法: {ot_method}, 权重: {ot_weight})")
                    corrected_batch, _ = train_gan_with_ot(
                        datasetA,
                        datasetB,
                        batch_groups_batch_float32,
                        n_epochs=self.n_epochs_gan,
                        batch_size=self.gan_batch_size,
                        loss_recorder=self.loss_recorder,
                        ot_weight=ot_weight,
                        ot_method=ot_method,
                        autoencoder=self.autoencoder,  # 传入自编码器
                        latent_A=latent_A,             # 传入潜在表示A
                        latent_B=latent_B              # 传入潜在表示B
                    )
                else:
                    print("使用原始GAN进行批次校正")
                    corrected_batch, _ = train_gan(
                        datasetA,
                        datasetB,
                        batch_groups_batch_float32,
                        n_epochs=self.n_epochs_gan,
                        batch_size=self.gan_batch_size,
                        loss_recorder=self.loss_recorder
                    )
                
                corrected_data[batch] = corrected_batch
            
            # 更新参考集和潜在表示
            reference_set = np.vstack([reference_set, corrected_data[batch]])
            # 重新提取校正后数据的潜在表示
            corrected_latent = self.extract_features(corrected_data[batch])
            reference_latent = np.vstack([reference_latent, corrected_latent])
        
        # 合并所有校正后的数据
        all_corrected = []
        all_corrected_labels = []
        all_corrected_cell_names = []
    
        for batch in unique_batches:
            all_corrected.append(corrected_data[batch])
            batch_mask = adata_processed.obs[batch_key] == batch
            all_corrected_labels.extend([batch] * len(corrected_data[batch]))
            all_corrected_cell_names.extend(adata_processed.obs_names[batch_mask].tolist())
    
        corrected_matrix = np.vstack(all_corrected)
    
        print(f"最终校正矩阵形状: {corrected_matrix.shape}")
        print(f"最终细胞名称数量: {len(all_corrected_cell_names)}")
    
        # 最终检查
        if len(all_corrected_cell_names) != corrected_matrix.shape[0]:
            raise ValueError(
                f"细胞名称数量 ({len(all_corrected_cell_names)}) 与数据行数 ({corrected_matrix.shape[0]}) 不匹配!")
    
        # 创建校正后的 AnnData 对象
        corrected_adata = sc.AnnData(
            scipy.sparse.csr_matrix(corrected_matrix)  # 使用稀疏矩阵节省空间
        )
        corrected_adata.obs_names = all_corrected_cell_names
        corrected_adata.var_names = self.hvg_indices
        corrected_adata.obs[batch_key] = all_corrected_labels
    
        # 保存校正数据
        if save_dir:
            self.save_corrected_adata(corrected_adata, save_dir, batch_key)
    
        # 保存模型
        if model_path and not os.path.exists(model_path):
            self.save_model(model_path)
    
        # 绘制损失曲线
        if plot_loss:
            loss_plot_path = os.path.join(save_dir, "training_losses.png") if save_dir else None
            self.loss_recorder.plot_losses(loss_plot_path)
        if vis_components:
            self.visualize_components(adata_processed, batch_key, 'drug_type', save_dir)
        return corrected_adata

    def apply_preprocessing_to_adata(self, adata: sc.AnnData, batch_key: str = 'batch') -> sc.AnnData:
        """对新数据应用预处理"""
        if self.hvg_indices is None:
            raise ValueError("预处理参数不可用。请先训练或加载模型。")

        print("对新数据应用预处理...")

        # 只保留高变基因
        common_genes = adata.var_names.intersection(self.hvg_indices)
        adata_filtered = adata[:, common_genes]

        # 添加缺失的高变基因（用0填充）
        missing_genes = set(self.hvg_indices) - set(common_genes)
        if missing_genes:
            print(f"添加 {len(missing_genes)} 个缺失基因（用0填充）")
            # 创建新的 AnnData 对象，包含所有高变基因
            new_X = np.zeros((adata.n_obs, len(self.hvg_indices)), dtype=np.float32)

            # 填充现有基因的数据
            gene_indices = {gene: idx for idx, gene in enumerate(self.hvg_indices)}
            for i, gene in enumerate(common_genes):
                new_X[:, gene_indices[gene]] = adata_filtered.X[:, i] if hasattr(adata_filtered.X,
                                                                                 'toarray') else adata_filtered.X[:, i]

            adata_processed = sc.AnnData(new_X)
            adata_processed.obs = adata.obs.copy()
            adata_processed.var_names = self.hvg_indices
        else:
            adata_processed = adata_filtered.copy()
            # 确保基因顺序一致
            adata_processed = adata_processed[:, self.hvg_indices]

        # 应用相同的预处理
        X = adata_processed.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = X.astype(np.float32)
        X = np.log1p(X / np.sum(X, axis=1, keepdims=True) * 1e4)

        if self.scaler_mean is not None and self.scaler_std is not None:
            X = (X - self.scaler_mean) / self.scaler_std
            X = np.clip(X, -10, 10)

        adata_processed.X = X

        return adata_processed

    def save_corrected_adata(self, corrected_adata: sc.AnnData, output_dir: str, batch_key: str = 'batch'):
        """保存校正后的 AnnData 对象 - 使用正确的批次键"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存 h5ad 文件
        h5ad_path = os.path.join(output_dir, "corrected_data.h5ad")
        corrected_adata.write(h5ad_path)

        # 同时保存 CSV 格式以备不时之需
        # csv_path = os.path.join(output_dir, "corrected_data.csv")
        # if hasattr(corrected_adata.X, 'toarray'):
        #     corrected_df = pd.DataFrame(
        #         corrected_adata.X.toarray(),
        #         index=corrected_adata.obs_names,
        #         columns=corrected_adata.var_names
        #     )
        # else:
        #     corrected_df = pd.DataFrame(
        #         corrected_adata.X,
        #         index=corrected_adata.obs_names,
        #         columns=corrected_adata.var_names
        #     )
        # corrected_df.to_csv(csv_path)

        # 保存批次信息 - 使用正确的批次键
        batch_df = pd.DataFrame({
            'cell': corrected_adata.obs_names,
            batch_key: corrected_adata.obs[batch_key]  # 使用传入的批次键
        })
        batch_df.to_csv(os.path.join(output_dir, "batch_info.csv"), index=False)

        print(f"校正数据已保存到 {output_dir}")
        print(f"h5ad 文件: {h5ad_path}")
        print(f"细胞数量: {corrected_adata.n_obs}, 基因数量: {corrected_adata.n_vars}")

