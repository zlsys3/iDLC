# batch_correction.py
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
                 n_epochs_gan=100, ae_batch_size=256, gan_batch_size=64, lr=0.001, patience=15):
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
        """Data preprocessing based on AnnData"""
        print("Starting data preprocessing based on AnnData...")

        # Check batch information
        if batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

        print(f"Input data shape: {adata.shape}")
        print(f"Batches: {adata.obs[batch_key].unique().tolist()}")

        # Save original cell names
        original_obs_names = adata.obs_names.copy()

        # Use custom preprocessing - pass batch_key
        adata = custom_data_preprocess(adata, key=batch_key, n_top_genes=self.n_top_genes)

        # Check if any cells were filtered
        if len(adata.obs_names) != len(original_obs_names):
            print(f"Note: Cell count changed from {len(original_obs_names)} to {len(adata.obs_names)} after filtering")

        # Save preprocessing information
        self.hvg_indices = adata.var_names.tolist()

        # Correctly handle sparse matrix statistics calculation
        if hasattr(adata.X, 'toarray'):
            print("Detected sparse matrix, converting to dense matrix for statistics...")
            dense_matrix = adata.X.toarray()
            self.scaler_mean = dense_matrix.mean(axis=0)
            self.scaler_std = dense_matrix.std(axis=0)
            del dense_matrix
        else:
            self.scaler_mean = adata.X.mean(axis=0)
            self.scaler_std = adata.X.std(axis=0)

        # Avoid division by zero
        self.scaler_std = np.where(self.scaler_std == 0, 1, self.scaler_std)

        # Get batch information - use the provided batch_key
        unique_batches = adata.obs[batch_key].unique()
        self.batch_info = {batch: idx for idx, batch in enumerate(unique_batches)}
        self.batch_num = len(self.batch_info)

        print(f"Preprocessing completed, final data shape: {adata.shape}")
        print(f"Number of batches: {self.batch_num}")

        return adata

    def train_autoencoder(self, adata, batch_key: str = 'batch'):
        """Train autoencoder"""
        # Ensure data is dense matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray().astype(np.float32)
        else:
            X = adata.X.astype(np.float32)

        n_cells, n_genes = X.shape

        # Get batch labels and map to numbers - use provided batch_key
        batch_labels = adata.obs[batch_key].map(self.batch_info).values

        # Convert to one-hot encoding
        batch_onehot = np.zeros((n_cells, self.batch_num))
        batch_onehot[np.arange(n_cells), batch_labels] = 1
        batch_onehot = torch.FloatTensor(batch_onehot)

        # Initialize autoencoder
        self.autoencoder = iDLCAutoencoder(n_genes, self.batch_num, self.latent_dim)

        if torch.cuda.is_available():
            self.autoencoder = self.autoencoder.cuda()
            batch_onehot = batch_onehot.cuda()

        # Optimizer
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        early_stopping = SimpleEarlyStopping(patience=self.patience)

        # Prepare data
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), batch_onehot)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.ae_batch_size, shuffle=True, drop_last=True
        )

        # Training loop
        self.autoencoder.train()
        for epoch in range(self.n_epochs_ae):
            total_loss = 0
            total_recon_loss = 0
            total_content_loss = 0
            total_class_loss = 0

            for batch_idx, (x_batch, batch_onehot_batch) in enumerate(dataloader):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    batch_onehot_batch = batch_onehot_batch.cuda()

                optimizer.zero_grad()

                # Forward pass
                encoded, decoded, bio_part = self.autoencoder(x_batch)

                # Split encoding result
                batch_part = encoded[:, :self.batch_num]

                # Calculate reconstruction loss
                recon_loss = criterion(decoded, x_batch)

                # Calculate classification loss
                class_loss = classification_criterion(batch_part, torch.argmax(batch_onehot_batch, dim=1))

                # Generate random batch noise
                random_batch_noise = torch.randn_like(batch_onehot_batch)
                random_indices = torch.randint(0, self.batch_num, (x_batch.size(0),))
                if torch.cuda.is_available():
                    random_indices = random_indices.cuda()
                random_batch_noise.zero_().scatter_(1, random_indices.unsqueeze(1), 1)

                # Reconstruct using random batch noise
                random_reconstructed = self.autoencoder.decoder(
                    torch.cat([random_batch_noise, bio_part], dim=1)
                )

                # Encode random reconstruction again
                new_encoded, _, new_bio_part = self.autoencoder(random_reconstructed)

                # Calculate content loss
                content_loss = criterion(new_bio_part, bio_part)

                # Total loss - simplified version
                loss = recon_loss + 0.5 * content_loss + 0.5 * class_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_content_loss += content_loss.item()
                total_class_loss += class_loss.item()

            # Calculate average losses
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_content = total_content_loss / len(dataloader)
            avg_class = total_class_loss / len(dataloader)

            # Record three types of losses
            self.loss_recorder.record_ae_loss(avg_loss, avg_recon, avg_content, avg_class)

            # Learning rate scheduling
            scheduler.step(avg_loss)

            # Early stopping check
            if early_stopping(avg_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"AE Epoch {epoch + 1}/{self.n_epochs_ae}, Loss: {avg_loss:.4f}, "
                      f"Recon: {avg_recon:.4f}, Content: {avg_content:.4f}, "
                      f"Class: {avg_class:.4f}, LR: {current_lr:.6f}")

        self.is_trained = True

    def extract_features(self, X):
        """Extract features"""
        self.autoencoder.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                X = torch.FloatTensor(X).cuda()
            else:
                X = torch.FloatTensor(X)

            _, _, bio_part = self.autoencoder(X)
            return bio_part.cpu().numpy()

    def save_model(self, filepath):
        """Save model and preprocessing parameters"""
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
        """Load model and preprocessing parameters"""
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

    def correct(self, adata: sc.AnnData, batch_key: str = 'batch', save_dir: str = None,
                model_path: str = None, plot_loss: bool = True, k: int = 50,
                use_ot: bool = False, ot_method='mmd', ot_weight=0.1) -> sc.AnnData:
        """Main correction function - fully based on AnnData"""

        print("Starting batch correction pipeline...")

        # If model path provided, load model
        if model_path and os.path.exists(model_path):
            print("Loading pre-trained model...")
            self.load_model(model_path)
            # Apply preprocessing to input data
            adata_processed = self.apply_preprocessing_to_adata(adata, batch_key)
        else:
            # Data preprocessing
            print("Preprocessing data...")
            adata_processed = self.preprocess_data(adata, batch_key)

            # Train autoencoder
            print("Training autoencoder...")
            self.train_autoencoder(adata_processed, batch_key)

        print(f"Processed data shape: {adata_processed.shape}")

        # Ensure data is dense matrix
        if hasattr(adata_processed.X, 'toarray'):
            X_all = adata_processed.X.toarray().astype(np.float32)
        else:
            X_all = adata_processed.X.astype(np.float32)

        # Extract latent representations for all cells
        print("Extracting biological latent representations...")
        all_latent = self.extract_features(X_all)
        print(f"Latent representation shape: {all_latent.shape}")

        # Group original data and latent representations by batch
        batch_groups = {}
        latent_groups = {}
        unique_batches = adata_processed.obs[batch_key].unique()

        for batch in unique_batches:
            batch_mask = adata_processed.obs[batch_key] == batch
            # Original data
            batch_groups[batch] = X_all[batch_mask]
            # Latent representations
            latent_groups[batch] = all_latent[batch_mask]

        # Select reference batch
        ref_batch = max(batch_groups.keys(), key=lambda x: len(batch_groups[x]))
        print(f"Reference batch: {ref_batch}")

        # Correct each batch
        corrected_data = {}
        reference_set = batch_groups[ref_batch].copy()
        reference_latent = latent_groups[ref_batch].copy()

        # Process reference batch
        corrected_data[ref_batch] = batch_groups[ref_batch]

        # Correct other batches
        batches_to_correct = [batch for batch in unique_batches if batch != ref_batch]

        for batch in batches_to_correct:
            print(f"Correcting batch: {batch} relative to reference batch {ref_batch}")

            # Use latent representations to search for MNN pairs
            print("Searching for MNN pairs using latent representations...")
            pairs = acquire_mnn_pairs(reference_latent, latent_groups[batch], k=k)

            if len(pairs) < 10:
                print(f"Insufficient MNN pairs ({len(pairs)}), using direct correction")
                corrected_data[batch] = batch_groups[batch]
            else:
                # Prepare GAN training data
                datasetA = batch_groups[batch][[y for x, y in pairs]]
                datasetB = reference_set[[x for x, y in pairs]]

                # Ensure data is float32 type
                datasetA = datasetA.astype(np.float32)
                datasetB = datasetB.astype(np.float32)
                batch_groups_batch_float32 = batch_groups[batch].astype(np.float32)

                # Prepare latent representations for OT loss
                latent_A = latent_groups[batch][[y for x, y in pairs]]
                latent_B = reference_latent[[x for x, y in pairs]]

                # Choose whether to use OT regularization
                if use_ot:
                    print(f"Using OT-regularized GAN for batch correction (method: {ot_method}, weight: {ot_weight})")
                    corrected_batch, _ = train_gan_with_ot(
                        datasetA,
                        datasetB,
                        batch_groups_batch_float32,
                        n_epochs=self.n_epochs_gan,
                        batch_size=self.gan_batch_size,
                        loss_recorder=self.loss_recorder,
                        ot_weight=ot_weight,
                        ot_method=ot_method,
                        autoencoder=self.autoencoder,  # Pass autoencoder
                        latent_A=latent_A,  # Pass latent A
                        latent_B=latent_B  # Pass latent B
                    )
                else:
                    print("Using original GAN for batch correction")
                    corrected_batch, _ = train_gan(
                        datasetA,
                        datasetB,
                        batch_groups_batch_float32,
                        n_epochs=self.n_epochs_gan,
                        batch_size=self.gan_batch_size,
                        loss_recorder=self.loss_recorder
                    )

                corrected_data[batch] = corrected_batch

            # Update reference set and latent representations
            reference_set = np.vstack([reference_set, corrected_data[batch]])
            # Re-extract latent representations from corrected data
            corrected_latent = self.extract_features(corrected_data[batch])
            reference_latent = np.vstack([reference_latent, corrected_latent])

        # Merge all corrected data
        all_corrected = []
        all_corrected_labels = []
        all_corrected_cell_names = []

        for batch in unique_batches:
            all_corrected.append(corrected_data[batch])
            batch_mask = adata_processed.obs[batch_key] == batch
            all_corrected_labels.extend([batch] * len(corrected_data[batch]))
            all_corrected_cell_names.extend(adata_processed.obs_names[batch_mask].tolist())

        corrected_matrix = np.vstack(all_corrected)

        print(f"Final corrected matrix shape: {corrected_matrix.shape}")
        print(f"Final cell names count: {len(all_corrected_cell_names)}")

        # Final check
        if len(all_corrected_cell_names) != corrected_matrix.shape[0]:
            raise ValueError(
                f"Cell names count ({len(all_corrected_cell_names)}) doesn't match data row count ({corrected_matrix.shape[0]})!")

        # Create corrected AnnData object
        corrected_adata = sc.AnnData(
            scipy.sparse.csr_matrix(corrected_matrix)  # Use sparse matrix to save space
        )
        corrected_adata.obs_names = all_corrected_cell_names
        corrected_adata.var_names = self.hvg_indices
        corrected_adata.obs[batch_key] = all_corrected_labels

        # Save corrected data
        if save_dir:
            self.save_corrected_adata(corrected_adata, save_dir, batch_key)

        # Save model
        if model_path and not os.path.exists(model_path):
            self.save_model(model_path)

        # Plot loss curves
        if plot_loss:
            loss_plot_path = os.path.join(save_dir, "training_losses.png") if save_dir else None
            self.loss_recorder.plot_losses(loss_plot_path)

        return corrected_adata

    def apply_preprocessing_to_adata(self, adata: sc.AnnData, batch_key: str = 'batch') -> sc.AnnData:
        """Apply preprocessing to new data"""
        if self.hvg_indices is None:
            raise ValueError("Preprocessing parameters not available. Please train or load model first.")

        print("Applying preprocessing to new data...")

        # Keep only highly variable genes
        common_genes = adata.var_names.intersection(self.hvg_indices)
        adata_filtered = adata[:, common_genes]

        # Add missing highly variable genes (filled with 0)
        missing_genes = set(self.hvg_indices) - set(common_genes)
        if missing_genes:
            print(f"Adding {len(missing_genes)} missing genes (filled with 0)")
            # Create new AnnData object containing all highly variable genes
            new_X = np.zeros((adata.n_obs, len(self.hvg_indices)), dtype=np.float32)

            # Fill data for existing genes
            gene_indices = {gene: idx for idx, gene in enumerate(self.hvg_indices)}
            for i, gene in enumerate(common_genes):
                new_X[:, gene_indices[gene]] = adata_filtered.X[:, i] if hasattr(adata_filtered.X,
                                                                                 'toarray') else adata_filtered.X[:, i]

            adata_processed = sc.AnnData(new_X)
            adata_processed.obs = adata.obs.copy()
            adata_processed.var_names = self.hvg_indices
        else:
            adata_processed = adata_filtered.copy()
            # Ensure gene order consistency
            adata_processed = adata_processed[:, self.hvg_indices]

        # Apply same preprocessing
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
        """Save corrected AnnData object - using correct batch key"""
        os.makedirs(output_dir, exist_ok=True)

        # Save h5ad file
        h5ad_path = os.path.join(output_dir, "corrected_data.h5ad")
        corrected_adata.write(h5ad_path)

        # Save batch information - using correct batch key
        batch_df = pd.DataFrame({
            'cell': corrected_adata.obs_names,
            batch_key: corrected_adata.obs[batch_key]  # Use provided batch key
        })
        batch_df.to_csv(os.path.join(output_dir, "batch_info.csv"), index=False)

        print(f"Corrected data saved to {output_dir}")
        print(f"h5ad file: {h5ad_path}")
        print(f"Cell count: {corrected_adata.n_obs}, Gene count: {corrected_adata.n_vars}")