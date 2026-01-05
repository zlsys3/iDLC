# losses.py
import torch
import matplotlib.pyplot as plt
import numpy as np


def calculate_gradient_penalty(real_data, fake_data, D):
    """Calculate gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_data.size(0), 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    prob_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda() if torch.cuda.is_available() else
        torch.ones(prob_interpolated.size()),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class ImprovedLossRecorder:
    def __init__(self):
        self.ae_losses = {
            'total': [],
            'reconstruction': [],
            'content': [],
            'classification': []
        }
        self.gan_d_losses = []
        self.gan_g_losses = []

    def record_ae_loss(self, total, recon, content, class_loss):
        self.ae_losses['total'].append(total)
        self.ae_losses['reconstruction'].append(recon)
        self.ae_losses['content'].append(content)
        self.ae_losses['classification'].append(class_loss)

    def record_gan_loss(self, d_loss, g_loss):
        self.gan_d_losses.append(d_loss)
        self.gan_g_losses.append(g_loss)

    def plot_losses(self, save_path=None):
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Autoencoder total loss
        epochs_ae = range(1, len(self.ae_losses['total']) + 1)
        axes[0, 0].plot(epochs_ae, self.ae_losses['total'], label='Total Loss', color='blue', linewidth=2)
        axes[0, 0].set_title('Autoencoder Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Autoencoder loss components
        axes[0, 1].plot(epochs_ae, self.ae_losses['reconstruction'], label='Reconstruction Loss', color='red')
        axes[0, 1].plot(epochs_ae, self.ae_losses['content'], label='Content Loss', color='green')
        axes[0, 1].plot(epochs_ae, self.ae_losses['classification'], label='Classification Loss', color='orange')
        axes[0, 1].set_title('Autoencoder Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # GAN losses
        epochs_gan = range(1, len(self.gan_d_losses) + 1)
        axes[1, 0].plot(epochs_gan, self.gan_d_losses, label='Discriminator Loss', color='purple')
        axes[1, 0].plot(epochs_gan, self.gan_g_losses, label='Generator Loss', color='brown')
        axes[1, 0].set_title('GAN Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Loss component distribution (pie chart)
        if len(epochs_ae) > 0:
            # Calculate average loss proportions
            avg_recon = np.mean(self.ae_losses['reconstruction'])
            avg_content = np.mean(self.ae_losses['content'])
            avg_class = np.mean(self.ae_losses['classification'])
            total_avg = avg_recon + avg_content + avg_class

            if total_avg > 0:
                sizes = [avg_recon / total_avg, avg_content / total_avg, avg_class / total_avg]
                labels = ['Reconstruction\n{:.1f}%'.format(sizes[0] * 100),
                          'Content\n{:.1f}%'.format(sizes[1] * 100),
                          'Classification\n{:.1f}%'.format(sizes[2] * 100)]
                colors = ['lightcoral', 'lightgreen', 'lightsalmon']

                axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                               startangle=90)
                axes[1, 1].set_title('Average Loss Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")

        # plt.show()

        # Additional detailed loss curves
        self.plot_detailed_losses(save_path)

    def plot_detailed_losses(self, save_path=None):
        """Plot more detailed loss curves"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # All autoencoder losses in one plot
        epochs_ae = range(1, len(self.ae_losses['total']) + 1)

        axes[0].plot(epochs_ae, self.ae_losses['total'], label='Total Loss', linewidth=3, color='black')
        axes[0].plot(epochs_ae, self.ae_losses['reconstruction'], label='Reconstruction', linestyle='--')
        axes[0].plot(epochs_ae, self.ae_losses['content'], label='Content', linestyle='--')
        axes[0].plot(epochs_ae, self.ae_losses['classification'], label='Classification', linestyle='--')
        axes[0].set_title('Autoencoder Detailed Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Losses on log scale
        axes[1].semilogy(epochs_ae, self.ae_losses['total'], label='Total Loss', linewidth=3, color='black')
        axes[1].semilogy(epochs_ae, self.ae_losses['reconstruction'], label='Reconstruction', linestyle='--')
        axes[1].semilogy(epochs_ae, self.ae_losses['content'], label='Content', linestyle='--')
        axes[1].semilogy(epochs_ae, self.ae_losses['classification'], label='Classification', linestyle='--')
        axes[1].set_title('Autoencoder Loss Curves (Log Scale)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log scale)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            detailed_path = save_path.replace('.png', '_detailed.png')
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            print(f"Detailed loss plot saved to {detailed_path}")

        # plt.show()


class SimpleEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


# Add to the end of losses.py

class OTLoss:
    """Optimal Transport loss, can be added as regularization to existing GAN"""

    def __init__(self, method='mmd', weight=0.1, **kwargs):
        self.method = method
        self.weight = weight
        self.kwargs = kwargs

    def __call__(self, real_data, fake_data):
        """Calculate OT distance between real and generated data"""
        if self.method == 'mmd':
            ot_loss = compute_mmd_distance(real_data, fake_data, **self.kwargs)
        elif self.method == 'wasserstein':
            ot_loss = compute_wasserstein_distance_simple(real_data, fake_data)
        elif self.method == 'energy':
            ot_loss = compute_energy_distance(real_data, fake_data)
        elif self.method == 'sinkhorn':
            ot_loss = sinkhorn_distance_simple(real_data, fake_data, **self.kwargs)
        else:
            raise ValueError(f"Unknown OT method: {self.method}")

        return self.weight * ot_loss


# Add some helper functions (if already defined in ot_utils.py, no need to repeat here)
def compute_mmd_distance(x, y, kernel='rbf', sigma=1.0):
    """Calculate Maximum Mean Discrepancy (MMD)"""
    # Implementation same as in ot_utils.py
    if kernel == 'rbf':
        xx = torch.exp(-torch.cdist(x, x) ** 2 / (2 * sigma ** 2))
        yy = torch.exp(-torch.cdist(y, y) ** 2 / (2 * sigma ** 2))
        xy = torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))
    else:  # Linear kernel
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return torch.clamp(mmd, min=0.0)


def compute_wasserstein_distance_simple(x, y):
    """Simplified Wasserstein distance calculation"""
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def compute_energy_distance(x, y):
    """Calculate energy distance"""
    xx = torch.mean(torch.cdist(x, x, p=2))
    yy = torch.mean(torch.cdist(y, y, p=2))
    xy = torch.mean(torch.cdist(x, y, p=2))
    energy_dist = 2 * xy - xx - yy
    return torch.clamp(energy_dist, min=0.0)


def sinkhorn_distance_simple(x, y, eps=1, n_iter=50):
    """Simplified Sinkhorn distance calculation"""
    n = x.shape[0]
    m = y.shape[0]
    C = torch.cdist(x, y, p=2) ** 2
    C = C / (C.max() + 1e-8)
    K = torch.exp(-C / eps)
    u = torch.ones(n, 1, device=x.device) / n
    v = torch.ones(m, 1, device=y.device) / m

    for _ in range(n_iter):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.t() @ u + 1e-8)

    P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
    ot_dist = (P * C).sum()
    return ot_dist


class OTLossWithAutoencoder:
    """Optimal Transport loss using autoencoder latent representations"""

    def __init__(self, autoencoder, method='mmd', weight=0.1, **kwargs):
        self.autoencoder = autoencoder
        self.method = method
        self.weight = weight
        self.kwargs = kwargs

    def __call__(self, real_data, fake_data, latent_data=None):
        """Calculate OT distance between real and generated data"""
        if self.method == 'mmd':
            ot_loss = compute_mmd_distance_with_autoencoder(real_data, fake_data, self.autoencoder, **self.kwargs)
        elif self.method == 'wasserstein':
            ot_loss = compute_wasserstein_distance_with_autoencoder(real_data, fake_data, self.autoencoder)
        elif self.method == 'energy':
            ot_loss = compute_energy_distance_with_autoencoder(real_data, fake_data, self.autoencoder)
        elif self.method == 'sinkhorn':
            # If pre-computed latent data is available, use it
            if latent_data is not None:
                ot_loss = sinkhorn_distance_with_latent(real_data, fake_data, self.autoencoder, latent_data,
                                                        **self.kwargs)
            else:
                ot_loss = sinkhorn_distance_with_autoencoder(real_data, fake_data, self.autoencoder, **self.kwargs)
        else:
            raise ValueError(f"Unknown OT method: {self.method}")

        return self.weight * ot_loss


# Add helper functions
def compute_mmd_distance_with_autoencoder(x, y, autoencoder, kernel='rbf', sigma=1.0):
    """Calculate MMD using features extracted by autoencoder"""
    with torch.no_grad():
        _, _, x_latent = autoencoder(x)
        _, _, y_latent = autoencoder(y)

    # Calculate MMD using extracted latent features
    return compute_mmd_distance(x_latent, y_latent, kernel=kernel, sigma=sigma)


def compute_wasserstein_distance_with_autoencoder(x, y, autoencoder):
    """Calculate Wasserstein distance using features extracted by autoencoder"""
    with torch.no_grad():
        _, _, x_latent = autoencoder(x)
        _, _, y_latent = autoencoder(y)

    # Calculate L2 distance of sorted features
    x_sorted, _ = torch.sort(x_latent, dim=0)
    y_sorted, _ = torch.sort(y_latent, dim=0)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def compute_energy_distance_with_autoencoder(x, y, autoencoder):
    """Calculate energy distance using features extracted by autoencoder"""
    with torch.no_grad():
        _, _, x_latent = autoencoder(x)
        _, _, y_latent = autoencoder(y)

    # Calculate energy distance
    return compute_energy_distance(x_latent, y_latent)


def sinkhorn_distance_with_autoencoder(x, y, autoencoder, eps=1, n_iter=50):
    """Calculate Sinkhorn distance using features extracted by autoencoder"""
    with torch.no_grad():
        _, _, x_latent = autoencoder(x)
        _, _, y_latent = autoencoder(y)

    return sinkhorn_distance_simple(x_latent, y_latent, eps=eps, n_iter=n_iter)


def sinkhorn_distance_with_latent(x, y, autoencoder, real_latent, eps=1, n_iter=50):
    """Calculate Sinkhorn distance using pre-computed real data latent representations and generated data latent representations extracted from autoencoder"""
    # Extract latent representations of generated data
    with torch.no_grad():
        _, _, fake_latent = autoencoder(y)

    n = real_latent.shape[0]
    m = fake_latent.shape[0]

    # Calculate cost matrix (based on Euclidean distance of latent representations)
    C = torch.cdist(real_latent, fake_latent, p=2) ** 2

    # Normalize
    C = C / (C.max() + 1e-8)

    # Sinkhorn iteration
    K = torch.exp(-C / eps)
    u = torch.ones(n, 1, device=real_latent.device) / n
    v = torch.ones(m, 1, device=fake_latent.device) / m

    for _ in range(n_iter):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.t() @ u + 1e-8)

    # Calculate transport distance
    P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
    ot_dist = (P * C).sum()

    return ot_dist