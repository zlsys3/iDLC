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

        #plt.show()

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

        #plt.show()


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