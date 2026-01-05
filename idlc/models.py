# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResidualBlock(nn.Module):
    def __init__(self, dim):
        super(SimpleResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class iDLCAutoencoder(nn.Module):
    def __init__(self, input_dim: int, batch_num: int, latent_dim: int = 256):
        super(iDLCAutoencoder, self).__init__()
        self.batch_num = batch_num
        self.latent_dim = latent_dim

        # Encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            SimpleResidualBlock(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SimpleResidualBlock(512),
            nn.Linear(512, batch_num + latent_dim)
        )

        # Decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(batch_num + latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SimpleResidualBlock(512),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            SimpleResidualBlock(1024),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x, batch_noise=None):
        # Encoding
        encoded = self.encoder(x)

        # Split into batch noise and biological information
        batch_part = encoded[:, :self.batch_num]
        bio_part = encoded[:, self.batch_num:]

        # Use provided batch noise if available
        if batch_noise is not None:
            batch_part = batch_noise

        # Recombine
        combined = torch.cat([batch_part, bio_part], dim=1)

        # Decoding
        decoded = self.decoder(combined)

        return encoded, decoded, bio_part


class SimpleGenerator(nn.Module):
    def __init__(self, input_dim):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return torch.relu(self.model(x) + x)  # Residual connection


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


# Add the following classes to models.py

class CriticNetwork(nn.Module):
    """Critic network for learning feature embeddings and transport costs"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=128):
        super(CriticNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # Output layer, generate feature embeddings
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Return feature embeddings"""
        return self.network(x)

    def compute_cosine_cost(self, x1, x2):
        """Calculate cosine distance as transport cost"""
        feat1 = self.forward(x1)
        feat2 = self.forward(x2)

        # Normalize
        feat1_norm = feat1 / (feat1.norm(dim=1, keepdim=True) + 1e-8)
        feat2_norm = feat2 / (feat2.norm(dim=1, keepdim=True) + 1e-8)

        # Cosine distance: 1 - cosine_similarity
        cost = 1 - (feat1_norm * feat2_norm).sum(dim=1)
        return cost


class OTGenerator(SimpleGenerator):
    """OT-GAN generator, inherits from SimpleGenerator but adds feature matching loss"""

    def __init__(self, input_dim):
        super(OTGenerator, self).__init__(input_dim)

    def feature_matching_loss(self, real_features, generated_features):
        """Feature matching loss, ensuring generated data feature distribution matches real data"""
        return torch.mean(torch.abs(real_features - generated_features))

    # Ensure forward method correctly handles device
    def forward(self, x):
        # Ensure input x and model weights are on the same device
        return torch.relu(self.model(x) + x)  # Residual connection