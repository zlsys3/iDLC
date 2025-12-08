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