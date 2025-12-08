import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from annoy import AnnoyIndex

from .models import SimpleGenerator, SimpleDiscriminator
from .losses import calculate_gradient_penalty


def train_gan(datasetA, datasetB, original_data, n_epochs=100, batch_size=256, loss_recorder=None):
    """Train simplified GAN for batch correction with loss recording"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Cleaning GPU cache, current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Ensure data is float32 type
    datasetA = datasetA.astype(np.float32)
    datasetB = datasetB.astype(np.float32)
    original_data = original_data.astype(np.float32)

    data_size = datasetA.shape[1]
    n_critic = 5
    lr = 0.0001  # Lower learning rate for stability

    # Initialize simplified GAN
    G = SimpleGenerator(data_size)
    D = SimpleDiscriminator(data_size)

    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Data loader
    dataset = list(zip(datasetA, datasetB))
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    gan_losses = {'d_loss': [], 'g_loss': []}

    for epoch in range(n_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        d_iterations = 0
        g_iterations = 0

        for i, (data_A, data_B) in enumerate(dataloader):
            batch_size = data_A.shape[0]

            # Prepare inputs - ensure correct data type
            real_data = torch.FloatTensor(data_B)  # Now data_B is already float32
            z = torch.FloatTensor(data_A)  # Now data_A is already float32

            if torch.cuda.is_available():
                real_data = real_data.cuda()
                z = z.cuda()

            # Train discriminator
            optimizer_D.zero_grad()

            gen_data = G(z)
            real_validity = D(real_data)
            fake_validity = D(gen_data.detach())

            gradient_penalty = calculate_gradient_penalty(real_data, gen_data, D)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            epoch_d_loss += d_loss.item()
            d_iterations += 1

            # Train generator (every n_critic iterations)
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_data = G(z)
                fake_validity = D(gen_data)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                g_iterations += 1

        # Calculate average losses for the epoch
        avg_d_loss = epoch_d_loss / d_iterations if d_iterations > 0 else 0
        avg_g_loss = epoch_g_loss / g_iterations if g_iterations > 0 else 0

        # Record losses
        gan_losses['d_loss'].append(avg_d_loss)
        gan_losses['g_loss'].append(avg_g_loss)

        # Record to loss recorder if provided
        if loss_recorder is not None:
            loss_recorder.record_gan_loss(avg_d_loss, avg_g_loss)

        if (epoch + 1) % 10 == 0:
            print(f"GAN Epoch {epoch + 1}/{n_epochs}, D loss: {avg_d_loss:.4f}, "
                  f"G loss: {avg_g_loss:.4f}")

    # Clean up training memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Cleaning GPU cache after training, current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Use small batches for inference to avoid memory issues
    G.eval()
    corrected_batches = []
    infer_batch_size = 64  # Use smaller batch size for inference

    with torch.no_grad():
        for i in range(0, len(original_data), infer_batch_size):
            end_idx = min(i + infer_batch_size, len(original_data))
            z_batch = torch.FloatTensor(original_data[i:end_idx])

            if torch.cuda.is_available():
                z_batch = z_batch.cuda()

            corrected_batch = G(z_batch)
            corrected_batches.append(corrected_batch.cpu().numpy())

            # Clean batch memory
            if torch.cuda.is_available():
                del z_batch, corrected_batch
                torch.cuda.empty_cache()

        corrected = np.vstack(corrected_batches)

    # Clean model memory
    if torch.cuda.is_available():
        del G, D
        torch.cuda.empty_cache()

    return corrected, gan_losses


def acquire_mnn_pairs(X, Y, k=50, n_trees=50):
    """Find mutual nearest neighbors pairs"""
    # Ensure input data is float32 type
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)

    f = X.shape[1]
    t1 = AnnoyIndex(f, 'euclidean')
    t2 = AnnoyIndex(f, 'euclidean')

    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])

    t1.build(n_trees)
    t2.build(n_trees)

    mnn_mat = np.zeros((len(X), len(Y)), dtype=bool)
    for i in range(len(X)):
        neighbors = t2.get_nns_by_vector(X[i], k)
        mnn_mat[i, neighbors] = True

    reverse_mnn_mat = np.zeros((len(X), len(Y)), dtype=bool)
    for i in range(len(Y)):
        neighbors = t1.get_nns_by_vector(Y[i], k)
        reverse_mnn_mat[neighbors, i] = True

    mnn_mat = np.logical_and(reverse_mnn_mat, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat > 0))]

    return pairs