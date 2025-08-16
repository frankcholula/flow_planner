import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 12x12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 6x6
            nn.ReLU(),
            nn.Flatten(),
        )

        # We need to calculate the flattened size after the conv layers
        # For a 96x96 input, it becomes 256 * 6 * 6 = 9216
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_log_var = nn.Linear(256 * 6 * 6, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 6 * 6)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # -> 96x96
            nn.Sigmoid(),  # Sigmoid to output pixel values between 0 and 1
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 6, 6)  # Reshape to image format
        return self.decoder(result)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var


# VAE Loss Function
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    # Reconstruction Loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + beta * KLD
