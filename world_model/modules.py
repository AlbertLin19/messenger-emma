import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, channel_size, latent_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(in_features=latent_size, out_features=latent_size),
        )

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, channel_size, latent_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=latent_size),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=channel_size, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.network(x)
