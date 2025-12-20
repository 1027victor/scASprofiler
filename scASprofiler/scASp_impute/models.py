from __future__ import annotations

import math

from .utils import weights_init_normal

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Dynamic hidden_dim convolutional encoder.

    Refactor of your updated Encoder (3 conv blocks, stride=2).
    """

    def __init__(
        self,
        img_size: int,
        channels: int,
        latent_dim: int,
        base_channels: int = 32,
        hidden_dim: int | None = None,
        hidden_dim_ratio: float = 0.25,
        hidden_dim_min: int = 128,
        hidden_dim_max: int = 1024,
    ) -> None:
        super().__init__()

        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
        )

        conv_out_res = math.ceil(img_size / 8)
        self.flat_dim = (c * 4) * (conv_out_res**2)

        if hidden_dim is None:
            hidden_dim = int(self.flat_dim * hidden_dim_ratio)
            hidden_dim = max(hidden_dim_min, min(hidden_dim, hidden_dim_max))
        self.hidden_dim = int(hidden_dim)

        self.fc_hidden = nn.Sequential(
            nn.Linear(self.flat_dim, self.hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc_hidden(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size: int, channels: int, latent_dim: int, ncls: int):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.ncls = ncls

        self.cn1 = 32
        self.l1p = nn.Sequential(
            nn.Linear(latent_dim, self.cn1 * (img_size**2)))
        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )
        self.conv_blocks_02p = nn.Sequential(
            nn.Upsample(scale_factor=img_size),
            nn.Conv2d(ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )
        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, label_oh: torch.Tensor) -> torch.Tensor:
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, self.img_size, self.img_size)
        out01 = self.conv_blocks_01p(out)

        label_oh = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks_02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return out1


class Discriminator(nn.Module):
    def __init__(self, img_size: int, channels: int, ncls: int):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.ncls = ncls

        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32

        self.pre = nn.Sequential(
            nn.Linear(img_size**2 * (channels + 1),
                      (channels + 1) * self.down_size0**2)
        )

        self.down = nn.Sequential(
            nn.Conv2d(channels + 1, self.cn1,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1 // 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(),
        )

        self.conv_blocks02p = nn.Sequential(
            nn.Upsample(scale_factor=self.down_size),
            nn.Conv2d(ncls, self.cn1 // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        down_dim = (self.cn1 // 2 + self.cn1 // 4) * (self.down_size**2)
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(),
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(self.cn1 // 2 + self.cn1 // 4, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, channels, kernel_size=2),
            nn.Sigmoid(),
        )

        self.to_full_res = nn.Upsample(
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, img_mask: torch.Tensor, label_oh: torch.Tensor) -> torch.Tensor:
        bs = img_mask.size(0)

        out00 = self.pre(img_mask.view(bs, -1)).view(
            bs, self.channels + 1, self.down_size0, self.down_size0
        )
        out01 = self.down(out00)

        label_map = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks02p(label_map)

        out1 = torch.cat((out01, out02), dim=1)

        out_fc = self.fc(out1.view(bs, -1))

        out_low_res = self.up(
            out_fc.view(bs, (self.cn1 // 2 + self.cn1 // 4),
                        self.down_size, self.down_size)
        )

        out_full_res = self.to_full_res(out_low_res)
        return out_full_res


class VAEGAN(nn.Module):
    def __init__(self, img_size: int, channels: int, latent_dim: int, ncls: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            img_size=img_size,
            channels=channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            img_size=img_size,
            channels=channels,
            latent_dim=latent_dim,
            ncls=ncls,
        )

        self.discriminator = Discriminator(
            img_size=img_size,
            channels=channels,
            ncls=ncls,
        )

        self.encoder.apply(weights_init_normal)
        self.decoder.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def forward(self, x: torch.Tensor, label_oh: torch.Tensor):
        bs = x.size(0)
        z_mean, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()
        epsilon = torch.randn(bs, self.latent_dim).to(x.device)

        z = z_mean + std * epsilon
        x_tilda = self.decoder(z, label_oh)
        return z_mean, z_logvar, x_tilda
