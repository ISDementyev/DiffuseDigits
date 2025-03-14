import torch
import torch.nn as nn


def corrupt(x, intensity):
    noise = torch.randn_like(x)
    intensity = intensity.view(-1, 1, 1, 1)

    return x * (1 - intensity) + intensity * noise

class UNet(nn.Module):
    """Simple UNet Application"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )

        self.silu = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

        def forward(self, x):
            h = []
            for i, layer in enumerate(self.down_layers):
                x = self.silu(layer(x)) # go through layer and activation function
                if i < 2:
                    h.append(x) # store out for skipping in UNet
                    x = self.downscale(x)

            for i, layer in enumerate(self.up_layers):
                if i > 0: # for all but the first uplayer
                    x = self.upscale(x)
                    x += h.pop()
                x = self.silu(layer(x))