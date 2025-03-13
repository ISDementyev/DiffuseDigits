import torch

def corrupt(x, intensity):
    noise = torch.randn_like(x)
    intensity = intensity.view(-1, 1, 1, 1)

    return x * (1 - intensity) + intensity * noise