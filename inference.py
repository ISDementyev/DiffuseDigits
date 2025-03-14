import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt

seed = 1
torch.manual_seed(seed)

# Check for MacOS metal, if unavailable check for GPUs, if unavailable then implement standard CPU.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load digits dataset (monochrome) from MNIST
dataset = torchvision.datasets.MNIST(
    root="/Volumes/T9/mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
)

# load dataset into a dataloader, higher batch size
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

if __name__ == "__main__":
    x, y = next(iter(train_dataloader))
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].set_title("Input data")
    # axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
    # plt.show()


