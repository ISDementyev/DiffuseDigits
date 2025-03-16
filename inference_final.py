import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from typing import Dict
import ray
import tempfile
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from filelock import FileLock

# setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
mnist_mean, mnist_std = 0.1307, 0.3081
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mnist_mean,), (mnist_std,))])
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=transform)

from time import time
import os

torch.manual_seed(129)  # for reproducibility

dirname = "diffusion_models_serial_1"
os.makedirs(dirname, exist_ok=True)


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


# net = BasicUNet()
# x = torch.rand(8, 1, 28, 28)
# net(x).shape

def load_data(data_dir="./mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,)),
    ])

    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (1, 28, 28), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (1, 28, 28), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


def train_diffusion(config):
    net = BasicUNet()
    net.to(device)

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    # Load existing checkpoint through `get_checkpoint()` API.
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            opt.load_state_dict(optimizer_state)

    if config["smoke_test"]:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()

    train_dataloader = DataLoader(dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=True,
                                  num_workers=8)
    losses = []
    for epoch in range(config["n_epochs"]):
        for x, y in train_dataloader:
            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount)  # Create our noisy x

            # Get the model prediction
            pred = net(noisy_x)

            # Calculate the loss
            loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

        # if config["n_epochs"] - 5 <= epoch <= config["n_epochs"] - 1:
        #     last_5_losses.append(avg_loss)

    last_5_losses_avg = sum(losses[-5:]) / len(losses[-5:])
    print(last_5_losses_avg)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            (net.state_dict(), opt.state_dict()), path
        )
        checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report(
            {"loss": last_5_losses_avg},
            checkpoint=checkpoint)
    print("Training finished!")


def main(num_samples=50, gpus_per_trial=1, smoke_test=False):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "n_epochs": tune.choice([5, 10, 15, 20, 25]),
        "smoke_test": smoke_test,
    }
    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_diffusion),
            resources={"cpu": 16, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    # test_best_model(best_result, smoke_test=smoke_test)

# This trains the UNet, given a list of different hyperparameters (e.g. epoch list, etc.)
def train_variants(batch_size_list, learning_rates_list, epoch_list):
    final_unets = []
    unets_and_last_5_epochs = []

    for batch_size in batch_size_list:
        for lr in learning_rates_list:
            net = BasicUNet()
            net.to(device)
            t1 = time()

            for n_epochs in epoch_list:
                train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                loss_fn = nn.MSELoss()

                # The optimizer
                opt = torch.optim.Adam(net.parameters(), lr=lr)

                # Keeping a record of the losses for later viewing
                losses = []
                last_5_losses = []

                # The training loop
                for epoch in range(n_epochs):
                    for x, y in train_dataloader:
                        # Get some data and prepare the corrupted version
                        x = x.to(device)  # Data on the GPU
                        noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
                        noisy_x = corrupt(x, noise_amount)  # Create our noisy x

                        # Get the model prediction
                        pred = net(noisy_x)

                        # Calculate the loss
                        loss = loss_fn(pred, x)

                        # Backprop and update the params:
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        # Store the loss for later
                        losses.append(loss.item())

                    # Print our the average of the loss values for this epoch:
                    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
                    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

                    if n_epochs - 5 <= epoch <= n_epochs - 1:
                        last_5_losses.append(avg_loss)

                last_5_loss_average = sum(last_5_losses) / len(last_5_losses)
                # View the loss curve
                plt.plot(losses)
                plt.title(
                    f"Losses of Diffusion UNet with batch size {batch_size}, {n_epochs} epochs, learning rate of {lr}")
                plt.ylabel("Loss")
                plt.xlabel("Training step")
                plt.savefig(f"./{dirname}/loss-batch_size{batch_size}-epochs{n_epochs}-lr{lr}.png",
                            dpi=600, bbox_inches='tight')
                plt.ylim(0, 0.2)
                plt.show()

            final_unets.append(net)
            unets_and_last_5_epochs.append((net, last_5_loss_average, time() - t1))

    return final_unets, unets_and_last_5_epochs


t1 = time()
print("Begin")
# main()
batch_size_list = [16, 32, 64, 128]
learning_rates_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
epoch_list = [5, 10, 15, 20, 25]
all_trained_models = train_variants(batch_size_list, learning_rates_list, epoch_list)
print(f"Final time: {time() - t1} s")

# best hyperparameters (batch_size 16, learning_rate 5e-4, n_epochs 10)
# Final Loss (Goal set of <0.1): 0.069