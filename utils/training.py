import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from torch import nn
from torch.utils.data import DataLoader, Dataset


def plot_progress(train_losses: list[float], val_losses: list[float]) -> None:
    """Plots train and val lossse per batch.

    Args:
        train_losses (list[float]): Train loss list
        val_losses (list[float]): Validataion loss list
    """
    clear_output(True)

    f, (ax1) = plt.subplots(nrows=1, ncols=1)
    f.set_figheight(6)
    f.set_figwidth(20)

    ax1.plot(train_losses, label="train loss")
    ax1.plot(val_losses, label="val loss")
    ax1.plot(np.zeros_like(train_losses), "--", label="zero")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Batch number")
    ax1.legend()

    plt.show()


class EmbeddingDataset(Dataset):
    """Dataset for embeddings."""

    def __init__(
        self, embeddings: torch.Tensor, targets: pd.DataFrame | None = None
    ):
        self.embeddings = embeddings
        self.targets = None
        if targets is not None:
            self.targets = torch.tensor(targets.values).float()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.embeddings[idx, :], self.targets[idx, :]
        else:
            return self.embeddings[idx, :]


def train_loop(
    n_epochs: int,
    loss_func: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
):
    """Train loop for MLP training with progress plotting.

    Args:
        n_epochs (int): Number of epochs.
        loss_func (nn.Module): Loss function instance.
        model (nn.Module): Model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        train_dataloader (DataLoader): Train dataloader.
        val_dataloader (DataLoader): Validation dataloader.
        device (torch.device): Torch device.
    """
    losses = []
    val_losses = []
    for i in range(n_epochs):
        for j, (x_train, y_train) in enumerate(train_dataloader):
            x_train = x_train.to(device)
            y_train = y_train.float().to(device)

            model.train()
            preds = model(x_train)
            train_loss = loss_func(preds, y_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            model.eval()

            x_val, y_val = next(iter(val_dataloader))

            x_val = x_val.to(device)
            y_val = y_val.float().to(device)
            with torch.no_grad():
                val_preds = model(x_val)
                val_loss = loss_func(val_preds, y_val)

                losses.append(train_loss.item())

                val_losses.append(val_loss.item())

                plot_progress(losses, val_losses)


def get_mlp_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> np.ndarray:
    """_summary_

    Args:
        model (nn.Module): Model instance.
        dataloader (DataLoader): Dataloader.
        device (torch.device): Torch device.

    Returns:
        np.ndarray: 2D predictions array.
    """
    mlp_preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            preds = model(x)
            mlp_preds.append(preds)
    mlp_preds = torch.cat(mlp_preds)
    mlp_preds = mlp_preds.cpu().detach().numpy()
    return mlp_preds
