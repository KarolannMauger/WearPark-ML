"""Training pipeline for the WearPark CNN 1D model.

Trains a binary Parkinson tremor classifier on combined PADS + Monipar data.
The best checkpoint (lowest validation loss) is saved to ``models/``.

Techniques applied:
    - **Mixup augmentation** (Zhang et al., 2018): linearly interpolates pairs
      of training samples to reduce over-confidence and improve generalisation.
    - **Gaussian noise**: additive white noise on each batch for robustness.
    - **Gyro channel masking**: handled upstream in the dataset loader.
    - **CosineAnnealingLR**: smoothly decays the learning rate over all epochs.
    - **Gradient clipping** (max norm 1.0): prevents gradient explosion.
    - **Early stopping**: halts training when validation loss stalls.

Usage:
    python train.py

Outputs:
    models/wearpark_cnn1d_best.pt  — best model checkpoint
    models/norm_mean.npy           — per-channel normalisation mean
    models/norm_std.npy            — per-channel normalisation std
    results/training_curves.png    — loss and accuracy plots
    results/history.json           — epoch-level metrics
"""

import json
import os
import time
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model   import WearParkCNN1D


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "batch_size"     : 32,
    "learning_rate"  : 1e-3,
    "epochs"         : 120,
    "patience"       : 25,
    "test_size"      : 0.25,
    "val_size"       : 0.15,
    "random_state"   : 42,
    "dropout"        : 0.3,
    "fc_dropout"     : 0.5,
    "weight_decay"   : 5e-4,
    "mixup_alpha"    : 0.4,     # Beta distribution parameter for Mixup
    "noise_std"      : 0.02,    # Std of additive Gaussian noise
    "gyro_mask_prob" : 0.2,     # Probability of zeroing gyro channels per sample
    "use_monipar"    : True,
}


def get_device() -> torch.device:
    """Select the best available compute device.

    Returns:
        torch.device: ``mps`` → ``cuda`` → ``cpu``.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Mixup augmentation
# ---------------------------------------------------------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device: str = "cpu",) -> tuple:
    """Apply Mixup augmentation to a batch.

    Generates convex combinations of sample pairs using a mixing coefficient
    drawn from Beta(alpha, alpha).

    Args:
        x (torch.Tensor): Input batch of shape (batch, channels, length).
        y (torch.Tensor): Label batch of shape (batch,).
        alpha (float): Beta distribution parameter. Higher values push lambda
            closer to 0.5 (stronger mixing). Defaults to 0.4.
        device (str | torch.device): Device for the permutation index tensor.

    Returns:
        tuple:
            - x_mixed (torch.Tensor): Mixed input batch.
            - y_a     (torch.Tensor): Original labels.
            - y_b     (torch.Tensor): Labels of the shuffled partner samples.
            - lam     (float): Mixing coefficient in [0, 1].

    References:
        Zhang et al., 2018. "mixup: Beyond Empirical Risk Minimization."
        https://arxiv.org/abs/1710.09412
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion: nn.Module, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float,) -> torch.Tensor:
    """Compute the Mixup loss as a weighted combination of two BCE terms.

    Args:
        criterion (nn.Module): Base loss function (e.g. ``BCEWithLogitsLoss``).
        logits (torch.Tensor): Raw model outputs of shape (batch,).
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Shuffled partner labels.
        lam (float): Mixing coefficient from ``mixup_batch``.

    Returns:
        torch.Tensor: Scalar mixed loss.
    """
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple:
    """Run one training epoch with Mixup and Gaussian noise.

    Args:
        model (nn.Module): The CNN model in training mode.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Gradient optimiser.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.
        config (dict): Training configuration dictionary (``noise_std``,
            ``mixup_alpha`` keys are used).

    Returns:
        tuple[float, float]: ``(mean_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Additive white noise for signal augmentation
        X = X + torch.randn_like(X) * config["noise_std"]

        X_mix, y_a, y_b, lam = mixup_batch(X, y, alpha=config["mixup_alpha"],
                                             device=device)
        optimizer.zero_grad()
        logits = model(X_mix).squeeze(1)
        loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).long()
        correct += (
            lam       * (preds == y_a.long()).float() +
            (1 - lam) * (preds == y_b.long()).float()
        ).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device,) -> tuple:
    """Run one validation epoch without augmentation.

    Args:
        model (nn.Module): The CNN model (switched to eval mode internally).
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Compute device.

    Returns:
        tuple[float, float]: ``(mean_loss, accuracy)`` over the epoch.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X).squeeze(1)
        loss   = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.5).long() == y.long()).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_curves(history: dict, path: str) -> None:
    """Save loss and accuracy learning curves as a PNG image.

    Args:
        history (dict): Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc`` — each a list of per-epoch values.
        path (str): Output file path (e.g. ``results/training_curves.png``).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key_t, key_v, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",  "Accuracy"),
    ]:
        ax.plot(epochs, history[key_t], label="Train")
        ax.plot(epochs, history[key_v], label="Val")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Learning curves saved → {path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train() -> str:
    """Execute the full training loop and save the best model.

    Loads data, builds the model and optimiser, runs epochs with early stopping,
    and persists the best checkpoint along with normalisation statistics.

    Returns:
        str: Absolute path to the saved best model checkpoint (``.pt`` file).

    Raises:
        FileNotFoundError: If the PADS dataset is not found at the expected path.
    """
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_ratio, mean, std = get_dataloaders(
        batch_size     = CONFIG["batch_size"],
        test_size      = CONFIG["test_size"],
        val_size       = CONFIG["val_size"],
        random_state   = CONFIG["random_state"],
        gyro_mask_prob = CONFIG["gyro_mask_prob"],
        use_monipar    = CONFIG["use_monipar"],
    )

    np.save(os.path.join(MODELS_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODELS_DIR, "norm_std.npy"),  std)

    model = WearParkCNN1D(
        dropout    = CONFIG["dropout"],
        fc_dropout = CONFIG["fc_dropout"],
    ).to(device)
    model.count_params()

    pos_weight = torch.tensor([class_ratio], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_model_path  = os.path.join(MODELS_DIR, "wearpark_cnn1d_best.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining — {CONFIG['epochs']} epochs | patience {CONFIG['patience']}\n")
    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion,
                                     device, CONFIG)
        v_loss, v_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
              f"Train {t_loss:.4f}/{t_acc:.3f} | "
              f"Val {v_loss:.4f}/{v_acc:.3f} | "
              f"LR {lr:.6f}")

        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : v_loss,
                "val_acc"     : v_acc,
                "config"      : CONFIG,
            }, best_model_path)
            print(f"  ✓ Checkpoint saved (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nCompleted in {time.time() - start:.1f}s")
    save_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best model → {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train()
