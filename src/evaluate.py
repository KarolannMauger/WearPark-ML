"""Evaluation module for the WearPark CNN 1D model.

Computes metrics at two granularities:

Window-level:
    One prediction per (subject, task, wrist) combination — 12 predictions
    per PADS subject. Reflects raw model performance on individual 10-second
    windows.

Subject-level (clinical):
    Probabilities across a subject's 12 windows are averaged into a single
    score, then thresholded. This mirrors the intended clinical use case:
    one decision per patient session.

Outputs written to ``results/``:
    - ``roc_windows.png``     — ROC curve at window level
    - ``roc_subjects.png``    — ROC curve at subject level
    - ``confusion_subjects.png`` — confusion matrix at subject level
    - ``metrics.json``        — numeric metrics (used by ``predict.py`` to
                                load the optimal classification threshold)

Usage:
    python evaluate.py
"""

import json
import os
import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset import get_dataloaders, N_WINDOWS
from model   import WearParkCNN1D


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BEST_MODEL  = os.path.join(MODELS_DIR, "wearpark_cnn1d_best.pt")


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


@torch.no_grad()
def get_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,) -> tuple:
    """Collect predicted probabilities and ground-truth labels from a loader.

    Args:
        model (nn.Module): Trained model in eval mode.
        loader (DataLoader): Data loader to iterate over.
        device (torch.device): Device to run inference on.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - probs:  float array of shape (N,), values in [0, 1].
            - labels: float array of shape (N,), values in {0, 1}.
    """
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        probs = torch.sigmoid(model(X.to(device))).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def aggregate_by_subject(y_prob_win: np.ndarray, y_true_win: np.ndarray, n_windows: int = N_WINDOWS,) -> tuple:
    """Aggregate window-level predictions to subject-level by mean pooling.

    Assumes windows are ordered contiguously per subject (as produced by
    ``PADSDataset`` with a fixed subject list).

    Args:
        y_prob_win (np.ndarray): Window probabilities of shape (N,).
        y_true_win (np.ndarray): Window labels of shape (N,).
        n_windows (int): Number of windows per subject. Defaults to
            ``N_WINDOWS`` (12 for PADS).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - subj_probs:  mean probability per subject, shape (S,).
            - subj_labels: ground-truth label per subject, shape (S,).
    """
    n_subjects = len(y_prob_win) // n_windows
    subj_probs, subj_labels = [], []
    for i in range(n_subjects):
        s = i * n_windows
        e = s + n_windows
        subj_probs.append(y_prob_win[s:e].mean())
        subj_labels.append(int(y_true_win[s]))
    return np.array(subj_probs), np.array(subj_labels)


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,) -> tuple:
    """Find the classification threshold that maximises the Youden J statistic.

    Youden J = TPR - FPR. This maximises the gap between sensitivity and the
    false-positive rate, balancing both error types without class-weight bias.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_prob (np.ndarray): Predicted probabilities in [0, 1].

    Returns:
        tuple[float, float, float]:
            - threshold: optimal decision boundary.
            - fpr:       false-positive rate at that threshold.
            - tpr:       true-positive rate (sensitivity) at that threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def save_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    thr: float,
    fpr_opt: float,
    tpr_opt: float,
    title: str,
    path: str,
) -> None:
    """Plot and save a ROC curve with the optimal operating point marked.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_prob (np.ndarray): Predicted probabilities.
        auc (float): Area under the ROC curve.
        thr (float): Optimal threshold (shown in legend).
        fpr_opt (float): FPR at the optimal threshold.
        tpr_opt (float): TPR at the optimal threshold.
        title (str): Plot title.
        path (str): Output PNG file path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.scatter([fpr_opt], [tpr_opt], color="red", zorder=5,
                label=f"Optimal threshold = {thr:.2f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_cm(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: str,) -> None:
    """Plot and save a confusion matrix.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_pred (np.ndarray): Binary predicted labels.
        title (str): Plot title.
        path (str): Output PNG file path.
    """
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-PD", "Parkinson"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def evaluate() -> dict:
    """Load the best checkpoint and evaluate on the held-out test set.

    Computes window-level and subject-level metrics, saves plots, and writes
    ``results/metrics.json`` (including the optimal threshold consumed by
    ``WearParkPredictor``).

    Returns:
        dict: Nested metrics dictionary::

            {
                "by_subject": {
                    "auc_roc": float,
                    "threshold_opt": float,
                    "accuracy": float,
                    "precision": float,
                    "recall": float,
                    "f1": float,
                }
            }

    Raises:
        FileNotFoundError: If no trained model checkpoint is found.
    """
    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(BEST_MODEL):
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL}.\n"
            "Run: python train.py"
        )

    checkpoint = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    config     = checkpoint.get("config", {})
    print(f"Checkpoint loaded — epoch {checkpoint['epoch']} "
          f"| val_loss {checkpoint['val_loss']:.4f} "
          f"| val_acc {checkpoint['val_acc']:.3f}")

    _, _, test_loader, _, _, _ = get_dataloaders(
        batch_size   = config.get("batch_size",   32),
        test_size    = config.get("test_size",   0.25),
        val_size     = config.get("val_size",    0.15),
        random_state = config.get("random_state",  42),
    )

    model = WearParkCNN1D(
        dropout    = config.get("dropout",    0.3),
        fc_dropout = config.get("fc_dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    y_prob_win,  y_true_win  = get_predictions(model, test_loader, device)
    y_prob_subj, y_true_subj = aggregate_by_subject(y_prob_win, y_true_win)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (25% held-out)")
    print("=" * 60)

    for scope, y_prob, y_true in [
        ("WINDOW-LEVEL  (12 windows / subject)", y_prob_win,  y_true_win),
        ("SUBJECT-LEVEL (mean-pooled)",           y_prob_subj, y_true_subj),
    ]:
        thr, _, _ = optimal_threshold(y_true, y_prob)
        y_pred    = (y_prob >= thr).astype(int)
        auc       = roc_auc_score(y_true, y_prob)
        print(f"\n--- {scope} ---")
        print(f"  AUC-ROC   : {auc:.3f}")
        print(f"  Threshold : {thr:.2f}  (Youden J optimum)")
        print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.3f}")
        print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"  F1-score  : {f1_score(y_true, y_pred, zero_division=0):.3f}")

    thr_s, fpr_s, tpr_s = optimal_threshold(y_true_subj, y_prob_subj)
    y_pred_s = (y_prob_subj >= thr_s).astype(int)
    auc_s    = roc_auc_score(y_true_subj, y_prob_subj)

    print("\n--- Full classification report (subject-level) ---")
    print(classification_report(y_true_subj, y_pred_s,
                                  target_names=["Non-PD", "Parkinson"]))

    # Save plots
    save_roc(
        y_true_win, y_prob_win,
        roc_auc_score(y_true_win, y_prob_win),
        *optimal_threshold(y_true_win, y_prob_win),
        "ROC Curve — Window Level",
        os.path.join(RESULTS_DIR, "roc_windows.png"),
    )
    save_roc(
        y_true_subj, y_prob_subj, auc_s, thr_s, fpr_s, tpr_s,
        "ROC Curve — Subject Level",
        os.path.join(RESULTS_DIR, "roc_subjects.png"),
    )
    save_cm(
        y_true_subj, y_pred_s,
        "Confusion Matrix — Subject Level",
        os.path.join(RESULTS_DIR, "confusion_subjects.png"),
    )

    metrics_out = {
        "by_subject": {
            "auc_roc"      : float(auc_s),
            "threshold_opt": float(thr_s),
            "accuracy"     : float(accuracy_score(y_true_subj, y_pred_s)),
            "precision"    : float(precision_score(y_true_subj, y_pred_s, zero_division=0)),
            "recall"       : float(recall_score(y_true_subj, y_pred_s, zero_division=0)),
            "f1"           : float(f1_score(y_true_subj, y_pred_s, zero_division=0)),
        }
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nPlots and metrics saved → {RESULTS_DIR}/")
    return metrics_out


if __name__ == "__main__":
    evaluate()
