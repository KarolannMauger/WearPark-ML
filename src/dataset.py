"""Dataset loaders for WearPark ML training pipeline.

Supports two data sources that are combined for training:

PADS (primary):
    469 subjects (276 PD, 79 HC, 114 DD→non-PD), Apple Watch Series 4,
    6-channel IMU (acc + gyro), 976 samples @ 100 Hz per task window.
    Reference: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

Monipar (secondary):
    28 subjects, accelerometer only (gyro channels zeroed),
    resampled from 50 Hz → 100 Hz to match PADS format.
    Reference: https://zenodo.org/records/8104853

Both sources are normalised to shape (6, 1000) for compatibility with the
ICM-20948 sensor used in WearPark hardware (10 s @ 100 Hz).

Split strategy:
    PADS    — 75% train / 25% test stratified *by subject* (no window leakage).
    Monipar — pre-split in ``preprocess_monipar.py`` (75/25 by window).

Augmentation (training only):
    - Gyro channel masking: zeroes channels 3-5 with probability ``gyro_mask_prob``
      to make the model robust when gyroscope data is absent or noisy.
    - Gaussian noise and Mixup are applied in ``train.py``.
"""

import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(
    BASE_DIR, "datasets", "physionet.org", "files",
    "parkinsons-disease-smartwatch", "1.0.0",
)
MOVEMENT_DIR = os.path.join(DATA_ROOT, "preprocessed", "movement")
FILE_LIST    = os.path.join(DATA_ROOT, "preprocessed", "file_list.csv")
MONIPAR_DIR  = os.path.join(BASE_DIR, "datasets", "monipar", "processed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CHANNELS_TOTAL  = 132   # channels in a PADS binary file (11 tasks × 2 wrists × 6 sensors)
N_SENSOR_CH       = 6     # AccX/Y/Z + GyroX/Y/Z
N_WRIST_CH        = 12    # 2 × N_SENSOR_CH (left + right wrist interleaved)
N_TIMESTEPS_PADS  = 976   # native PADS timesteps per window
N_TIMESTEPS       = 1000  # target length — matches ICM-20948 @ 100 Hz (10 s)

# Tasks selected for training (indices into the 11-task PADS layout)
# Chosen for tremor relevance: resting (Relaxed) and postural (RelaxedTask, StretchHold, HoldWeight)
SELECTED_TASK_INDICES = [0, 1, 2, 3, 4, 5]
SELECTED_TASK_NAMES   = [
    "Relaxed1", "Relaxed2",
    "RelaxedTask1", "RelaxedTask2",
    "StretchHold", "HoldWeight",
]
N_WINDOWS = len(SELECTED_TASK_INDICES) * 2


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class PADSDataset(Dataset):
    """PyTorch dataset wrapping preprocessed PADS binary files.

    Each subject's ``.bin`` file contains 132 channels × 976 timesteps.
    This dataset extracts one window per (task, wrist) combination and
    resamples it to 1000 timesteps.

    Args:
        subject_ids (list[str]): Zero-padded subject IDs, e.g. ``["001", "042"]``.
        labels (list[int]): Binary labels aligned with ``subject_ids``
            (1 = Parkinson, 0 = non-Parkinson).
        movement_dir (str): Directory containing ``XXX_ml.bin`` files.
        mean (np.ndarray | None): Per-channel mean of shape (6, 1) for
            z-score normalisation. ``None`` skips normalisation.
        std (np.ndarray | None): Per-channel std of shape (6, 1).
        gyro_mask_prob (float): Probability of zeroing gyroscope channels
            (indices 3-5) during training. Defaults to 0.0.

    Note:
        Each subject yields ``N_WINDOWS`` samples (6 tasks x 2 wrists = 12).
    """

    def __init__(
        self,
        subject_ids: list,
        labels: list,
        movement_dir: str = MOVEMENT_DIR,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        gyro_mask_prob: float = 0.0,
    ):
        self.movement_dir   = movement_dir
        self.mean           = mean
        self.std            = std
        self.gyro_mask_prob = gyro_mask_prob

        self.samples = [
            (sid, task_idx, wrist_offset, int(lbl))
            for sid, lbl in zip(subject_ids, labels)
            for task_idx in SELECTED_TASK_INDICES
            for wrist_offset in [0, 1]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return a single normalised window and its label.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - signal: float32 tensor of shape (6, 1000).
                - label:  float32 scalar (0.0 or 1.0).
        """
        sid, task_idx, wrist_offset, label = self.samples[idx]

        path     = os.path.join(self.movement_dir, f"{sid}_ml.bin")
        data     = np.fromfile(path, dtype=np.float32).reshape(N_CHANNELS_TOTAL, N_TIMESTEPS_PADS)
        ch_start = task_idx * N_WRIST_CH + wrist_offset * N_SENSOR_CH
        signal   = data[ch_start : ch_start + N_SENSOR_CH]

        # Resample 976 → 1000 to match ICM-20948 window length
        signal = resample(signal, N_TIMESTEPS, axis=1).astype(np.float32)

        if self.mean is not None:
            signal = (signal - self.mean) / (self.std + 1e-8)

        if self.gyro_mask_prob > 0 and np.random.rand() < self.gyro_mask_prob:
            signal[3:] = 0.0

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(label,  dtype=torch.float32),
        )


class MoniparDataset(Dataset):
    """PyTorch dataset wrapping preprocessed Monipar windows.

    Windows are pre-loaded as NumPy arrays (shape: N x 6 x 1000).
    Gyroscope channels (3-5) are already zeroed since the Monipar
    device only recorded accelerometer data.

    Args:
        windows (np.ndarray): Array of shape (N, 6, 1000), dtype float32.
        labels (np.ndarray): Binary label array of shape (N,).
        mean (np.ndarray | None): Per-channel mean (6, 1) for normalisation.
        std (np.ndarray | None): Per-channel std (6, 1) for normalisation.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None,):
        self.windows = windows
        self.labels  = labels
        self.mean    = mean
        self.std     = std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        """Return a single normalised Monipar window.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - signal: float32 tensor of shape (6, 1000).
                - label:  float32 scalar.
        """
        signal = self.windows[idx].copy()

        if self.mean is not None:
            signal = (signal - self.mean) / (self.std + 1e-8)

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def load_splits(test_size: float = 0.25, val_size: float = 0.15, random_state: int = 42,) -> tuple:
    """Load PADS labels and produce stratified subject-level splits.

    The split is performed on subjects, not windows, preventing data leakage
    between train and test sets.

    Args:
        test_size (float): Fraction of subjects held out for the test set.
            Defaults to 0.25.
        val_size (float): Fraction of the *total* dataset used for validation.
            Defaults to 0.15.
        random_state (int): Reproducibility seed. Defaults to 42.

    Returns:
        tuple: ``(X_train, X_val, X_test, y_train, y_val, y_test, class_ratio)``
            where ``class_ratio = n_neg / n_pos`` (used as ``pos_weight`` in
            ``BCEWithLogitsLoss``).
    """
    df = pd.read_csv(FILE_LIST)
    df["subject_id"]   = df["id"].apply(lambda x: f"{int(x):03d}")
    df["binary_label"] = (df["label"] == 1).astype(int)

    ids    = df["subject_id"].tolist()
    labels = df["binary_label"].tolist()

    X_tv, X_test, y_tv, y_test = train_test_split(
        ids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    adj_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=adj_val,
        stratify=y_tv,
        random_state=random_state,
    )

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos

    print("=" * 62)
    print(f"PADS: {len(ids)} subjects  |  75% train / 25% test split")
    print(f"  Train : {len(X_train)} subjects → {len(X_train) * N_WINDOWS} windows  "
          f"(PD={n_pos}, non-PD={n_neg})")
    print(f"  Val   : {len(X_val)} subjects → {len(X_val) * N_WINDOWS} windows")
    print(f"  Test  : {len(X_test)} subjects → {len(X_test) * N_WINDOWS} windows")
    print("=" * 62)

    return X_train, X_val, X_test, y_train, y_val, y_test, n_neg / n_pos


def compute_normalization_stats(subject_ids: list, movement_dir: str = MOVEMENT_DIR,) -> tuple:
    """Compute per-channel mean and std from training subjects only.

    Statistics are computed over all selected task windows of all provided
    subjects. Must be called on the training split exclusively to avoid
    data leakage into validation/test sets.

    Args:
        subject_ids (list[str]): Training subject IDs.
        movement_dir (str): Path to the ``.bin`` files directory.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - mean: shape (6, 1), dtype float32.
            - std:  shape (6, 1), dtype float32.
    """
    windows = []
    for sid in subject_ids:
        path = os.path.join(movement_dir, f"{sid}_ml.bin")
        data = np.fromfile(path, dtype=np.float32).reshape(N_CHANNELS_TOTAL, N_TIMESTEPS_PADS)
        for task_idx in SELECTED_TASK_INDICES:
            for wrist_offset in [0, 1]:
                ch_start = task_idx * N_WRIST_CH + wrist_offset * N_SENSOR_CH
                windows.append(data[ch_start : ch_start + N_SENSOR_CH])

    stacked = np.stack(windows, axis=0)
    mean = stacked.mean(axis=(0, 2)).reshape(N_SENSOR_CH, 1).astype(np.float32)
    std  = stacked.std(axis=(0, 2)).reshape(N_SENSOR_CH, 1).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def get_dataloaders(
    batch_size: int = 32,
    test_size: float = 0.25,
    val_size: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
    gyro_mask_prob: float = 0.2,
    use_monipar: bool = True,
) -> tuple:
    """Build and return train, validation, and test DataLoaders.

    Handles PADS splitting, normalisation stat computation, optional Monipar
    concatenation, and ``pos_weight`` recalculation for class imbalance.

    Args:
        batch_size (int): Mini-batch size for all loaders. Defaults to 32.
        test_size (float): Subject fraction reserved for testing. Defaults to 0.25.
        val_size (float): Fraction of total data used for validation. Defaults to 0.15.
        random_state (int): Random seed for reproducible splits. Defaults to 42.
        num_workers (int): DataLoader worker processes. Defaults to 0 (main process).
        gyro_mask_prob (float): Probability of masking gyro channels during
            training (robustness to accelerometer-only devices). Defaults to 0.2.
        use_monipar (bool): Append Monipar windows to the combined train and test
            sets when the processed files are present. Defaults to True.

    Returns:
        tuple:
            - train_loader (DataLoader)
            - val_loader   (DataLoader)
            - test_loader  (DataLoader)
            - class_ratio  (float) — ``n_neg / n_pos``, use as ``pos_weight``
            - mean         (np.ndarray) — normalisation mean (6, 1)
            - std          (np.ndarray) — normalisation std  (6, 1)
    """
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_ratio) = load_splits(test_size, val_size, random_state)

    print("Computing normalisation stats (train PADS only)...")
    mean, std = compute_normalization_stats(X_train)

    train_pads = PADSDataset(X_train, y_train, mean=mean, std=std,
                              gyro_mask_prob=gyro_mask_prob)
    val_pads   = PADSDataset(X_val,   y_val,   mean=mean, std=std)
    test_pads  = PADSDataset(X_test,  y_test,  mean=mean, std=std)

    train_ds = train_pads
    val_ds   = val_pads
    test_ds  = test_pads

    monipar_available = (
        use_monipar
        and os.path.exists(os.path.join(MONIPAR_DIR, "windows_train.npy"))
    )

    if monipar_available:
        m_wins_tr = np.load(os.path.join(MONIPAR_DIR, "windows_train.npy"))
        m_lbls_tr = np.load(os.path.join(MONIPAR_DIR, "labels_train.npy"))
        m_wins_te = np.load(os.path.join(MONIPAR_DIR, "windows_test.npy"))
        m_lbls_te = np.load(os.path.join(MONIPAR_DIR, "labels_test.npy"))

        train_monipar = MoniparDataset(m_wins_tr, m_lbls_tr, mean=mean, std=std)
        test_monipar  = MoniparDataset(m_wins_te, m_lbls_te, mean=mean, std=std)

        train_ds = ConcatDataset([train_pads, train_monipar])
        test_ds  = ConcatDataset([test_pads,  test_monipar])

        pd_monipar = int(np.sum(m_lbls_tr == 1))
        hc_monipar = int(np.sum(m_lbls_tr == 0))
        print(f"Monipar train : {len(m_wins_tr)} windows (PD={pd_monipar}, HC={hc_monipar})")
        print(f"Monipar test  : {len(m_wins_te)} windows")
        print(f"Combined train: {len(train_ds)} windows")
        print(f"Combined test : {len(test_ds)} windows")

        all_train_labels = (
            [int(l) for l in y_train] * N_WINDOWS
            + [int(l) for l in m_lbls_tr]
        )
        n_pos_total = sum(all_train_labels)
        n_neg_total = len(all_train_labels) - n_pos_total
        class_ratio = n_neg_total / n_pos_total
        print(f"Combined pos_weight: {class_ratio:.3f}")
    else:
        print("Monipar not available — using PADS only")

    print("=" * 62)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_ratio,
        mean,
        std,
    )
