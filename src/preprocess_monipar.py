"""Monipar dataset preprocessor.

Converts raw Monipar ``.mat`` files into windowed NumPy arrays compatible
with the WearPark CNN 1D input format (6 channels x 1000 samples).

Source dataset:
    Three ``.mat`` cell arrays, each containing multi-week recordings
    for a cohort of subjects:

    ==============================  ======  ===================
    File                            Label   Description
    ==============================  ======  ===================
    MONIPAR_PD_SUPERVISED.mat       PD = 1  Supervised PD sessions
    MONIPAR_PD_REMOTE.mat           PD = 1  Remote PD home recordings
    MONIPAR_HEALTHYCONTROL.mat      HC = 0  Healthy controls
    ==============================  ======  ===================

Preprocessing pipeline (per window):
    1. Extract exercises 1 (resting tremor) and 2 (postural tremor).
    2. Segment into consecutive 500-sample windows (10 s @ 50 Hz).
    3. Upsample 50 Hz → 100 Hz via ``scipy.signal.resample`` → 1000 samples.
    4. Convert accelerometer from m/s² to g (÷ 9.81).
    5. Apply 4th-order Butterworth high-pass filter at 0.5 Hz to remove gravity.
    6. Zero-pad gyroscope channels (indices 3-5) — Monipar has no gyroscope.

Output files written to ``datasets/monipar/processed/``:
    - ``windows_train.npy``  shape (N_train, 6, 1000), dtype float32
    - ``labels_train.npy``   shape (N_train,),          dtype float32
    - ``windows_test.npy``   shape (N_test,  6, 1000), dtype float32
    - ``labels_test.npy``    shape (N_test,),           dtype float32

Usage:
    python preprocess_monipar.py

Note:
    Place the three ``.mat`` files in ``datasets/monipar/`` before running.
    The split is window-level (not subject-level) due to Monipar's
    smaller cohort size.
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(BASE, "datasets", "monipar")
OUT_DIR = os.path.join(IN_DIR, "processed")

FS_IN      = 50    # Hz — Monipar native sampling rate
FS_OUT     = 100   # Hz — target rate (matches ICM-20948 / PADS)
WIN_IN     = 500   # samples @ 50 Hz = 10 s
WIN_OUT    = 1000  # samples @ 100 Hz
GRAVITY    = 9.81  # m/s² per g
EXERCISES  = {1, 2}  # exercise codes for resting and postural tremor
TEST_RATIO = 0.25


def _highpass(signal3: np.ndarray, fs: int = FS_OUT, cutoff: float = 0.5) -> np.ndarray:
    """Apply a high-pass Butterworth filter to remove gravitational DC offset.

    Args:
        signal3 (np.ndarray): Accelerometer signal of shape (3, N).
        fs (int): Sampling frequency in Hz. Defaults to ``FS_OUT`` (100).
        cutoff (float): Cut-off frequency in Hz. Defaults to 0.5.

    Returns:
        np.ndarray: Filtered signal of shape (3, N).
    """
    b, a = butter(4, cutoff / (fs / 2), btype="high")
    out  = signal3.copy()
    for i in range(3):
        out[i] = filtfilt(b, a, out[i])
    return out


def _process_segment(acc3: np.ndarray) -> list:
    """Slice a continuous accelerometer segment into 1000-sample windows.

    Each 500-sample chunk (10 s @ 50 Hz) is upsampled to 1000 samples,
    converted to g, high-pass filtered, and zero-padded with gyro channels.

    Args:
        acc3 (np.ndarray): Accelerometer data of shape (3, N) in m/s².

    Returns:
        list[np.ndarray]: List of windows, each of shape (6, 1000), float32.
    """
    wins     = []
    n_windows = acc3.shape[1] // WIN_IN
    for k in range(n_windows):
        chunk = acc3[:, k * WIN_IN : (k + 1) * WIN_IN]  # (3, 500)
        up    = resample(chunk, WIN_OUT, axis=1)          # (3, 1000) at 100 Hz
        up   /= GRAVITY                                    # m/s² -> g
        up    = _highpass(up)
        win6  = np.zeros((6, WIN_OUT), dtype=np.float32)
        win6[:3] = up.astype(np.float32)
        wins.append(win6)
    return wins


def _extract_windows(mat_array: np.ndarray, label: int) -> tuple:
    """Iterate over a Monipar cell array and extract all valid windows.

    Each cell contains one subject's recording for one week.
    Only segments labelled as exercise 1 or 2 are processed.

    Args:
        mat_array (np.ndarray): Cell array loaded from a ``.mat`` file,
            shape (subjects, weeks).
        label (int): Class label for all extracted windows (0 = HC, 1 = PD).

    Returns:
        tuple[list[np.ndarray], list[int]]:
            - windows: list of (6, 1000) float32 arrays.
            - labels:  list of ``label`` repeated ``len(windows)`` times.
    """
    windows = []
    for r in range(mat_array.shape[0]):
        for c in range(mat_array.shape[1]):
            cell = mat_array[r, c]
            if cell.size == 0:
                continue

            acc = cell[:, 1:4].T  # (3, N) — AccX/Y/Z in m/s²
            exl = cell[:, 4]       # exercise label per sample

            in_seg    = False
            seg_start = 0
            for i in range(len(exl)):
                if exl[i] in EXERCISES and not in_seg:
                    in_seg    = True
                    seg_start = i
                elif exl[i] not in EXERCISES and in_seg:
                    in_seg  = False
                    windows.extend(_process_segment(acc[:, seg_start:i]))
            if in_seg:
                windows.extend(_process_segment(acc[:, seg_start:]))

    return windows, [label] * len(windows)


def preprocess_and_save(test_ratio: float = TEST_RATIO) -> tuple:
    """Load all Monipar ``.mat`` files, preprocess, and save train/test splits.

    Args:
        test_ratio (float): Fraction of windows reserved for the test set.
            Defaults to 0.25.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ``(windows_train, labels_train, windows_test, labels_test)``

    Raises:
        FileNotFoundError: If a required ``.mat`` file is missing from
            ``datasets/monipar/``.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    pd_s = sio.loadmat(os.path.join(IN_DIR, "MONIPAR_PD_SUPERVISED.mat"))
    pd_r = sio.loadmat(os.path.join(IN_DIR, "MONIPAR_PD_REMOTE.mat"))
    hc_f = sio.loadmat(os.path.join(IN_DIR, "MONIPAR_HEALTHYCONTROL.mat"))

    pd_arr_s = pd_s[[k for k in pd_s if not k.startswith("_")][0]]
    pd_arr_r = pd_r[[k for k in pd_r if not k.startswith("_")][0]]
    hc_arr   = hc_f[[k for k in hc_f if not k.startswith("_")][0]]

    print("Extracting PD Supervised...")
    wins_pds, lbls_pds = _extract_windows(pd_arr_s, label=1)
    print(f"  {len(wins_pds)} windows")

    print("Extracting PD Remote...")
    wins_pdr, lbls_pdr = _extract_windows(pd_arr_r, label=1)
    print(f"  {len(wins_pdr)} windows")

    print("Extracting Healthy Controls...")
    wins_hc, lbls_hc = _extract_windows(hc_arr, label=0)
    print(f"  {len(wins_hc)} windows")

    all_wins = np.array(wins_pds + wins_pdr + wins_hc, dtype=np.float32)
    all_lbls = np.array(lbls_pds + lbls_pdr + lbls_hc, dtype=np.float32)

    rng = np.random.default_rng(42)
    idx      = rng.permutation(len(all_wins))
    all_wins = all_wins[idx]
    all_lbls = all_lbls[idx]

    split       = int(len(all_wins) * (1 - test_ratio))
    wins_train  = all_wins[:split]
    wins_test   = all_wins[split:]
    lbls_train  = all_lbls[:split]
    lbls_test   = all_lbls[split:]

    np.save(os.path.join(OUT_DIR, "windows_train.npy"), wins_train)
    np.save(os.path.join(OUT_DIR, "labels_train.npy"),  lbls_train)
    np.save(os.path.join(OUT_DIR, "windows_test.npy"),  wins_test)
    np.save(os.path.join(OUT_DIR, "labels_test.npy"),   lbls_test)

    pd_count = int(np.sum(all_lbls == 1))
    hc_count = int(np.sum(all_lbls == 0))
    print(f"\nMonipar preprocessing complete:")
    print(f"  Total : {len(all_wins)} windows  (PD={pd_count}, HC={hc_count})")
    print(f"  Train : {len(wins_train)} | Test: {len(wins_test)}")
    print(f"  Saved to {OUT_DIR}")
    return wins_train, lbls_train, wins_test, lbls_test


if __name__ == "__main__":
    preprocess_and_save()
