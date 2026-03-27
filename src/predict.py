"""Inference interface for WearPark backend integration.

Exposes ``WearParkPredictor``, a stateful class that loads the trained CNN 1D
model once and handles all preprocessing required to convert raw ICM-20948
sensor data into a structured prediction result.

Data flow::

    ICM-20948 @ 100 Hz
        └─► Backend stores motion_entry in MongoDB
                └─► POST /predict/binary  (api.py)
                        └─► WearParkPredictor.predict_from_binary()
                                └─► { state, label, probability, confidence }

Input specification:
    - 1000 samples at 100 Hz = exactly 10 seconds of recording.
    - 6 channels: AccX, AccY, AccZ in m/s², GyroX, GyroY, GyroZ in rad/s.
    - The ICM-20948 must be configured to output in those units
      (see WearPark-Embedded repository).

Preprocessing pipeline (applied automatically):
    1. Transpose to (6, N) if input arrives as (N, 6).
    2. Convert AccX/Y/Z from m/s² to g (÷ 9.81).
    3. Remove DC gravity offset: 4th-order Butterworth high-pass at 0.5 Hz.
    4. Z-score normalisation using training-set statistics.

Three-state classification:
    ============  ====================  ===================================
    State         Probability range     Meaning
    ============  ====================  ===================================
    ``ok``        < 0.35                No significant tremor detected
    ``monitoring``  0.35 - 0.65         Ambiguous; recommend temporal aggregation
    ``parkinson`` ≥ 0.65                Strong tremor signal consistent with PD
    ============  ====================  ===================================
"""

import json
import os
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from model import WearParkCNN1D


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL = os.path.join(MODELS_DIR, "wearpark_cnn1d_best.pt")
NORM_MEAN  = os.path.join(MODELS_DIR, "norm_mean.npy")
NORM_STD   = os.path.join(MODELS_DIR, "norm_std.npy")

N_CHANNELS  = 6     # AccX/Y/Z + GyroX/Y/Z
N_TIMESTEPS = 1000  # 10 s at 100 Hz
FS          = 100   # sampling frequency (Hz)

# Three-state decision boundaries
_THRESHOLD_LOW  = 0.35
_THRESHOLD_HIGH = 0.65


class WearParkPredictor:
    """Stateful inference class for the WearPark CNN 1D model.

    Instantiate once at application startup (e.g. FastAPI lifespan), then call
    ``predict*`` methods for each incoming sensor window. The model and
    normalisation statistics are kept in memory between calls.

    Args:
        model_path (str): Path to the ``.pt`` checkpoint file.
        mean_path (str): Path to the ``norm_mean.npy`` file.
        std_path (str): Path to the ``norm_std.npy`` file.

    Attributes:
        model (WearParkCNN1D | None): Loaded model in eval mode.
        mean (np.ndarray | None): Normalisation mean, shape (6, 1).
        std (np.ndarray | None): Normalisation std, shape (6, 1).
        threshold (float): Optimal decision threshold loaded from
            ``results/metrics.json`` after training. Defaults to 0.5.
        device (torch.device): Compute device selected at init time.

    Example:
        >>> predictor = WearParkPredictor()
        >>> predictor.load()
        >>> result = predictor.predict_from_dict({
        ...     "accel_x": [...],  # 1000 values in m/s²
        ...     "accel_y": [...],
        ...     "accel_z": [...],
        ...     "gyro_x":  [...],  # 1000 values in rad/s
        ...     "gyro_y":  [...],
        ...     "gyro_z":  [...],
        ... })
        >>> print(result["state"])   # "ok" | "monitoring" | "parkinson"
    """

    def __init__(
        self,
        model_path: str = BEST_MODEL,
        mean_path: str  = NORM_MEAN,
        std_path: str   = NORM_STD,
    ):
        self.model_path = model_path
        self.mean_path  = mean_path
        self.std_path   = std_path
        self.model      = None
        self.mean       = None
        self.std        = None
        self.threshold  = 0.5

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load(self) -> None:
        """Load the model weights, normalisation stats, and optimal threshold.

        Must be called once before any ``predict*`` method.

        Raises:
            FileNotFoundError: If the model checkpoint is not found.
                Run ``python train.py`` to generate it.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}\n"
                "Run: python train.py"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device,
                                 weights_only=True)
        config     = checkpoint.get("config", {})

        self.model = WearParkCNN1D(
            dropout    = config.get("dropout",    0.3),
            fc_dropout = config.get("fc_dropout", 0.5),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.mean = np.load(self.mean_path)
        self.std  = np.load(self.std_path)

        # Use the Youden-optimal threshold computed during evaluation
        metrics_path = os.path.join(BASE_DIR, "results", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            self.threshold = metrics.get("by_subject", {}).get("threshold_opt", 0.5)

        print(
            f"WearParkPredictor ready — device={self.device} | "
            f"threshold={self.threshold:.2f} | "
            f"epoch={checkpoint['epoch']}"
        )

    @staticmethod
    def _preprocess(signal: np.ndarray) -> np.ndarray:
        """Preprocess a raw ICM-20948 signal for model inference.

        Steps:
            1. Ensure shape is (6, N).
            2. Convert accelerometer axes from m/s² to g.
            3. Apply a 4th-order Butterworth high-pass filter (0.5 Hz) to
               remove the gravitational DC component from acc channels.

        Args:
            signal (np.ndarray): Raw IMU data of shape (6, N) or (N, 6).
                Channel order: AccX, AccY, AccZ (m/s²), GyroX, GyroY, GyroZ (rad/s).

        Returns:
            np.ndarray: Preprocessed signal of shape (6, N), accelerometer
                channels in g with gravity removed.
        """
        if signal.ndim == 2 and signal.shape[1] == N_CHANNELS:
            signal = signal.T

        signal = signal.astype(np.float32).copy()

        signal[:3] /= 9.81

        b, a = butter(4, 0.5 / (FS / 2), btype="high")
        for i in range(3):
            signal[i] = filtfilt(b, a, signal[i])

        return signal

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> dict:
        """Predict Parkinson probability from a raw IMU signal array.

        Args:
            signal (np.ndarray): Shape (6, N) or (N, 6).
                Channels: AccX/Y/Z in m/s², GyroX/Y/Z in rad/s.
                N = 1000 recommended (10 s @ 100 Hz).

        Returns:
            dict: Prediction result with the following keys:

                - ``prediction``  (int):   0 = non-Parkinson, 1 = Parkinson
                - ``probability`` (float): Parkinson probability in [0, 1]
                - ``state``       (str):   ``"ok"`` | ``"monitoring"`` | ``"parkinson"``
                - ``label``       (str):   Human-readable state label
                - ``confidence``  (str):   ``"high"`` | ``"medium"`` | ``"low"``

        Raises:
            RuntimeError: If ``load()`` has not been called.
            ValueError: If the signal has an unexpected number of channels.
        """
        if self.model is None:
            raise RuntimeError("Call predictor.load() before predict().")

        signal = self._preprocess(signal)

        if signal.shape[0] != N_CHANNELS:
            raise ValueError(
                f"Expected {N_CHANNELS} channels, got {signal.shape[0]}."
            )

        # Z-score normalisation using training-set statistics
        signal = (signal - self.mean) / (self.std + 1e-8)

        tensor = torch.tensor(signal).unsqueeze(0).to(self.device)
        prob   = torch.sigmoid(self.model(tensor)).item()

        # Three-state decision
        if prob >= _THRESHOLD_HIGH:
            state = "parkinson"
            label = "Parkinson"
            pred  = 1
        elif prob >= _THRESHOLD_LOW:
            state = "monitoring"
            label = "Monitoring"
            pred  = 0
        else:
            state = "ok"
            label = "Non-Parkinson"
            pred  = 0

        margin     = abs(prob - 0.5)
        confidence = "high" if margin > 0.3 else "medium" if margin > 0.15 else "low"

        return {
            "prediction" : pred,
            "probability": round(prob, 4),
            "state"      : state,
            "label"      : label,
            "confidence" : confidence,
        }

    def predict_from_dict(self, imu_dict: dict) -> dict:
        """Predict from a JSON-style channel dictionary.

        Convenience wrapper used by the ``/predict/arrays`` API endpoint.

        Args:
            imu_dict (dict): Keys: ``accel_x``, ``accel_y``, ``accel_z``,
                ``gyro_x``, ``gyro_y``, ``gyro_z``. Each value is a list of
                1000 floats. Accelerometer values in m/s², gyroscope in rad/s.

        Returns:
            dict: Same structure as ``predict()``.
        """
        signal = np.array([
            imu_dict["accel_x"],
            imu_dict["accel_y"],
            imu_dict["accel_z"],
            imu_dict["gyro_x"],
            imu_dict["gyro_y"],
            imu_dict["gyro_z"],
        ], dtype=np.float32)
        return self.predict(signal)

    def predict_from_binary(self, raw_bytes: bytes) -> dict:
        """Predict directly from the raw binary payload stored in MongoDB.

        Decodes the ``data.$binary.base64`` field of a ``motion_entries``
        document without any intermediate conversion step.

        Binary format: N x 7 little-endian float32 columns::

            [Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ]

        The ``Time`` column (index 0) is discarded automatically.

        Args:
            raw_bytes (bytes): Raw bytes decoded from the base64 field.

        Returns:
            dict: Same structure as ``predict()``.
        """
        vals   = np.frombuffer(raw_bytes, dtype="<f4").reshape(-1, 7)
        signal = vals[:, 1:].T
        return self.predict(signal)


# -------------------------------MAIN--------------------------------------------
if __name__ == "__main__":
    predictor = WearParkPredictor()
    predictor.load()

    # Smoke test with a synthetic random signal
    dummy_imu = {k: list(np.random.randn(N_TIMESTEPS).astype(float))
                 for k in ("accel_x", "accel_y", "accel_z",
                           "gyro_x",  "gyro_y",  "gyro_z")}

    result = predictor.predict_from_dict(dummy_imu)
    print("\nPrediction result:")
    print(json.dumps(result, indent=2))
