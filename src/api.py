"""FastAPI service exposing the WearPark CNN 1D model for backend consumption.

The WearPark-Backend (Java Spring Boot) calls this service after accumulating
1000 IMU samples from the ICM-20948 sensor. The model is loaded once at startup
and kept in memory for the lifetime of the process.

Endpoints:
    POST /predict/arrays  — six float lists (AccX/Y/Z, GyroX/Y/Z)
    POST /predict/binary  — raw base64 payload from a MongoDB motion_entry
    GET  /health          — liveness and model readiness check

Running the server:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Example call from Spring Boot backend:

.. code-block:: java

    // DTOs
    public record IMUBinaryRequest(
        @JsonProperty("base64_data") String base64Data,
        @JsonProperty("nb_entries")  int    nbEntries
    ) {}

    public record PredictionResult(
        int    prediction,
        double probability,
        String state,
        String label,
        String confidence
    ) {}

    // Service — Spring Boot avec RestTemplate
    @Service
    public class WearParkMLService {
        private static final String ML_API_URL = "http://localhost:8001/predict/binary";
        private final RestTemplate restTemplate;

        public WearParkMLService(RestTemplateBuilder builder) {
            this.restTemplate = builder.build();
        }

        public PredictionResult predict(MotionEntry motionEntry) {
            String base64Data = motionEntry.getData().getBinary().getBase64();

            IMUBinaryRequest requestBody = new IMUBinaryRequest(base64Data, 1000);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<IMUBinaryRequest> request = new HttpEntity<>(requestBody, headers);

            ResponseEntity<PredictionResult> response = restTemplate.exchange(
                ML_API_URL,
                HttpMethod.POST,
                request,
                PredictionResult.class
            );

            return response.getBody();
        }
    }

    // Controller usage example
    PredictionResult result = mlService.predict(motionEntry);

    String state       = result.state();        // "ok" | "monitoring" | "parkinson"
    double probability = result.probability();  // 0.0 - 1.0
    String label       = result.label();        // "Non-Parkinson" | "Monitoring" | "Parkinson"
"""

import base64
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from predict import N_CHANNELS, N_TIMESTEPS, WearParkPredictor


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class IMUArrays(BaseModel):
    """Request body for the ``/predict/arrays`` endpoint.

    Attributes:
        accel_x: Accelerometer X-axis, 1000 floats in m/s².
        accel_y: Accelerometer Y-axis, 1000 floats in m/s².
        accel_z: Accelerometer Z-axis, 1000 floats in m/s².
        gyro_x:  Gyroscope X-axis, 1000 floats in rad/s.
        gyro_y:  Gyroscope Y-axis, 1000 floats in rad/s.
        gyro_z:  Gyroscope Z-axis, 1000 floats in rad/s.
    """

    accel_x: List[float]
    accel_y: List[float]
    accel_z: List[float]
    gyro_x:  List[float]
    gyro_y:  List[float]
    gyro_z:  List[float]

    @field_validator("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
    @classmethod
    def check_length(cls, v: List[float]) -> List[float]:
        """Reject signals shorter than 100 samples (< 1 second).

        Args:
            v (List[float]): Channel values to validate.

        Returns:
            List[float]: The validated list, unchanged.

        Raises:
            ValueError: If the list contains fewer than 100 elements.
        """
        if len(v) < 100:
            raise ValueError(f"Signal too short: {len(v)} samples (minimum 100)")
        return v


class IMUBinary(BaseModel):
    """Request body for the ``/predict/binary`` endpoint.

    Attributes:
        base64_data: Contents of ``data.$binary.base64`` from a MongoDB
            ``motion_entries`` document.
        nb_entries: Expected number of timesteps. Informational; not used
            for truncation. Defaults to 1000.
    """

    base64_data: str
    nb_entries:  int = N_TIMESTEPS


class PredictionResult(BaseModel):
    """Response schema returned by all ``/predict/*`` endpoints.

    Attributes:
        prediction:  Binary classification — 0 = non-Parkinson, 1 = Parkinson.
        probability: Parkinson probability in [0.0, 1.0].
        state:       Decision state — ``"ok"``, ``"monitoring"``, or ``"parkinson"``.
        label:       Human-readable state label.
        confidence:  Model confidence — ``"high"``, ``"medium"``, or ``"low"``.
    """

    prediction:  int
    probability: float
    state:       str
    label:       str
    confidence:  str


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
predictor = WearParkPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model before accepting requests, release on shutdown."""
    predictor.load()
    yield


app = FastAPI(
    title="WearPark ML API",
    description=(
        "Binary Parkinson tremor detection from ICM-20948 wrist IMU signals.\n\n"
        "Trained on PADS (469 subjects) + Monipar datasets using a residual "
        "CNN 1D architecture. Input: 10 s @ 100 Hz (1000 samples, 6 channels)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Service health check")
def health() -> dict:
    """Return the service status and model readiness.

    Returns:
        dict: JSON object with ``status`` (``"ok"`` or ``"model_not_loaded"``),
            ``model_ready`` (bool), and ``threshold`` (float | None).
    """
    ready = predictor.model is not None
    return {
        "status"     : "ok" if ready else "model_not_loaded",
        "model_ready": ready,
        "threshold"  : predictor.threshold if ready else None,
    }


@app.post(
    "/predict/arrays",
    response_model=PredictionResult,
    summary="Predict from channel arrays",
)
def predict_arrays(body: IMUArrays) -> PredictionResult:
    """Run inference from six explicit float-list channels.

    Intended for backends that decode IMU bytes before sending, or for
    direct testing without a MongoDB document.

    Args:
        body (IMUArrays): Six channel arrays of 1000 floats each.

    Returns:
        PredictionResult: Structured prediction with state and confidence.

    Raises:
        HTTPException 422: On preprocessing or shape mismatch errors.

    Example request body::

        {
            "accel_x": [0.03, -0.01, ...],
            "accel_y": [-9.72, ...],
            "accel_z": [0.14, ...],
            "gyro_x":  [0.003, ...],
            "gyro_y":  [0.010, ...],
            "gyro_z":  [-0.002, ...]
        }
    """
    try:
        result = predictor.predict_from_dict(body.model_dump())
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post(
    "/predict/binary",
    response_model=PredictionResult,
    summary="Predict from MongoDB binary payload",
)
def predict_binary(body: IMUBinary) -> PredictionResult:
    """Run inference directly from a MongoDB motion_entry binary blob.

    No decoding required on the backend side — pass the raw base64 string
    from the ``data.$binary.base64`` field of a ``motion_entries`` document.

    Expected binary format: N x 7 little-endian float32 columns::

        [Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ]

    The ``Time`` column is discarded automatically.

    Args:
        body (IMUBinary): Base64-encoded binary data and optional entry count.

    Returns:
        PredictionResult: Structured prediction with state and confidence.

    Raises:
        HTTPException 422: On base64 decode failure or shape mismatch.

    Example request body::

        {
            "base64_data": "CAAAALAAhT+MOJvA...",
            "nb_entries": 1000
        }
    """
    try:
        raw    = base64.b64decode(body.base64_data)
        result = predictor.predict_from_binary(raw)
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
