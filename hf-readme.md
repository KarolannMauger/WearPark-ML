---
title: WearPark ML
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# WearPark ML

Parkinson tremor detection API using a residual CNN 1D trained on wrist IMU signals (ICM-20948, 100 Hz).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness and model status |
| `POST` | `/predict/binary` | Predict from MongoDB `motion_entries` binary payload |
| `POST` | `/predict/arrays` | Predict from six explicit float-list channels |

## Input

- **1000 samples** at 100 Hz (10 seconds)
- **6 channels**: AccX/Y/Z in m/s², GyroX/Y/Z in rad/s

## Output

```json
{
  "prediction": 1,
  "probability": 0.8241,
  "state": "parkinson",
  "label": "Parkinson",
  "confidence": "high"
}
```

## States

| State | Probability | Meaning |
|-------|-------------|---------|
| `ok` | < 0.35 | No significant tremor |
| `monitoring` | 0.35 – 0.65 | Ambiguous, monitor over time |
| `parkinson` | ≥ 0.65 | Strong tremor signal consistent with PD |
