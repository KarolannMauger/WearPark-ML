# WearPark ML

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-red?style=flat)
![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?style=flat)

> Binary Parkinson tremor detection from wrist IMU signals using a residual 1D CNN.
> Part of the **WearPark** research project.

---

## Overview

WearPark ML classifies wrist IMU signals (accelerometer + gyroscope) into two categories:

| Label | Meaning |
|---|---|
| **Parkinson** | Signal pattern consistent with Parkinsonian tremor |
| **Non-Parkinson** | Healthy or other movement disorder |

The model processes **1000 samples at 100 Hz** (~10 seconds) from an ICM-20948 sensor and returns a three-state output used by the WearPark backend to trigger user notifications.

---

## System Architecture

```
ICM-20948 (100 Hz)
      │
      ▼
 Embedded (CircuitPython)
      │  IMU stream
      ▼
  Backend (Node.js)
      │  POST /predict/binary
      ▼
  WearPark ML (FastAPI)
      │  { state, probability, label }
      ▼
  Backend → Push Notification
      │
      ▼
   Mobile App
```

### Three-state output

| State | Probability | User message |
|---|---|---|
| `ok` | < 0.35 | No alert |
| `monitoring` | 0.35 – 0.65 | "Unusual movement detected — consult your physician" |
| `parkinson` | > 0.65 | "Tremor pattern detected — consult your physician" |

---

## Model

**Architecture:** Residual CNN 1D
**Parameters:** ~170K
**Input:** `(batch, 6, 1000)` — 6 channels × 1000 time steps
**Output:** `(batch, 1)` — raw logit → sigmoid probability

```
Input (6, 1000)
   │
   ├─ ResBlock 1 — Conv1D(6→32,  k=11) + skip
   ├─ ResBlock 2 — Conv1D(32→64, k=7)  + skip
   └─ ResBlock 3 — Conv1D(64→128, k=5) + skip
   │
   AdaptiveAvgPool1d(1)
   │
   Linear(128→64) → Dropout(0.5) → Linear(64→1)
```

**Training dataset:**
- [PADS](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) — 469 subjects (276 PD · 79 HC · 114 DD)
- [Monipar](https://github.com/Monipar) — 28 subjects

**Labels:** PD = 1 · (HC + DD) = 0
**Split:** 75% train / 25% test — stratified by subject

---

## Repository Structure

```
WearPark-ML/
├── src/
│   ├── dataset.py            # PADS + Monipar data loaders
│   ├── model.py              # Residual CNN 1D architecture
│   ├── train.py              # Training loop (Mixup, early stopping)
│   ├── evaluate.py           # Metrics, ROC, confusion matrix
│   ├── predict.py            # WearParkPredictor inference class
│   ├── preprocess_monipar.py # Monipar dataset preprocessing
│   └── api.py                # FastAPI service
├── docs/
│   └── generate.py           # Local doc generation (pdoc)
├── .github/
│   └── workflows/
│       └── docs.yml          # Auto-deploy docs to GitHub Pages
├── models/                   # Saved checkpoints (not tracked)
├── results/                  # Metrics + plots (not tracked)
├── requirements.txt
└── LICENSE
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/KarolannMauger/WearPark-ML.git
cd WearPark-ML
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the PADS dataset under:
```
datasets/physionet.org/files/parkinsons-disease-smartwatch/1.0.0/
```

### 3. Train

```bash
cd src
python train.py
# Saves best checkpoint to ../models/best_model.pth
# Saves normalisation stats to ../models/norm_stats.npz
```

### 4. Evaluate

```bash
python evaluate.py
# Outputs ../results/metrics.json, roc_curve.png, confusion_matrix.png
```

### 5. Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

---

## API

### `POST /predict/binary`
Accepts a raw base64 payload from a MongoDB `motion_entry` document.

```json
{
  "base64_data": "<base64 string>",
  "nb_entries": 1000
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.82,
  "state": "parkinson",
  "label": "Parkinson",
  "confidence": "high"
}
```

### `GET /health`
Returns model readiness status.

---

## Related Repositories

| Repository | Description |
|---|---|
| [WearPark-Embedded](https://github.com/KarolannMauger/WearPark-Embedded) | CircuitPython firmware — ICM-20948 data acquisition |
| [WearPark-Backend](https://github.com/KarolannMauger/WearPark-Backend) | Node.js API — data storage and ML orchestration |
| [WearPark-App](https://github.com/KarolannMauger/WearPark-App) | Mobile app — user notifications and history |

---

## References

- [PADS Dataset](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) — Varghese et al., *npj Parkinson's Disease*, 2024
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [Mixup Augmentation](https://arxiv.org/abs/1710.09412) — Zhang et al., 2018
- [CNN for Parkinson from IMU](https://arxiv.org/abs/1808.02870) — Um et al., 2018

---

## License

Copyright © 2026 Karolann Mauger. All rights reserved.
This project is released under a [Research Source License](./LICENSE) — view only, no commercial use, no redistribution without written permission.
See [LICENSE](./LICENSE) for full terms.

> **Medical disclaimer:** This software is a research prototype and is NOT a certified medical device. It must not be used as a substitute for professional medical diagnosis or treatment.
