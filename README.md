# LogWatch — Federated Deep Learning Based Log Anomaly Detection System

LogWatch is a privacy-preserving deep learning based log anomaly detection system that uses LSTM Autoencoders and Federated Learning to detect anomalies in distributed system logs without sharing raw log data.

This project implements a complete end-to-end pipeline including log parsing, sequence modeling, anomaly detection, federated training, evaluation, and visualization.

---

# Overview

Modern distributed systems generate massive volumes of logs. Detecting anomalies manually or using rule-based systems is inefficient, non-scalable, and incapable of detecting unknown failures.

LogWatch solves this problem using:

- Deep Learning (LSTM Autoencoder)
- Federated Learning (Flower Framework)
- Log Parsing using template extraction
- Sliding window sequence modeling
- Unsupervised anomaly detection
- Privacy-preserving distributed training

---

# Key Features

- Fully automated log anomaly detection pipeline
- LSTM Autoencoder based sequence reconstruction model
- Federated learning support for distributed privacy-preserving training
- Log parsing using Drain-inspired template extraction
- Sliding window sequence generation
- Reconstruction error based anomaly detection
- Multiple thresholding methods:
  - Percentile
  - Standard deviation
  - Interquartile range (IQR)
- Complete evaluation system:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC Curve
- Visualization of anomalies
- Experiment tracking and logging
- GPU support (CUDA)

---

# System Architecture

## Pipeline Flow

Raw Logs
↓
Log Parser (Drain)
↓
Structured Logs (Event IDs)
↓
Sequence Builder (Sliding Window)
↓
LSTM Autoencoder Training
↓
Reconstruction Error Computation
↓
Threshold Calculation
↓
Anomaly Detection
↓
Evaluation and Visualization

## Federated Learning Flow

```
Client Logs → Local Training → Model Weights → Server Aggregation → Global Model
```

Raw log data never leaves client machines, ensuring privacy.

---

# Machine Learning Model

## Model: LSTM Autoencoder

Purpose:
Learn normal log sequence patterns and detect anomalies using reconstruction error.

Concept:

- Normal sequence → Low reconstruction error
- Anomalous sequence → High reconstruction error

Model components:

- Encoder LSTM
- Latent vector representation
- Decoder LSTM
- Reconstruction error computation

---

# Dataset

Default dataset used:

HDFS Log Dataset

Source:
https://github.com/logpai/loghub

Contains:

- Raw system logs
- Ground truth anomaly labels

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flower (Federated Learning)
- YAML
- Requests
- TQDM

---

# Project Structure

```
LogWatch/
│
├── main.py
├── config.yaml
├── requirements.txt
├── model.pth
├── results.csv
│
├── data/
│   ├── raw_logs/
│   │   ├── HDFS.log
│   │   └── anomaly_label.csv
│   │
│   ├── parsed_logs/
│   │   └── parsed.csv
│   │
│   └── processed/
│       └── sequences.npy
│
├── log_parsing/
│   └── drain_parser.py
│
├── feature_engineering/
│   └── sequence_features.py
│
├── model/
│   ├── lstm_autoencoder.py
│   └── trainer.py
│
├── thresholding/
│   └── threshold.py
│
├── visualization/
│   ├── anomaly_plots.py
│   └── roc_curve.py
│
├── utils/
│   ├── config_loader.py
│   ├── data_downloader.py
│   ├── logger.py
│   ├── save_results.py
│   └── experiment_logger.py
│
├── federated/
│   ├── client.py
│   ├── server.py
│   └── data_partition.py
│
├── evaluation.py
│
└── experiments/
    └── experiment_*.json
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/solvingharshproblems/WatchLog.git
cd WatchLog
```

Install dependencies:

pip install -r requirements.txt

Run Complete Pipeline:

python main.py

This will automatically:
	•	Download dataset
	•	Parse logs
	•	Build sequences
	•	Train LSTM model
	•	Detect anomalies
	•	Evaluate performance
	•	Generate plots
	•	Save experiment results

Output Files Generated:

model.pth
results.csv
experiments/
plots/

Evaluation Metrics:

The system evaluates performance using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix
	•	ROC Curve

Example output:

Accuracy  : 0.96
Precision : 0.94
Recall    : 0.95
F1 Score  : 0.945

Research Contributions

This project demonstrates:
	•	Deep sequence modeling for anomaly detection
	•	Federated learning for privacy-preserving ML
	•	Real-world distributed log anomaly detection pipeline
	•	End-to-end ML system engineering
	•	Production-style ML pipeline implementation

Applications
	•	Cloud infrastructure monitoring
	•	Cybersecurity intrusion detection
	•	Distributed system monitoring
	•	Data center fault detection
	•	DevOps automation

Future Improvements
	•	Transformer-based anomaly detection
	•	Real-time streaming support
	•	Online learning capability
	•	Explainable anomaly detection
	•	Web dashboard visualization
