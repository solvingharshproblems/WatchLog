# WatchLog — Federated Deep Learning Based Log Anomaly Detection System

WatchLog is a privacy-preserving deep learning based log anomaly detection system that uses LSTM Autoencoders and Federated Learning to detect anomalies in distributed system logs without sharing raw log data.

This project implements a complete end-to-end pipeline including log parsing, sequence modeling, anomaly detection, federated training, evaluation, and visualization.

---

## 📊 Project Presentation

You can view the WatchLog project presentation here:

🔗 [WatchLog PPT Presentation](https://docs.google.com/presentation/d/1jX7OODJnP09QK0rDgZv_pJZ5Cvx0rn_T/edit?usp=sharing)

---

# Overview

Modern distributed systems generate massive volumes of logs. Detecting anomalies manually or using rule-based systems is inefficient, non-scalable, and incapable of detecting unknown failures.

WatchLog solves this problem using:

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
WatchLog/
│
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── models/
│   ├── trained/
│   │   ├── hdfs_model.pth
│   │   ├── bgl_model.pth
│   │   ├── apache_model.pth
│   │   ├── hadoop_model.pth
│   │   └── openstack_model.pth
│   │
│   ├── lstm_autoencoder.py
│   ├── transformer_encoder.py        # future upgrade
│   └── trainer.py
│
├── data/
│   ├── raw_logs/
│   │   ├── HDFS/
│   │   │   ├── HDFS.log
│   │   │   └── anomaly_label.csv
│   │   │
│   │   ├── BGL/
│   │   │   ├── BGL.log
│   │   │   └── anomaly_label.csv
│   │   │
│   │   ├── Apache/
│   │   │   ├── Apache.log
│   │   │   └── anomaly_label.csv
│   │   │
│   │   ├── Hadoop/
│   │   │   ├── Hadoop.log
│   │   │   └── anomaly_label.csv
│   │   │
│   │   └── OpenStack/
│   │       ├── OpenStack.log
│   │       └── anomaly_label.csv
│   │
│   ├── parsed_logs/
│   │   ├── HDFS/
│   │   │   └── parsed.csv
│   │   ├── BGL/
│   │   ├── Apache/
│   │   ├── Hadoop/
│   │   └── OpenStack/
│   │
│   └── processed/
│       ├── HDFS/
│       │   └── sequences.npy
│       ├── BGL/
│       ├── Apache/
│       ├── Hadoop/
│       └── OpenStack/
│
├── log_parsing/
│   ├── drain_parser.py
│   ├── loghub_parser_config.py       # dataset specific regex rules
│   └── template_miner.py
│
├── feature_engineering/
│   ├── sequence_features.py
│   ├── embedding_features.py         # future transformer embeddings
│   └── sliding_window.py
│
├── thresholding/
│   ├── threshold.py
│   ├── adaptive_threshold.py         # distribution fitting method
│   └── percentile_threshold.py
│
├── visualization/
│   ├── anomaly_plots.py
│   ├── roc_curve.py
│   ├── confusion_matrix.py
│   └── training_curves.py
│
├── utils/
│   ├── config_loader.py
│   ├── data_downloader.py
│   ├── dataset_loader.py
│   ├── logger.py
│   ├── save_results.py
│   ├── experiment_logger.py
│   └── seed.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluation.py
│   └── model_validation.py
│
├── federated/
│   ├── client.py
│   ├── server.py
│   ├── aggregation.py
│   └── data_partition.py
│
├── experiments/
│   ├── HDFS/
│   │   └── experiment_*.json
│   ├── BGL/
│   ├── Apache/
│   ├── Hadoop/
│   └── OpenStack/
│
└── notebooks/
    ├── dataset_analysis.ipynb
    ├── anomaly_visualization.ipynb
    └── model_comparison.ipynb
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
