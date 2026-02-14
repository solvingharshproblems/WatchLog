import os
import random
import numpy as np
import torch

from utils.config_loader import load_config
from utils.data_downloader import download_dataset
from utils.logger import get_logger
from utils.save_results import save_results
from utils.experiment_logger import save_experiment
from log_parsing.drain_parser import parse_logs
from feature_engineering.sequence_features import build_sequences
from model.trainer import train_model
from model.lstm_autoencoder import LSTMAutoencoder
from thresholding.threshold import compute_threshold, detect
from visualization.anomaly_plots import plot_anomalies
from visualization.roc_curve import plot_roc
from evaluation import evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger()

def run():
    logger.info("Loading configuration...")
    config = load_config()
    logger.info("Downloading dataset...")

    download_dataset(
        config['data']['dataset_url'],
        config['data']['raw_log_path']
    )
    logger.info("Parsing logs...")

    parse_logs(
        config['data']['raw_log_path'],
        "data/parsed_logs/parsed.csv"
    )
    logger.info("Building sequences...")

    build_sequences(
        "data/parsed_logs/parsed.csv",
        "data/processed/sequences.npy",
        config['model']['window_size']
    )
    logger.info("Training model...")

    train_model(
        config,
        "data/processed/sequences.npy"
    )
    logger.info("Loading sequences...")
    sequences = np.load("data/processed/sequences.npy")

    sequences_tensor = torch.tensor(
        sequences,
        dtype=torch.float32
    ).unsqueeze(-1).to(DEVICE)
    logger.info("Loading trained model...")

    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=config['model']['hidden_size'],
        latent_dim=config['model']['latent_size']
    ).to(DEVICE)

    model.load_state_dict(
        torch.load("model.pth", map_location=DEVICE)
    )
    model.eval()
    logger.info("Computing reconstruction errors...")

    with torch.no_grad():
        output = model(sequences_tensor)
        errors = (
            (sequences_tensor - output) ** 2
        ).mean(dim=(1, 2)).cpu().numpy()
    logger.info("Computing anomaly threshold...")

    threshold = compute_threshold(
        errors,
        method="percentile",
        value=config['threshold']['percentile']
    )
    anomalies = detect(errors, threshold)
    logger.info(f"Detected {anomalies.sum()} anomalies")
    save_results(errors, anomalies)
    logger.info("Evaluating model...")

    metrics = evaluate(
        anomalies,
        "data/raw_logs/anomaly_label.csv",
        config['model']['window_size']
    )
    save_experiment(config, metrics)
    logger.info("Plotting anomaly detection results...")
    plot_anomalies(errors, anomalies)

    try:
        import pandas as pd
        labels = pd.read_csv("data/raw_logs/anomaly_label.csv")
        y_true = labels['Label'].values[:len(errors)]
        plot_roc(errors, y_true)
        
    except Exception as e:
        logger.warning(f"ROC plotting skipped: {e}")
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run()