import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate(predictions, label_path, window_size):

    print("[INFO] Loading ground truth labels...")

    df = pd.read_csv(label_path)

    label_map = {
        "Normal": 0,
        "Anomaly": 1
    }

    true_labels = df["Label"].map(label_map).values

    true_labels = true_labels[:len(predictions)]

    predictions = predictions.astype(int)

    accuracy = accuracy_score(true_labels, predictions)

    precision = precision_score(true_labels, predictions, zero_division=0)

    recall = recall_score(true_labels, predictions, zero_division=0)

    f1 = f1_score(true_labels, predictions, zero_division=0)

    print("\n========== Evaluation Results ==========")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(confusion_matrix(true_labels, predictions))

    print("\nClassification Report:")

    print(classification_report(true_labels, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }