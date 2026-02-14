import json
import os
from datetime import datetime


def save_experiment(config, metrics):

    os.makedirs("experiments", exist_ok=True)

    experiment = {
        "timestamp": datetime.now().isoformat(),

        "model": {
            "window_size": config["model"]["window_size"],
            "hidden_size": config["model"]["hidden_size"],
            "latent_size": config["model"]["latent_size"],
            "learning_rate": config["model"]["learning_rate"],
            "epochs": config["model"]["epochs"],
            "batch_size": config["model"]["batch_size"]
        },

        "threshold": config["threshold"]["percentile"],

        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }
    }

    filename = f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(experiment, f, indent=4)

    print(f"[INFO] Experiment saved to {filename}")