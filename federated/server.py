import flwr as fl
import torch
from model.lstm_autoencoder import LSTMAutoencoder

def save_global_model(parameters):
    model = LSTMAutoencoder(1,64,16)
    state_dict=dict(zip(
        model.state_dict().keys(),
        parameters
    ))
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(),"global_model.pth")
    print("Global model saved")

def main():
    print("Starting Federated Server...")
    fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "avg_loss": sum(m["loss"] for m in metrics)/len(metrics)
        }
    )
)

if __name__=="__main__":
    main()