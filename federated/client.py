import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.lstm_autoencoder import LSTMAutoencoder
from model.trainer import train_one_epoch
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.model = LSTMAutoencoder(input_dim=1, hidden_dim=64, latent_dim=16)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        data_tensor = torch.tensor(self.data).float()

        # Add feature dimension
        data_tensor = data_tensor.unsqueeze(-1)   # (N, seq_len) → (N, seq_len, 1)
        torch.save(self.model.state_dict(), f"client_model_{id(self)}.pth")
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        train_one_epoch(self.model, loader)

        return self.get_parameters(config), len(self.data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        data_tensor = torch.tensor(self.data).float()
        data_tensor = data_tensor.unsqueeze(-1)   # (N, seq_len) → (N, seq_len, 1)

        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, len(self.data), {}


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(data_path),
    )