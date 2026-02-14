import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = self.fc_enc(hidden[-1])

        # Decode
        hidden_dec = self.fc_dec(latent).unsqueeze(1)
        hidden_dec = hidden_dec.repeat(1, x.size(1), 1)

        output, _ = self.decoder(hidden_dec)

        return output
    