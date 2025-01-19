from typing import List

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        mask = torch.ones_like(h0).bernoulli_(1 - self.dropout_rate).to(x.device) / (1 - self.dropout_rate)

        h0 = h0 * mask
        c0 = c0 * mask

        output, (hidden, cell) = self.lstm(x, (h0, c0))

        last_hidden = F.dropout(hidden[-1], p=self.dropout_rate, training=self.training)

        out = self.fc(last_hidden)
        return out.squeeze(1)

    def predict_with_uncertainty(self, x, num_samples=100):
        self.train()

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self(x)
                predictions.append(output)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0) 

        return mean, variance
