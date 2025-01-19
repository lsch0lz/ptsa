from typing import List

import torch
from torch import nn
from torch import Tensor

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_mean = nn.Linear(hidden_size, 1)  # Mean prediction
        self.fc_log_var = nn.Linear(hidden_size, 1)  # Log variance prediction
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        
        mean = self.fc_mean(out[:, -1, :])
        log_var = self.fc_log_var(out[:, -1, :])  # Predict log variance for numerical stability
        
        return mean.squeeze(1), log_var.squeeze(1)

    def predict_with_uncertainty(self, x, num_samples=100):
        self.train()  # Enable dropout for MC sampling

        predictions = []
        log_variances = []

        for _ in range(num_samples):
            with torch.no_grad():
                mean, log_var = self(x)
                predictions.append(mean)
                log_variances.append(log_var)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)

        # Epistemic uncertainty
        model_variance = predictions.var(dim=0)
        
        # Aleatoric uncertainty (convert log variance to variance)
        aleatoric_variance = torch.exp(torch.stack(log_variances).mean(dim=0))

        # Total uncertainty = epistemic + aleatoric
        total_variance = model_variance + aleatoric_variance

        return mean, total_variance

