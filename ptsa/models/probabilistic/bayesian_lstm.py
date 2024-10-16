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
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        h0: Tensor = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0: Tensor = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out: Tensor = nn.functional.dropout(out, p=self.dropout, training=self.training)
        
        out: Tensor = self.fc(out[:, -1, :])
   
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
