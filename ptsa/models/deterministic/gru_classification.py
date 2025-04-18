import torch
from torch import nn
from torch import Tensor


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        h0: Tensor = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        out: Tensor = self.fc(out[:, -1, :])

        return self.sigmoid(out).squeeze()
