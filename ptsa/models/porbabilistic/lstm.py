import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, depth=1, dropout=0.5, rec_dropout=0.0):
        super(BayesianLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.depth = depth
        self.dropout = dropout
        self.rec_dropout = rec_dropout

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, dropout=rec_dropout, batch_first=True)
            for _ in range(depth)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x = self.input_layer(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout_layer(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        if mask is not None:
            last_timestep = mask.sum(dim=1) - 1
            last_timestep = last_timestep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            x = x.gather(1, last_timestep).squeeze(1)
        else:
            x = x[:, -1, :]

        y_pred = self.output_layer(x)
        y_pred = F.relu(y_pred)

        return y_pred

    def loss(self, pred: Tensor, truth: Tensor) -> Tensor:
        return self.loss_fn(pred, truth)

    def predict(self, X: Tensor, mask: Tensor = None, num_samples: int = 100) -> [Tensor, Tensor]:
        self.train()
        predictions = []

        for _ in range(num_samples):
            with torch.no_grad():
                pred = self(X, mask)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred