import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, batch_size, output_length, hidden_dim, hidden_size_2, num_layers, dropout, device):
        super(BayesianLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size_1 = hidden_dim
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.lstm1 = nn.LSTM(input_dim,
                             self.hidden_size_1,
                             num_layers=self.num_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.num_layers,
                             batch_first=True)
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

        self.to(device)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)

        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout, training=self.training)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output[:, -1, :]  # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred

    def init_hidden1(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size_1, device=self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size_1, device=self.device)
        return (hidden_state, cell_state)

    def init_hidden2(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size_2, device=self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size_2, device=self.device)
        return (hidden_state, cell_state)

    def loss(self, pred, truth):
        # Ensure both pred and truth are on the correct device
        pred = pred.to(self.device)
        truth = truth.to(self.device)
        return self.loss_fn(pred, truth)

    def predict(self, X: Tensor, num_samples: int = 100) -> [Tensor, Tensor]:
        self.train()  # Ensure dropout is active
        # Ensure input is on the correct device
        X = X.to(self.device)

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self(X)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        return mean_pred, std_pred