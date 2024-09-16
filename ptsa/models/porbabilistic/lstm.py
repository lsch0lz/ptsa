import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BayesianLSTM(nn.Module):
    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size_1 = 128
        self.hidden_size_2 = 32
        self.stacked_layers = 2
        self.dropout_probability = 0.5

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, x) -> Tensor:
        batch_size, seq_len, _ = x.size()

        hidden: [Tensor, Tensor] = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output: Tensor = F.dropout(output, p=self.dropout_probability, training=True)
        state: [Tensor, Tensor] = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output_dropout: Tensor = F.dropout(output, p=self.dropout_probability, training=True)
        output_last_cell: Tensor = output_dropout[:, -1, :]  # take the last decoder cell's outputs
        y_prediction: Tensor = self.fc(output_last_cell)

        return y_prediction

    def init_hidden1(self, batch_size) -> [Tensor, Tensor]:
        hidden_state: Tensor = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)
        cell_state: Tensor = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)

        return hidden_state, cell_state

    def init_hidden2(self, batch_size) -> [Tensor, Tensor]:
        hidden_state: Tensor = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)
        cell_state: Tensor = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)

        return hidden_state, cell_state

    def loss(self, pred, truth) -> Tensor:
        return self.loss_fn(pred, truth)

    def predict(self, X) -> Tensor:
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()
