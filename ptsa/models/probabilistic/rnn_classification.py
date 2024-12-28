import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)


        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        
        out = nn.functional.dropout(out[:, -1, :], p=self.dropout, training=self.training)

        out = self.fc(out)
        
        return self.sigmoid(out).squeeze()

    
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
