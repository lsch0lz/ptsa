import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout_rate = 0.0

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return F.dropout(x, p=self.dropout_rate, training=self.training)

class TransformerIHM(nn.Module):
    def __init__(
        self, 
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super(TransformerIHM, self).__init__()
        
        self.dropout_rate = dropout
        
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, d_model)
        torch.nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        
        encoder_layer = self._create_encoder_layer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation=activation
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.fc_logit = nn.Linear(d_model, 1)
        self.fc_log_var = nn.Linear(d_model, 1)
        
        torch.nn.init.xavier_uniform_(self.fc_logit.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.fc_log_var.weight, gain=0.1)
        
        self.d_model = d_model

    def _create_encoder_layer(self, d_model, nhead, dim_feedforward, dropout, activation):
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # Handle dropout manually
            activation=activation,
            batch_first=True,
            norm_first=True
        )

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.dropout_rate, training=self.training)

    def forward(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._apply_dropout(src)
        
        x = self.input_norm(x)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        # x = self._apply_dropout(x)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        # x = self._apply_dropout(x)
        
        x = x[:, -1, :]
        
        x = self.final_norm(x)
        x = self._apply_dropout(x)
        
        logits = self.fc_logit(x).squeeze(-1)
        log_var = self.fc_log_var(x).squeeze(-1)
        
        proba = torch.clamp(torch.sigmoid(logits), min=1e-6, max=1-1e-6)
        
        return proba, log_var

    def predict_with_uncertainty(
        self, 
        x: torch.Tensor,
        num_samples: int = 100,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()  # Enable dropout
        
        probas = []
        log_variances = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                proba, log_var = self(x, src_mask, src_key_padding_mask)
                probas.append(proba)
                log_variances.append(log_var)
        
        probas = torch.stack(probas)
        mean_proba = probas.mean(dim=0)

        mean_proba = torch.clamp(mean_proba, min=1e-6, max=1-1e-6)
        
        epistemic_uncertainty = probas.var(dim=0)

        aleatoric_uncertainty = torch.exp(torch.stack(log_variances).mean(dim=0))

        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_proba, total_uncertainty
