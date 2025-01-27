import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout

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
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, d_model)
        torch.nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = self._create_encoder_layer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Two output heads: one for logits and one for aleatoric uncertainty
        self.fc_logits = nn.Linear(d_model, 1)
        self.fc_log_var = nn.Linear(d_model, 1)
        
        torch.nn.init.xavier_uniform_(self.fc_logits.weight, gain=0.1)
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
        # Input dropout
        x = self._apply_dropout(src)
        
        # Normalize and project input
        x = self.input_norm(x)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self._apply_dropout(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        x = self._apply_dropout(x)
        
        # Use last sequence element
        x = x[:, -1, :]
        
        # Final shared processing
        x = self.final_norm(x)
        x = self._apply_dropout(x)
        
        # Generate logits and uncertainty
        logits = self.fc_logits(x).squeeze(-1)
        uncertainty = self.fc_log_var(x).squeeze(-1)
        
        return logits, uncertainty
    
    def get_uncertainty_components(
        self,
        x: torch.Tensor,
        num_samples: int = 100,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get detailed uncertainty breakdown for binary classification.
        
        Returns:
            Tuple containing:
            - mean probabilities (0-1)
            - epistemic uncertainty (variance in predictions from MC dropout)
            - aleatoric uncertainty (model's direct uncertainty prediction)
            - total uncertainty (combined epistemic and aleatoric)
        """
        self.train()  # Enable dropout
        
        logits_samples = []
        uncertainty_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits, uncertainty = self(x, src_mask, src_key_padding_mask)
                probs = torch.sigmoid(logits)
                logits_samples.append(probs)
                uncertainty_samples.append(torch.sigmoid(uncertainty))  # Transform to 0-1 scale
        
        # Calculate probabilities and uncertainties
        probs_samples = torch.stack(logits_samples)
        mean_probs = probs_samples.mean(dim=0)
        
        # Epistemic uncertainty from variance in predictions
        epistemic_uncertainty = probs_samples.var(dim=0)
        
        # Aleatoric uncertainty from model's direct prediction
        aleatoric_uncertainty = torch.stack(uncertainty_samples).mean(dim=0)
        
        # Total uncertainty is the sum of both types
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_probs, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty

    def predict_with_uncertainty(
        self, 
        x: torch.Tensor,
        num_samples: int = 100,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified prediction with total uncertainty.
        
        Returns:
            Tuple[Tensor, Tensor]: (mean probabilities, total uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        log_variances = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                mean, log_var = self(x, src_mask, src_key_padding_mask)
                predictions.append(mean)
                log_variances.append(log_var)
        
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        epistemic_variance = predictions.var(dim=0)
        aleatoric_variance = torch.exp(torch.stack(log_variances).mean(dim=0))
        
        total_variance = epistemic_variance + aleatoric_variance
        
        return mean_prediction, total_variance

