import torch
from torch import nn
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLOS(nn.Module):
    def __init__(
        self, 
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,  # Reduced from 8
        num_layers: int = 2,  # Reduced from 3
        dim_feedforward: int = 256,  # Reduced from 512
        dropout: float = 0.1,  # Reduced from 0.2
        activation: str = "gelu"  # Changed from relu
    ):
        super(TransformerLOS, self).__init__()
        
        # Layer Normalization before projection
        self.input_norm = nn.LayerNorm(input_size)
        
        # Input projection with smaller initialization
        self.input_projection = nn.Linear(input_size, d_model)
        torch.nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Important: Apply normalization first
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Additional normalization before final projection
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection with smaller initialization
        self.fc = nn.Linear(d_model, 1)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        
        self.d_model = d_model
        self.dropout_rate = dropout
        
    def forward(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_size]
            src_mask: Optional mask for self-attention
            src_key_padding_mask: Optional mask for padded elements
        """
        # Normalize input
        x = self.input_norm(src)
        
        # Project input to d_model dimensions with scaling
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Use the last sequence element for prediction
        x = x[:, -1, :]
        
        # Final normalization and projection
        x = self.final_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x.squeeze(-1)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor,
        num_samples: int = 100,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo Dropout inference to estimate uncertainty.
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = self(x, src_mask, src_key_padding_mask)
                predictions.append(output)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance
