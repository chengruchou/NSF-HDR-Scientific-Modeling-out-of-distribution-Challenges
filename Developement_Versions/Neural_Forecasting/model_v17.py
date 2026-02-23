"""
model_v17.py
Implementation of Neuro-Ensemble architecture (Temporal + Spatial streams with FiLM fusion)
Based on Research Plan
"""
import os
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1, num_features))

    def forward(self, x, mode: str):
        # x: (B, T, C, F)
        if mode == 'norm':
            self._mean = torch.mean(x, dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self._stdev + self._mean
            return x
        else:
            raise NotImplementedError

class TemporalBlock(nn.Module):
    """
    Channel-Independent Temporal Transformer (Likely PatchTST inspired but keeping it simple as Transformer)
    Processes (B*C, T, F)
    """
    def __init__(self, input_size, hidden_size, n_heads, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        # Using learnable positional encoding for T=20
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, hidden_size) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads, 
            dim_feedforward=hidden_size*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        # x: (B*C, T, F)
        B_C, T, F = x.shape
        h = self.embedding(x) # (B*C, T, D)
        h = h + self.pos_encoder[:, :T, :]
        h = self.transformer(h)
        return h

class SpatialBlock(nn.Module):
    """
    Population-Aware Spatial Transformer (iTransformer style)
    Processes (B, C, T*F) -> Transformer over C
    """
    def __init__(self, input_dim, hidden_size, n_heads, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads, 
            dim_feedforward=hidden_size*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        # x: (B, C, T*F) or similar representation of channel history
        h = self.embedding(x) # (B, C, D)
        h = self.transformer(h)
        return h

class NeuroEnsembleModel(nn.Module):
    def __init__(
        self,
        input_size: int, # Number of channels (C)
        num_features: int = 9, # features per channel (F)
        hidden_size: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        quantiles: list = None,
        history_steps: int = 10
    ):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.quantiles = quantiles
        self.history_steps = history_steps
        
        self.revin = RevIN(num_features)
        
        # Temporal Stream (Channel Independent)
        # We process the full T=20 sequence (masked) or T=10? 
        # v9 processes T=20. We will do the same.
        self.temporal_stream = TemporalBlock(num_features, hidden_size, n_heads, n_layers, dropout)
        
        # Spatial Stream (Population Dynamics)
        # We look at the history window (first 10 steps) to determine the "State"
        # Input dim: history_steps * num_features
        self.spatial_stream = SpatialBlock(history_steps * num_features, hidden_size, n_heads, n_layers, dropout)
        
        # FiLM Fusion
        self.film_gen = nn.Linear(hidden_size, 2 * hidden_size)
        
        # Output Head
        if self.quantiles is None:
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            self.value_head = nn.Linear(hidden_size, len(self.quantiles))

        # Change Head (Auxiliary task for spike detection)
        self.change_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, C, F)
        
        # 1. Instance Normalization
        x_norm = self.revin(x, 'norm') # (B, T, C, F)
        B, T, C, F = x_norm.shape
        
        # 2. Temporal Stream (Independent)
        # Flatten C into B -> (B*C, T, F)
        x_temp_in = x_norm.permute(0, 2, 1, 3).reshape(B*C, T, F)
        h_temp = self.temporal_stream(x_temp_in) # (B*C, T, D)
        h_temp = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3) # (B, T, C, D)
        
        # 3. Spatial Stream (Population Context)
        # Use history (0:history_steps)
        history = x_norm[:, :self.history_steps, :, :] # (B, 10, C, F)
        # Permute to (B, C, 10, F) -> flatten to (B, C, 10*F)
        x_spatial_in = history.permute(0, 2, 1, 3).reshape(B, C, -1)
        h_spatial = self.spatial_stream(x_spatial_in) # (B, C, D)
        
        # 4. Fusion (FiLM)
        # h_spatial is (B, C, D). We need to modulate h_temp (B, T, C, D).
        film_params = self.film_gen(h_spatial) # (B, C, 2*D)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # (B, C, D)
        
        # Broadcast to T
        gamma = gamma.unsqueeze(1) # (B, 1, C, D)
        beta = beta.unsqueeze(1)   # (B, 1, C, D)
        
        h_fused = h_temp * gamma + beta # (B, T, C, D)
        
        # 5. Prediction
        y = self.value_head(h_fused) # (B, T, C, Q)
        dy = self.change_head(h_fused) # (B, T, C, 1)
        
        if self.quantiles is None:
            y = y.squeeze(-1)
            
        dy = dy.squeeze(-1) # (B, T, C)

        # 6. Denormalize
        # We only predict feature index 0 (neural rate), but RevIN normalizes all F.
        # We need to denorm feature 0.
        
        mean_0 = self.revin._mean[..., 0:1] # (B, 1, C, 1)
        std_0 = self.revin._stdev[..., 0:1] # (B, 1, C, 1)
        
        # If Affine:
        if self.revin.affine:
            w = self.revin.affine_weight[..., 0:1]
            b = self.revin.affine_bias[..., 0:1]
            y = (y - b) / (w + 1e-10)
            
        y = y * std_0 + mean_0
        
        # dy is change in y. Scale by std.
        dy = dy * std_0.squeeze(-1) 
        
        return {"y_pred": y, "dy_pred": dy}


class Model:
    """
    Codabench Entry for NeuroEnsemble (v17)
    """
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name.lower().strip()
        
        self.hidden_size = 256
        self.n_heads = 4
        self.n_layers = 3
        self.dropout = 0.1
        self.quantiles = [0.1, 0.5, 0.9]
        
        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_file = "model_beignet_v17.pth"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi_v17.pth"
        else:
            raise ValueError(f"No such a monkey: {self.monkey_name}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = NeuroEnsembleModel(
            input_size=self.input_size,
            num_features=9,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            quantiles=self.quantiles,
            history_steps=10
        ).to(self.device)
        
        self._is_loaded = False
        
    def load(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, self.weight_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected weight file at {path}")
            
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
            
        if "model_state_dict" in state:
            state = state["model_state_dict"]
            
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        
    @torch.no_grad()
    def predict(self, X, batch_size: int = None):
        """
        X: (N, 20, C, 9)
        """
        if not self._is_loaded:
            self.load()
            
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        # Enforce competition rules
        init_steps = 10
        x = X.astype(np.float32, copy=False)
        
        if batch_size is None or batch_size <= 0:
            batch_size = int(os.getenv("NF_BATCH_SIZE", "0")) or 1
            
        outputs = []
        n = x.shape[0]
        
        for start in range(0, n, batch_size):
            end = min(start+batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(self.device) # (B, 20, C, 9)
            
            # Masking future (similar to v9, repeat step 9)
            xb[:, init_steps:, :, :] = xb[:, init_steps-1:init_steps, :, :]
            
            out_dict = self.model(xb)
            yb_q = out_dict["y_pred"] # (B, 20, C, Q)
            
            if yb_q.shape[-1] > 1:
                median_idx = self.quantiles.index(0.5)
                yb = yb_q[..., median_idx]
            else:
                yb = yb_q.squeeze(-1)
                
            # Overwrite history with gt
            yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
            
            outputs.append(yb.detach().cpu())
            
        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return y_np
