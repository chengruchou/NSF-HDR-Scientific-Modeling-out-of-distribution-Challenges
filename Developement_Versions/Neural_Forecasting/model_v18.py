"""
model_v18.py
Fusion of V17 (Neuro-Ensemble Arch) and V16 (Explicit Features + Global Norm)
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. Architecture Components
# ============================================================

class TemporalBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
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
        # x: (B*C, T, F_in)
        B_C, T, F = x.shape
        h = self.embedding(x)
        h = h + self.pos_encoder[:, :T, :]
        h = self.transformer(h)
        return h

class SpatialBlock(nn.Module):
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
        h = self.embedding(x)
        h = self.transformer(h)
        return h

class NeuroEnsembleModelV18(nn.Module):
    def __init__(
        self,
        input_size: int, 
        num_features: int = 27, # 9 * 3 (x, dx, ddx)
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
        
        # Removed RevIN - we assume input is already normalized globally
        
        # Temporal Stream input: (x, dx, ddx) -> dim 
        self.temporal_stream = TemporalBlock(num_features, hidden_size, n_heads, n_layers, dropout)
        
        # Spatial Stream input: history * num_features
        self.spatial_stream = SpatialBlock(history_steps * num_features, hidden_size, n_heads, n_layers, dropout)
        
        self.film_gen = nn.Linear(hidden_size, 2 * hidden_size)
        
        dim_out = len(self.quantiles) if self.quantiles else 1
        self.value_head = nn.Linear(hidden_size, dim_out)
        self.change_head = nn.Linear(hidden_size, 1)

    def _build_change_features(self, x):
        # x: (B, T, C, 9)
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = torch.cat([dx[:, :1, :], dx], dim=1)
        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddx = torch.cat([ddx[:, :1, :], ddx], dim=1)
        return dx, ddx

    def forward(self, x):
        # x: (B, T, C, F=9) - Is normalized globally outside
        B, T, C, F = x.shape
        
        # 0. Build Feature Augmentations (V16 style)
        dx, ddx = self._build_change_features(x) # (B, T, C, 9)
        x_aug = torch.cat([x, dx, ddx], dim=-1) # (B, T, C, 27)
        F_aug = x_aug.shape[-1]
        
        # 1. Temporal Stream
        x_temp_in = x_aug.permute(0, 2, 1, 3).reshape(B*C, T, F_aug)
        h_temp = self.temporal_stream(x_temp_in) 
        h_temp = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3) # (B, T, C, D)
        
        # 2. Spatial Stream
        history = x_aug[:, :self.history_steps, :, :]
        x_spatial_in = history.permute(0, 2, 1, 3).reshape(B, C, -1)
        h_spatial = self.spatial_stream(x_spatial_in) # (B, C, D)
        
        # 3. Fusion
        film_params = self.film_gen(h_spatial)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        h_fused = h_temp * gamma + beta
        
        # 4. Heads
        y = self.value_head(h_fused)
        dy = self.change_head(h_fused)
        
        if self.quantiles is None:
            y = y.squeeze(-1)
        dy = dy.squeeze(-1) # (B, T, C)

        return {"y_pred": y, "dy_pred": dy}


# ============================================================
# 2. Wrapper for Competition (V16 Specs)
# ============================================================

class Model:
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name.lower().strip()
        
        self.hidden_size = 256
        self.n_heads = 4
        self.n_layers = 3
        self.dropout = 0.1
        self.quantiles = [0.1, 0.5, 0.9]
        
        if self.monkey_name == "beignet":
            self.channel_count = 89
            self.input_size = 89
            self.weight_file = "model_beignet_v18.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
        elif self.monkey_name == "affi":
            self.channel_count = 239
            self.input_size = 239
            self.weight_file = "model_affi_v18.pth"
            self.stats_file = "train_data_average_std_affi.npz"
        else:
            print(f"Warning: Defaulting to Affi")
            self.monkey_name = "affi"
            self.channel_count = 239
            self.input_size = 239
            self.weight_file = "model_affi_v18.pth"
            self.stats_file = "train_data_average_std_affi.npz"
            
        base = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(base, self.stats_file)
        
        # Try to locate stats file
        if not os.path.exists(stats_path):
             if os.path.exists(self.stats_file):
                 stats_path = self.stats_file
             elif os.path.exists(os.path.join("dataset", self.stats_file)):
                 stats_path = os.path.join("dataset", self.stats_file)
             else:
                 print(f"Warning: Stats file not found: {self.stats_file}")

        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.average = stats["average"].astype(np.float32, copy=False)
            self.std = stats["std"].astype(np.float32, copy=False)
        else:
            self.average = None
            self.std = None
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Init Model with 27 features (9 * 3 for x, dx, ddx)
        self.model = NeuroEnsembleModelV18(
            input_size=self.channel_count,
            num_features=27, 
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            quantiles=self.quantiles,
            history_steps=10
        ).to(self.device)
        
        self._is_loaded = False
        self._denorm_params = None

    def _load_denorm_params(self):
        """
        v9 Style Normalization Params: Mean +/- 4*Std
        """
        if self.average is None or self.std is None:
            return 
        num_feats = self.average.shape[1] // self.channel_count
        avg_c_f = self.average.reshape(1, self.channel_count, num_feats)
        std_c_f = self.std.reshape(1, self.channel_count, num_feats)
        
        combine_max = avg_c_f + 4 * std_c_f
        combine_min = avg_c_f - 4 * std_c_f
        self._denorm_params = (combine_min, combine_max)

    def _normalize_torch(self, x):
        """
        v9 Style Normalization: Map [min, max] to [-1, 1]
        """
        if self._denorm_params is None: self._load_denorm_params()
        if self._denorm_params is None: return x
        
        combine_min, combine_max = self._denorm_params
        cm = torch.from_numpy(combine_min).to(x.device)
        cx = torch.from_numpy(combine_max).to(x.device)
        
        # Min-Max Scaling to [-1, 1]
        return ((x - cm) / (cx - cm)) * 2 - 1

    def _denorm_feature0_torch(self, x):
        """
        v9 Style Denormalization
        """
        if self._denorm_params is None: self._load_denorm_params()
        if self._denorm_params is None: return x
        
        combine_min, combine_max = self._denorm_params
        f0_min = torch.from_numpy(combine_min[..., 0]).to(x.device)
        f0_max = torch.from_numpy(combine_max[..., 0]).to(x.device)
        
        # Inverse Min-Max
        return ((x + 1) / 2) * (f0_max - f0_min) + f0_min

    def load(self):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, self.weight_file)
        if not os.path.exists(path):
             path = self.weight_file

        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected weight file at {path}")
            
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
            
        if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
                
        # Fix keys just in case
        new_state = {}
        for k, v in state.items():
            if k.startswith("base_model."):
                new_state[k[11:]] = v
            else:
                new_state[k] = v
                
        self.model.load_state_dict(new_state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        
    @torch.no_grad()
    def predict(self, X, batch_size: int = None):
        if not self._is_loaded:
            self.load()
            
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        init_steps = 10
        x = X.astype(np.float32, copy=False)
        
        if batch_size is None or batch_size <= 0:
            batch_size = 16
            
        outputs = []
        n = x.shape[0]
        
        for start in range(0, n, batch_size):
            end = min(start+batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(self.device)
            
            # 1. Normalize (Min-Max Style from v9)
            xb_norm = self._normalize_torch(xb)
            
            # Masking future
            xb_norm[:, init_steps:, :, :] = xb_norm[:, init_steps-1:init_steps, :, :]
            
            out_dict = self.model(xb_norm)
            yb_q = out_dict["y_pred"]
            
            if yb_q.shape[-1] > 1:
                median_idx = self.quantiles.index(0.5)
                yb = yb_q[..., median_idx]
            else:
                yb = yb_q.squeeze(-1)
                
            # 2. Denormalize
            yb_raw = self._denorm_feature0_torch(yb)
            
            # Overwrite past
            yb_raw[:, :init_steps, :] = xb[:, :init_steps, :, 0] # Use original raw x
            
            outputs.append(yb_raw.detach().cpu())
            
        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return y_np
