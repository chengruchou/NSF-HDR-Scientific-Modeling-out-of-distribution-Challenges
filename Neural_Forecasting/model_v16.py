"""
Model Submission: v15 Architecture (GCN) + v9 Normalization (Min-Max)
Target Performance: < 48k Total MSR
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. Model Architecture (GCN + Time2Vec)
# ==========================================

class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        lin = self.w0 * t + self.b0
        per = torch.sin(t.unsqueeze(-1) * self.w.view(1, 1, -1) + self.b.view(1, 1, -1))
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)


class TimeAwareGCN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, adj):
        B, T, C, H = x.shape
        x_flat = x.reshape(B * T, C, H)
        support = torch.matmul(adj, x_flat)
        output = self.proj(support)
        output = output.reshape(B, T, C, H)
        output = self.act(output)
        output = self.dropout(output)
        return self.norm(x + output)


class NFSTNDTModelV16(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        num_features: int = 9,
        use_time2vec: bool = True,
        time2vec_k: int = 8,
        use_delta_t: bool = True,
        quantiles: list = None,
        adj: torch.Tensor = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        
        if adj is None:
            adj = torch.eye(input_size)
        self.register_buffer('adj', adj)

        self.num_features = num_features
        time_dim = 0
        if use_delta_t: time_dim += 1
        if use_time2vec:
            time_dim += (time2vec_k + 1)
            self.time2vec = Time2Vec(time2vec_k)
        else:
            self.time2vec = None
        self.use_delta_t = use_delta_t
        self.time_dim = time_dim

        self.in_proj = nn.Linear(self.num_features * 3 + time_dim, hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.gcn = TimeAwareGCN(hidden_size, dropout)

        out_dim = len(quantiles) if quantiles else 1
        self.value_head = nn.Linear(hidden_size, out_dim)
        self.change_head = nn.Linear(hidden_size, 1)

    def _build_time_features(self, t):
        feats = []
        if self.use_delta_t:
            dt = t[:, 1:] - t[:, :-1]
            dt0 = dt[:, :1].clone()
            dt = torch.cat([dt0, dt], dim=1)
            dt = torch.clamp(dt, min=1e-6)
            feats.append(torch.log(dt).unsqueeze(-1))
        if self.time2vec is not None:
            feats.append(self.time2vec(t))
        return torch.cat(feats, dim=-1) if feats else None

    def _build_change_features(self, x):
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = torch.cat([dx[:, :1, :], dx], dim=1)
        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddx = torch.cat([ddx[:, :1, :], ddx], dim=1)
        return dx, ddx

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B, T, C, F = x.shape
        dx, ddx = self._build_change_features(x)
        h = torch.cat([x, dx, ddx], dim=-1)

        if self.time_dim > 0:
            if t is None:
                t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1)
            time_feats = self._build_time_features(t)
            time_feats = time_feats.unsqueeze(2).repeat(1, 1, C, 1)
            h = torch.cat([h, time_feats], dim=-1)

        h = self.in_proj(h)
        # Reshape for Transformer: (B*C, T, H)
        h_flat = h.reshape(B * C, T, self.hidden_size).transpose(0, 1)
        h_temp = self.encoder(h_flat) 
        h_temp = h_temp.transpose(0, 1)
        
        # Reshape for GCN: (B, T, C, H)
        h_spatial_in = h_temp.reshape(B, T, C, self.hidden_size)
        h_final = self.gcn(h_spatial_in, self.adj)

        y = self.value_head(h_final)
        dy = self.change_head(h_final).squeeze(-1)
        return torch.cat([y, dy.unsqueeze(-1)], dim=-1)


# ==========================================
# 2. Wrapper for Competition (v9 Normalization)
# ==========================================

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
            self.weight_file = "model_beignet_v16.pth" # 記得存成這個檔名
            self.stats_file = "train_data_average_std_beignet.npz"
        else:
            self.channel_count = 239
            self.input_size = 239
            self.weight_file = "model_affi_v16.pth"    # 記得存成這個檔名
            self.stats_file = "train_data_average_std_affi.npz"

        base = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(base, self.stats_file)
        
        if not os.path.exists(stats_path):
             if os.path.exists(self.stats_file):
                 stats_path = self.stats_file
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
        
        # Init Model
        self.model = NFSTNDTModelV16(
            input_size=self.channel_count, 
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            num_features=9,
            quantiles=self.quantiles,
            use_time2vec=True,
            time2vec_k=8,
            use_delta_t=True,
            adj=torch.eye(self.channel_count) 
        ).to(self.device)

        self._is_loaded = False
        self._denorm_params = None

    def _load_state_dict(self, state):
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        
        new_state = {}
        for k, v in state.items():
            if k.startswith("base_model."):
                new_state[k[11:]] = v
            else:
                new_state[k] = v
        
        if 'adj' in new_state:
            print(f"Loading adj matrix shape: {new_state['adj'].shape}")
            
        self.model.load_state_dict(new_state, strict=False)

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

        self._load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    def _load_denorm_params(self):
        """
        v9 Style Normalization Params: Mean +/- 4*Std
        """
        if self.average is None or self.std is None:
            return 
        num_feats = self.average.shape[1] // self.channel_count
        avg_c_f = self.average.reshape(1, self.channel_count, num_feats)
        std_c_f = self.std.reshape(1, self.channel_count, num_feats)
        
        # 【修正】使用 v9 的 4-sigma 範圍
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

    def _build_time_tensor(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        return t.unsqueeze(0).repeat(x.shape[0], 1)

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
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(self.device)

            # 1. Normalize (Min-Max Style from v9)
            xb_norm = self._normalize_torch(xb)
            
            # Masking future
            obs = xb_norm[:, :init_steps, :, :]
            last_obs = xb_norm[:, init_steps-1:init_steps, :, :]
            future_steps = xb.shape[1] - init_steps
            future_mask = last_obs.repeat(1, future_steps, 1, 1)
            xb_masked = torch.cat([obs, future_mask], dim=1)

            t_in = self._build_time_tensor(xb_masked)
            
            # 2. Model Inference
            out_tensor = self.model(xb_masked, t_in)
            
            # Median is index 1
            yb_pred = out_tensor[..., 1] 

            # 3. Denormalize (Min-Max Style from v9)
            yb_raw = self._denorm_feature0_torch(yb_pred)
            
            # Overwrite past with ground truth
            f0_history = torch.from_numpy(x[start:end, :init_steps, :, 0]).to(self.device)
            yb_raw[:, :init_steps, :] = f0_history

            outputs.append(yb_raw.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return y_np