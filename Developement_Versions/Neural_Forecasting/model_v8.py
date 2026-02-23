"""
this is model_v8.py, defining TimeAwareNFSTNDTModelV8 and Codabench Model wrapper
"""

import os
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    """
    Light feature-wise gating to down-weight noisy channels/features before mixing.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, F)
        return x * self.net(x)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (B, T, C, D), adj: (C, C)
        B, T, C, D = x.shape
        x_flat = x.view(B * T, C, D)
        support = torch.matmul(adj, x_flat)
        return self.projection(support).view(B, T, C, D)


class GatedAxialBlock(nn.Module):
    """
    Axial attention block with a depthwise temporal conv branch and LayerScale-style gating.
    - Temporal path: per-channel MHA + local depthwise Conv1d (captures short-range trends)
    - Channel path: per-timestep GCN across channels
    - FFN: GEGLU-style feedforward with residual gating
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.gcn = GCNLayer(d_model, d_model)

        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.res_dropout = nn.Dropout(dropout)

        self.ln_t = nn.LayerNorm(d_model)
        self.ln_c = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        # LayerScale: start small to stabilize deep stacks
        self.gamma_t = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_c = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_f = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, D)
        """
        B, T, C, D = x.shape

        # --- Temporal (per channel) ---
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, D)  # (B*C, T, D)
        xt_norm_org = self.ln_t(xt)
        xt_norm = xt_norm_org.transpose(0, 1)  # (T, B*C, D)

        attn_t, _ = self.time_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        attn_t = attn_t.transpose(0, 1)  # (B*C, T, D)

        conv_out = self.temporal_conv(xt_norm_org.transpose(1, 2)).transpose(1, 2)
        xt = xt + self.gamma_t * self.res_dropout(attn_t + conv_out)
        xt = xt.view(B, C, T, D).permute(0, 2, 1, 3).contiguous()  # (B, T, C, D)

        # --- Channel (per timestep) ---
        xg = self.ln_c(x)
        x2 = x + self.gcn(xg, adj)

        # --- FFN ---
        x3 = self.ln_f(x2)
        x3 = x2 + self.gamma_f * self.ffn(x3)
        return x3


class Time2Vec(nn.Module):
    """
    Time2Vec (Kazemi et al.)-style embedding:
      y = [w0*t + b0, sin(w1*t + b1), ..., sin(wk*t + bk)]
    Input t: (B, T) -> output (B, T, k+1)
    """
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


class TimeAwareNFSTNDTModelV8(nn.Module):
    """
    v8 keeps the v5/v7 backbone but is defined in its own module
    to allow stable imports from model_v8.py.
    - Accepts x: (B, T, C, F)
    - Optionally accepts t: (B, T), absolute timestamps
    - Output remains (B, T, C) for compatibility with v7/v8 wrappers.
    """
    def __init__(
        self,
        input_size: int = 96,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_T: int = 20,
        max_C: int = 512,
        num_features: int = 9,
        adj: torch.Tensor = None,
        use_time2vec: bool = True,
        time2vec_k: int = 8,
        use_delta_t: bool = True,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size (d_model) must be divisible by n_heads. Got {hidden_size} and {n_heads}.")

        self.d_model = hidden_size
        self.max_T = max_T
        self.use_delta_t = use_delta_t
        self.use_time2vec = use_time2vec

        if adj is None:
            adj = torch.eye(input_size)
        self.register_buffer("adj", adj)

        time_dim = 0
        if use_delta_t:
            time_dim += 1
        if use_time2vec:
            time_dim += (time2vec_k + 1)
            self.time2vec = Time2Vec(time2vec_k)
        else:
            self.time2vec = None
        self.time_dim = time_dim

        self.feature_gate = FeatureGate(num_features)
        self.in_proj = nn.Linear(num_features + time_dim, hidden_size)

        # positional embeddings (time + channel)
        self.time_pos = nn.Parameter(torch.zeros(1, max_T, 1, hidden_size))
        self.chan_pos_base = nn.Parameter(torch.zeros(1, 1, max_C, hidden_size))

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GatedAxialBlock(d_model=hidden_size, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # predict feature0 only -> output (B,T,C)
        self.out_proj = nn.Linear(hidden_size, 1)

        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.chan_pos_base, std=0.02)

    def _channel_pos(self, C: int) -> torch.Tensor:
        baseC = self.chan_pos_base.shape[2]
        if C <= baseC:
            return self.chan_pos_base[:, :, :C, :]
        reps = (C + baseC - 1) // baseC
        pos = self.chan_pos_base.repeat(1, 1, reps, 1)[:, :, :C, :]
        return pos

    def _build_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, T) absolute time
        return: (B, T, time_dim)
        """
        feats = []
        if self.use_delta_t:
            dt = t[:, 1:] - t[:, :-1]
            dt0 = dt[:, :1].clone()
            dt = torch.cat([dt0, dt], dim=1)
            dt = torch.clamp(dt, min=1e-6)
            feats.append(torch.log(dt).unsqueeze(-1))
        if self.use_time2vec and self.time2vec is not None:
            feats.append(self.time2vec(t))
        return torch.cat(feats, dim=-1) if feats else None

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, C, F)
        t: (B, T) optional timestamps
        return: (B, T, C)
        """
        B, T, C, F = x.shape

        # v8 keeps v5 time-aware fusion
        h = self.feature_gate(x)

        if self.time_dim > 0:
            if t is None:
                time_feats = torch.zeros(B, T, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                time_feats = self._build_time_features(t)
            time_feats = time_feats.unsqueeze(2).expand(B, T, C, self.time_dim)
            h = torch.cat([h, time_feats], dim=-1)

        h = self.in_proj(h)  # (B,T,C,D)
        h = h + self.time_pos[:, :T, :, :] + self._channel_pos(C)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h, self.adj)

        y = self.out_proj(h).squeeze(-1)    # (B,T,C)
        return y

    def predict(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        return self.forward(x, t)


# -------------------------
# Codabench submission wrapper
# -------------------------
class Model:
    """
    Codabench entry:
      - load(): load weights based on monkey_name
      - predict(X, batch_size): X is numpy (N,20,C,9)
                               return numpy (N,20,C)
    """
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name.lower().strip()

        self.hidden_size = 256
        self.n_heads = 4
        self.n_layers = 3
        self.dropout = 0.1

        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_file = "model_beignet_v8.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi_v8.pth"
            self.stats_file = "train_data_average_std_affi.npz"
        else:
            raise ValueError(f"No such a monkey: {self.monkey_name}")

        base = os.path.dirname(__file__)
        try:
            stats = np.load(os.path.join(base, self.stats_file))
            self.average = stats["average"].astype(np.float32, copy=False)
            self.std = stats["std"].astype(np.float32, copy=False)
        except FileNotFoundError:
            print(
                f"Warning: {self.stats_file} not found. "
                "Will load during predict."
            )
            self.average = None
            self.std = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeAwareNFSTNDTModelV8(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_T=20,
            max_C=max(self.input_size, 512),
            num_features=9,
        ).to(self.device)

        self._is_loaded = False
        self._denorm_params = None

    def _load_state_dict(self, state):
        # Accept plain state_dict or common checkpoint wrappers.
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)

    def load(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, self.weight_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected weight file at {path}")

        # weights_only is used when available (PyTorch>=2.0); falls back otherwise.
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")

        self._load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True

    def _load_denorm_params(self):
        if self.average is None or self.std is None:
            base = os.path.dirname(__file__)
            path = os.path.join(base, self.stats_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected stats file at {path}")
            stats = np.load(path)
            self.average = stats["average"].astype(np.float32, copy=False)
            self.std = stats["std"].astype(np.float32, copy=False)
        avg_cf = self.average
        std_cf = self.std
        num_feats = avg_cf.shape[1] // self.input_size
        avg_c_f = avg_cf.reshape(1, self.input_size, num_feats)
        std_c_f = std_cf.reshape(1, self.input_size, num_feats)
        combine_max = avg_c_f + 4 * std_c_f
        combine_min = avg_c_f - 4 * std_c_f
        self._denorm_params = (combine_min, combine_max)

    def _normalize_torch(self, x):
        # Bitwise-consistent with v4 normalization.
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        cm = torch.from_numpy(combine_min).to(x.device)
        cx = torch.from_numpy(combine_max).to(x.device)
        return ((x - cm) / (cx - cm)) * 2 - 1

    def _denorm_feature0_torch(self, x):
        # Bitwise-consistent with v4 feature-0 denormalization.
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        f0_min = torch.from_numpy(combine_min[..., 0]).to(x.device)
        f0_max = torch.from_numpy(combine_max[..., 0]).to(x.device)
        return ((x + 1) / 2) * (f0_max - f0_min) + f0_min

    def _build_time_tensor(self, x):
        # Synthetic timestamps for compatibility with time-aware encoder.
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        return t.unsqueeze(0).repeat(x.shape[0], 1)

    @torch.no_grad()
    def predict(self, X, batch_size: int = None):
        """
        X: numpy array
          - expected: (N,20,C,9) where only first 10 steps are meaningful
        return:
          - (N,20,C) float32
        """
        if not self._is_loaded:
            self.load()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim != 4:
            raise ValueError(f"Expected X with shape (N,20,C,9). Got {X.shape}")

        # basic shape check
        if X.shape[1] != 20:
            raise ValueError(f"Expected T=20. Got {X.shape[1]}")
        if X.shape[2] != self.input_size:
            raise ValueError(f"Expected C={self.input_size} for {self.monkey_name}. Got {X.shape[2]}")
        if X.shape[3] != 9:
            raise ValueError(f"Expected F=9. Got {X.shape[3]}")

        # enforce masking rule for inference: future repeats step9
        init_steps = 10
        x = X.astype(np.float32, copy=False)

        if batch_size is None or batch_size <= 0:
            batch_size = int(os.getenv("NF_BATCH_SIZE", "0")) or 1

        autocast_ctx = nullcontext()
        if self.device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

        outputs = []
        n = x.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(self.device)  # (B,20,C,9)

            xb = self._normalize_torch(xb)
            xb[:, init_steps:, :, :] = xb[:, init_steps - 1:init_steps, :, :]

            t_in = self._build_time_tensor(xb)
            with autocast_ctx:
                # v8 predicts a deterministic point output; if quantile heads exist
                # in other variants, we would explicitly select the median (q=0.5).
                yb = self.model(xb, t_in)  # (B,20,C)

            # enforce competition output definition: first 10 steps are given
            yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
            yb = self._denorm_feature0_torch(yb)
            outputs.append(yb.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return np.array(y_np)
