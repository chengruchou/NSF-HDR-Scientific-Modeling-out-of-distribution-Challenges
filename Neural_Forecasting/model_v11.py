"""
v11: ensemble-only wrapper around v9-equivalent model behavior.
"""
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


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


class NFSTNDTModelV11(nn.Module):
    """
    v11: architecture is identical to v9.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        num_features: int = None,
        use_time2vec: bool = True,
        time2vec_k: int = 8,
        use_delta_t: bool = True,
        quantiles: list = None,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size must be divisible by n_heads. Got {hidden_size} and {n_heads}.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.quantiles = quantiles

        self.num_features = num_features if num_features is not None else input_size

        time_dim = 0
        if use_delta_t:
            time_dim += 1
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
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        if self.quantiles is None:
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            self.value_head = nn.Linear(hidden_size, len(self.quantiles))

        self.change_head = nn.Linear(hidden_size, 1)

    def _build_time_features(self, t: torch.Tensor) -> torch.Tensor:
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

    def _build_change_features(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = torch.cat([dx[:, :1, :], dx], dim=1)

        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddx = torch.cat([ddx[:, :1, :], ddx], dim=1)

        return dx, ddx

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> dict:
        B, T, F = x.shape

        dx, ddx = self._build_change_features(x)
        h = torch.cat([x, dx, ddx], dim=-1)

        if self.time_dim > 0:
            if t is None:
                time_feats = torch.zeros(B, T, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                time_feats = self._build_time_features(t)
            h = torch.cat([h, time_feats], dim=-1)

        h = self.in_proj(h)
        h = self.encoder(h)

        y = self.value_head(h)
        if self.quantiles is None:
            y = y.squeeze(-1)

        dy = self.change_head(h).squeeze(-1)

        return {
            "y_pred": y,
            "dy_pred": dy,
        }


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
        self.quantiles = [0.1, 0.5, 0.9]

        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_files = [
                "model_beignet_v9_seed0.pth",
                "model_beignet_v9_seed1.pth",
                "model_beignet_v9_seed2.pth",
            ]
            self.stats_file = "train_data_average_std_beignet.npz"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_files = [
                "model_affi_v9_seed0.pth",
                "model_affi_v9_seed1.pth",
                "model_affi_v9_seed2.pth",
            ]
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
        self.models = [
            NFSTNDTModelV11(
                input_size=9,
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout,
                num_features=9,
                quantiles=self.quantiles,
            ).to(self.device)
            for _ in self.weight_files
        ]

        self._is_loaded = False
        self._denorm_params = None

    def _load_state_dict(self, model, state):
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        keys = list(state.keys())
        print("Loaded keys:", keys[:20])
        assert not any(k.startswith("base_model.") for k in keys), "Checkpoint has base_model.* prefix; save base_model.state_dict() instead"
        assert not any(k.startswith("head.") for k in keys), "Checkpoint has head.*; save base_model.state_dict() instead"
        model.load_state_dict(state, strict=True)

    def load(self):
        base = os.path.dirname(__file__)
        for model, weight_file in zip(self.models, self.weight_files):
            path = os.path.join(base, weight_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected weight file at {path}")

            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")

            self._load_state_dict(model, state)
            model.to(self.device)
            model.eval()

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
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        cm = torch.from_numpy(combine_min).to(x.device)
        cx = torch.from_numpy(combine_max).to(x.device)
        return ((x - cm) / (cx - cm)) * 2 - 1

    def _denorm_feature0_torch(self, x):
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        f0_min = torch.from_numpy(combine_min[..., 0]).to(x.device)
        f0_max = torch.from_numpy(combine_max[..., 0]).to(x.device)
        return ((x + 1) / 2) * (f0_max - f0_min) + f0_min

    def _build_time_tensor(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        return t.unsqueeze(0).repeat(x.shape[0], 1)

    def _forward_model(self, model, x, t):
        B, T, C, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, F)
        t_flat = t.unsqueeze(1).repeat(1, C, 1).view(B * C, T)
        out = model(x_flat, t_flat)
        y_pred = out["y_pred"]
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(-1)
        Q = y_pred.shape[-1]
        y_pred = y_pred.view(B, C, T, Q).permute(0, 2, 1, 3).contiguous()
        return y_pred

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

        if X.shape[1] != 20:
            raise ValueError(f"Expected T=20. Got {X.shape[1]}")
        if X.shape[2] != self.input_size:
            raise ValueError(f"Expected C={self.input_size} for {self.monkey_name}. Got {X.shape[2]}")
        if X.shape[3] != 9:
            raise ValueError(f"Expected F=9. Got {X.shape[3]}")

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
            xb = torch.from_numpy(x[start:end]).to(self.device)

            xb = self._normalize_torch(xb)
            xb[:, init_steps:, :, :] = xb[:, init_steps - 1:init_steps, :, :]

            t_in = self._build_time_tensor(xb)
            yb_accum = None
            with autocast_ctx:
                for model in self.models:
                    yb_q = self._forward_model(model, xb, t_in)

                    if yb_q.shape[-1] > 1:
                        median_idx = self.quantiles.index(0.5)
                        yb = yb_q[..., median_idx]
                    else:
                        yb = yb_q.squeeze(-1)

                    yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
                    yb = self._denorm_feature0_torch(yb)

                    if yb_accum is None:
                        yb_accum = yb.float()
                    else:
                        yb_accum = yb_accum + yb.float()

            yb_mean = yb_accum / float(len(self.models))
            outputs.append(yb_mean.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return np.array(y_np)
