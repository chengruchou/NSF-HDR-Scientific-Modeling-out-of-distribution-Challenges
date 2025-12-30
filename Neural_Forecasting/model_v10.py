"""
this is model_v10.py
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


class NFSTNDTModelV10(nn.Module):
    """
    v10: change-to-value coupling + dual encoders + weak periodic bias.
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
        alpha: float = 0.1,
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
        self.alpha = alpha

        # If num_features is not provided, assume input_size already equals F.
        self.num_features = num_features if num_features is not None else input_size

        time_dim = 0
        if use_delta_t:
            time_dim += 1
        if use_time2vec:
            time_dim += (time2vec_k + 1)
            self.time2vec = Time2Vec(time2vec_k)
        else:
            self.time2vec = None
        # v10: weak periodic inductive bias
        self.periods = [24, 168]
        time_dim += 2 * len(self.periods)

        self.use_delta_t = use_delta_t
        self.time_dim = time_dim

        # value encoder (level)
        self.in_proj_value = nn.Linear(self.num_features + time_dim, hidden_size)
        enc_layer_value = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder_value = nn.TransformerEncoder(enc_layer_value, num_layers=n_layers)

        # change encoder (dx, ddx)
        self.in_proj_change = nn.Linear(self.num_features * 2 + time_dim, hidden_size)
        enc_layer_change = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder_change = nn.TransformerEncoder(enc_layer_change, num_layers=n_layers)

        # fuse representations
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)

        # Value head: predict values or quantiles (backward compatible with v7/v8).
        if self.quantiles is None:
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            self.value_head = nn.Linear(hidden_size, len(self.quantiles))

        # Change head: predicts delta or spike score per horizon.
        self.change_head = nn.Linear(hidden_size, 1)

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
        if self.time2vec is not None:
            feats.append(self.time2vec(t))
        # periodic features
        for p in self.periods:
            angle = 2.0 * torch.pi * t / float(p)
            feats.append(torch.sin(angle).unsqueeze(-1))
            feats.append(torch.cos(angle).unsqueeze(-1))
        return torch.cat(feats, dim=-1) if feats else None

    def _build_change_features(self, x: torch.Tensor) -> tuple:
        """
        Build change-aware features:
          dx  = x[:, 1:] - x[:, :-1]
          ddx = dx[:, 1:] - dx[:, :-1]
        Pad to length T to preserve the time dimension.
        """
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = torch.cat([dx[:, :1, :], dx], dim=1)

        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddx = torch.cat([ddx[:, :1, :], ddx], dim=1)

        return dx, ddx

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> dict:
        """
        x: (B, T, F)
        t: (B, T) optional timestamps
        return:
          {
            "y_pred": (B, T, Q) or (B, T),
            "dy_pred": (B, T),
          }
        """
        B, T, F = x.shape

        if self.time_dim > 0:
            if t is None:
                time_feats = torch.zeros(B, T, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                time_feats = self._build_time_features(t)
        else:
            time_feats = None

        # value path
        if time_feats is None:
            v_in = x
        else:
            v_in = torch.cat([x, time_feats], dim=-1)
        v = self.in_proj_value(v_in)
        v = self.encoder_value(v)

        # change path
        dx, ddx = self._build_change_features(x)
        if time_feats is None:
            c_in = torch.cat([dx, ddx], dim=-1)
        else:
            c_in = torch.cat([dx, ddx, time_feats], dim=-1)
        c = self.in_proj_change(c_in)
        c = self.encoder_change(c)

        # fuse
        h = torch.cat([v, c], dim=-1)
        h = self.fuse(h)

        y_base = self.value_head(h)
        if self.quantiles is None:
            y_base = y_base.squeeze(-1)  # (B, T)

        dy = self.change_head(h).squeeze(-1)  # (B, T)

        # change-to-value coupling
        # if y_base.dim() == 3:
        #     y_corr = y_base + self.alpha * torch.sigmoid(dy).unsqueeze(-1)
        # else:
        #     y_corr = y_base + self.alpha * torch.sigmoid(dy)

        # return {
        #     "y_pred": y_corr,
        #     "dy_pred": dy,
        # }

        # dy_pred: (B, T)
        # y_base: (B, T) or (B, T, Q)

        corr = torch.tanh(dy)  # zero-mean, bounded

        if y_base.dim() == 3:
            corr = corr.unsqueeze(-1)

        y_corr = y_base + self.alpha * corr

        return {
            "y_pred": y_corr,
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
            self.weight_file = "model_beignet_v10.pth"
            self.stats_file = "train_data_average_std_beignet_v10.npz"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi_v10.pth"
            self.stats_file = "train_data_average_std_affi_v10.npz"
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
        self.model = NFSTNDTModelV10(
            input_size=9,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            num_features=9,
            quantiles=self.quantiles,
            alpha=0.1,
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
        keys = list(state.keys())
        print("Loaded keys:", keys[:20])
        assert not any(k.startswith("base_model.") for k in keys), "Checkpoint has base_model.* prefix; save base_model.state_dict() instead"
        assert not any(k.startswith("head.") for k in keys), "Checkpoint has head.*; save base_model.state_dict() instead"
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
        # self._load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), path), weights_only=True))
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

    def _forward_v10(self, x, t):
        # x: (B, T, C, F) -> run v10 on (B*C, T, F)
        B, T, C, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, F)
        t_flat = t.unsqueeze(1).repeat(1, C, 1).view(B * C, T)
        out = self.model(x_flat, t_flat)
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
                yb_q = self._forward_v10(xb, t_in)  # (B,20,C,Q)

            if yb_q.shape[-1] > 1:
                q_map = {float(q): i for i, q in enumerate(self.quantiles)}
                if 0.1 in q_map and 0.5 in q_map and 0.9 in q_map:
                    q10 = yb_q[..., q_map[0.1]]
                    q50 = yb_q[..., q_map[0.5]]
                    q90 = yb_q[..., q_map[0.9]]
                    yb = 0.7 * q50 + 0.15 * q10 + 0.15 * q90
                elif 0.5 in q_map:
                    yb = yb_q[..., q_map[0.5]]
                else:
                    yb = yb_q.mean(dim=-1)
            else:
                yb = yb_q.squeeze(-1)

            # enforce competition output definition: first 10 steps are given
            yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
            yb = self._denorm_feature0_torch(yb)
            outputs.append(yb.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return np.array(y_np)
