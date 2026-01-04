"""
this is model_v13.py

Changes from v12:
- add optional RevIN (reversible instance normalization) for per-sample, per-channel stability.
- keep quantile + change-head architecture; inference supports optional v9/v13 blending.
- maintain ensemble loading and v12-compatible IO for scoring.
"""
import os
import glob
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

ENABLE_BLEND = False
BLEND_ALPHA = 0.7
USE_REVIN = True
REVIN_AFFINE = True


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


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (per-sample, per-channel across time).
    forward_normalize(x) stores stats; reverse(x_pred) restores scale.
    """
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._cached_mean = None
        self._cached_std = None

    def forward_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=self.eps)
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        self._cached_mean = mean
        self._cached_std = std
        return x_norm

    def reverse(self, x: torch.Tensor, feature_idx: int = None) -> torch.Tensor:
        if self._cached_mean is None or self._cached_std is None:
            raise RuntimeError("RevIN reverse called before forward_normalize.")
        mean = self._cached_mean
        std = self._cached_std
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeezed = True
        if feature_idx is not None:
            mean = mean[..., feature_idx:feature_idx + 1]
            std = std[..., feature_idx:feature_idx + 1]
            if self.affine:
                gamma = self.gamma[..., feature_idx:feature_idx + 1]
                beta = self.beta[..., feature_idx:feature_idx + 1]
        else:
            gamma = self.gamma
            beta = self.beta
        x_out = x
        if self.affine:
            x_out = (x_out - beta) / torch.clamp(gamma, min=self.eps)
        x_out = x_out * std + mean
        if squeezed:
            x_out = x_out.squeeze(-1)
        return x_out

    def reverse_delta(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        if self._cached_std is None:
            raise RuntimeError("RevIN reverse_delta called before forward_normalize.")
        std = self._cached_std[..., feature_idx:feature_idx + 1]
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeezed = True
        if self.affine:
            gamma = self.gamma[..., feature_idx:feature_idx + 1]
        else:
            gamma = 1.0
        x_out = x / torch.clamp(gamma, min=self.eps)
        x_out = x_out * std
        if squeezed:
            x_out = x_out.squeeze(-1)
        return x_out


class NFSTNDTModelV9(nn.Module):
    """
    v9: change-aware, multi-head forecasting model.
    - Input: x (B, T, F), optional t (B, T)
    - Adds change-aware features (dx, ddx) before the encoder to improve spike sensitivity.
    - Two heads share the same encoder representation:
        * value head -> multi-horizon value/quantile prediction
        * change head -> multi-horizon change/spike prediction
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
        self.use_delta_t = use_delta_t
        self.time_dim = time_dim

        # Change-aware feature projection (x, dx, ddx concatenated).
        self.in_proj = nn.Linear(self.num_features * 3 + time_dim, hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

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
        return torch.cat(feats, dim=-1) if feats else None

    def _build_change_features(self, x: torch.Tensor) -> torch.Tensor:
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

        # v9 change-aware feature injection to improve spike sensitivity.
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
            y = y.squeeze(-1)  # (B, T)

        dy = self.change_head(h).squeeze(-1)  # (B, T)

        return {
            "y_pred": y,
            "dy_pred": dy,
        }


class NFSTNDTModelV13(nn.Module):
    """
    v13: v9-style change-aware transformer + optional RevIN normalization.
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
        use_revin: bool = True,
        revin_affine: bool = True,
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
        self.use_revin = use_revin

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
        self.use_delta_t = use_delta_t
        self.time_dim = time_dim

        if self.use_revin:
            self.revin = RevIN(self.num_features, affine=revin_affine)
        else:
            self.revin = None

        # Change-aware feature projection (x, dx, ddx concatenated).
        self.in_proj = nn.Linear(self.num_features * 3 + time_dim, hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

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
        return torch.cat(feats, dim=-1) if feats else None

    def _build_change_features(self, x: torch.Tensor) -> torch.Tensor:
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

        if self.use_revin:
            x = self.revin.forward_normalize(x)

        # change-aware feature injection
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
            y = y.squeeze(-1)  # (B, T)

        dy = self.change_head(h).squeeze(-1)  # (B, T)

        if self.use_revin:
            y = self.revin.reverse(y, feature_idx=0)
            dy = self.revin.reverse_delta(dy, feature_idx=0)

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
            self.weight_file = "model_beignet_v13.pth"
            self.weight_glob = "model_beignet_v13_seed*.pth"
            self.v9_weight_file = "model_beignet_v9.pth"
            self.v9_weight_glob = "model_beignet_v9_seed*.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi_v13.pth"
            self.weight_glob = "model_affi_v13_seed*.pth"
            self.v9_weight_file = "model_affi_v9.pth"
            self.v9_weight_glob = "model_affi_v9_seed*.pth"
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
        self.models = []
        self.models_v9 = []

        self.enable_blend = bool(int(os.getenv("NF_ENABLE_BLEND", "1" if ENABLE_BLEND else "0")))
        self.blend_alpha = float(os.getenv("NF_BLEND_ALPHA", str(BLEND_ALPHA)))
        self.use_revin = bool(int(os.getenv("NF_USE_REVIN", "1" if USE_REVIN else "0")))
        self.revin_affine = bool(int(os.getenv("NF_REVIN_AFFINE", "1" if REVIN_AFFINE else "0")))

        self._is_loaded = False
        self._denorm_params = None

    def _load_state_dict(self, model, state):
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
        model.load_state_dict(state, strict=True)

    def _load_ensemble(self, model_cls, weight_glob, weight_file, model_kwargs):
        base = os.path.dirname(__file__)
        ckpt_paths = sorted(glob.glob(os.path.join(base, weight_glob)))
        if not ckpt_paths:
            path = os.path.join(base, weight_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected weight file at {path}")
            ckpt_paths = [path]

        models = []
        for ckpt in ckpt_paths:
            model = model_cls(**model_kwargs).to(self.device)
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(ckpt, map_location="cpu")
            self._load_state_dict(model, state)
            model.to(self.device)
            model.eval()
            models.append(model)
        return models

    def load(self):
        self.models = self._load_ensemble(
            NFSTNDTModelV13,
            self.weight_glob,
            self.weight_file,
            dict(
                input_size=9,
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout,
                num_features=9,
                quantiles=self.quantiles,
                use_revin=self.use_revin,
                revin_affine=self.revin_affine,
            ),
        )
        print(f"Loaded {len(self.models)} v13 checkpoint(s) for ensemble inference.")

        if self.enable_blend:
            self.models_v9 = self._load_ensemble(
                NFSTNDTModelV9,
                self.v9_weight_glob,
                self.v9_weight_file,
                dict(
                    input_size=9,
                    hidden_size=self.hidden_size,
                    n_heads=self.n_heads,
                    n_layers=self.n_layers,
                    dropout=self.dropout,
                    num_features=9,
                    quantiles=self.quantiles,
                ),
            )
            print(f"Loaded {len(self.models_v9)} v9 checkpoint(s) for blending.")

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

    def _forward_base(self, model, x, t):
        # x: (B, T, C, F) -> run base model on (B*C, T, F)
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
                preds = []
                for model in self.models:
                    yb_q = self._forward_base(model, xb, t_in)  # (B,20,C,Q)
                    if yb_q.shape[-1] > 1:
                        median_idx = self.quantiles.index(0.5)
                        yb = yb_q[..., median_idx]
                    else:
                        yb = yb_q.squeeze(-1)
                    preds.append(yb)
                yb = torch.stack(preds, dim=0).mean(dim=0)

                if self.enable_blend:
                    preds_v9 = []
                    for model in self.models_v9:
                        yb_q_v9 = self._forward_base(model, xb, t_in)
                        if yb_q_v9.shape[-1] > 1:
                            median_idx = self.quantiles.index(0.5)
                            yb_v9 = yb_q_v9[..., median_idx]
                        else:
                            yb_v9 = yb_q_v9.squeeze(-1)
                        preds_v9.append(yb_v9)
                    yb_v9 = torch.stack(preds_v9, dim=0).mean(dim=0)
                    yb = self.blend_alpha * yb_v9 + (1.0 - self.blend_alpha) * yb

            # enforce competition output definition: first 10 steps are given
            yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
            yb = self._denorm_feature0_torch(yb)
            outputs.append(yb.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return np.array(y_np)
