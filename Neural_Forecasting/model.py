import os
import numpy as np
import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class GatedAxialBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.chan_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

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

        self.gamma_t = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_c = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_f = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape

        xt = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, D)
        xt_norm = self.ln_t(xt)
        attn_t, _ = self.time_attn(xt_norm, xt_norm, xt_norm, need_weights=False)

        conv_out = self.temporal_conv(xt_norm.transpose(1, 2)).transpose(1, 2)
        xt = xt + self.gamma_t * self.res_dropout(attn_t + conv_out)
        xt = xt.view(B, C, T, D).permute(0, 2, 1, 3).contiguous()

        xc = xt.view(B * T, C, D)
        xc_norm = self.ln_c(xc)
        attn_c, _ = self.chan_attn(xc_norm, xc_norm, xc_norm, need_weights=False)
        xc = xc + self.gamma_c * self.res_dropout(attn_c)
        x2 = xc.view(B, T, C, D)

        x3 = self.ln_f(x2)
        x3 = x2 + self.gamma_f * self.ffn(x3)
        return x3


class NFSTNDTModelV2(nn.Module):
    def __init__(
        self,
        input_size: int = 96,
        hidden_size: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_T: int = 20,
        max_C: int = 512,
        num_features: int = 9,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size (d_model) must be divisible by n_heads. Got {hidden_size} and {n_heads}."
            )

        self.d_model = hidden_size
        self.max_T = max_T
        self.num_features = num_features

        self.feature_gate = FeatureGate(num_features)
        self.in_proj = nn.Linear(num_features, hidden_size)

        self.time_pos = nn.Parameter(torch.zeros(1, max_T, 1, hidden_size))
        self.chan_pos_base = nn.Parameter(torch.zeros(1, 1, max_C, hidden_size))

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GatedAxialBlock(d_model=hidden_size, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, F = x.shape
        if T > self.time_pos.shape[1]:
            raise ValueError(f"T={T} exceeds max_T={self.time_pos.shape[1]}")
        if F != self.num_features:
            raise ValueError(f"Expected F={self.num_features}. Got {F}")

        h = self.feature_gate(x)
        h = self.in_proj(h)
        h = h + self.time_pos[:, :T, :, :] + self._channel_pos(C)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h)

        y = self.out_proj(h).squeeze(-1)
        return y


# -------------------------
# Codabench submission wrapper
# -------------------------
class Model:
    """
    Codabench entry:
      - load(): load weights based on monkey_name
      - predict(X): X is numpy (N,20,C,9)
                   return numpy (N,20,C)
    """
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name.lower().strip()

        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_file = "model_beignet.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
            self.hidden_size = 1024
            self.n_heads = 8
            self.n_layers = 6
            self.dropout = 0.1
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi.pth"
            self.stats_file = "train_data_average_std_affi.npz"
            self.hidden_size = 1024
            self.n_heads = 8
            self.n_layers = 6
            self.dropout = 0.1
        else:
            raise ValueError(f"No such a monkey: {self.monkey_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NFSTNDTModelV2(
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
        base = os.path.dirname(__file__)
        path = os.path.join(base, self.stats_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected stats file at {path}")
        stats = np.load(path)
        avg_cf = stats["average"].astype(np.float32, copy=False)
        std_cf = stats["std"].astype(np.float32, copy=False)
        num_feats = avg_cf.shape[1] // self.input_size
        avg_c_f = avg_cf.reshape(1, self.input_size, num_feats)
        std_c_f = std_cf.reshape(1, self.input_size, num_feats)
        combine_max = avg_c_f + 4 * std_c_f
        combine_min = avg_c_f - 4 * std_c_f
        self._denorm_params = (combine_min, combine_max)

    def _normalize(self, x):
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        return ((x - combine_min) / (combine_max - combine_min)) * 2 - 1

    def _denorm_feature0(self, x):
        if self._denorm_params is None:
            self._load_denorm_params()
        combine_min, combine_max = self._denorm_params
        f0_min = combine_min[..., 0]
        f0_max = combine_max[..., 0]
        return ((x + 1) / 2) * (f0_max - f0_min) + f0_min

    @torch.inference_mode()
    def predict(self, X):
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
        x = self._normalize(x)
        x_masked = x.copy()
        x_masked[:, init_steps:, :, :] = x_masked[:, init_steps-1:init_steps, :, :]

        xt = torch.from_numpy(x_masked).to(self.device)  # (N,20,C,9)

        y_hat = self.model(xt)  # (N,20,C)

        # enforce competition output definition: first 10 steps are given
        y_hat[:, :init_steps, :] = xt[:, :init_steps, :, 0]

        y_np = y_hat.detach().cpu().numpy().astype(np.float32, copy=False)
        y_np = self._denorm_feature0(y_np)
        return y_np.astype(np.float32, copy=False)
