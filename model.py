import os
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# STNDT-style model backbone
# (3D input: (B,T,C) -> (B,T,C))
# -------------------------
class _AxialBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.chan_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_t = nn.LayerNorm(d_model)
        self.ln_c = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, D)
        B, T, C, D = x.shape

        # Time attention per channel
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, D)  # (B*C, T, D)
        xt_ln = self.ln_t(xt)
        attn_t, _ = self.time_attn(xt_ln, xt_ln, xt_ln, need_weights=False)
        xt = xt + attn_t
        xt = xt.view(B, C, T, D).permute(0, 2, 1, 3).contiguous()  # (B,T,C,D)

        # Channel attention per time
        xc = xt.view(B * T, C, D)  # (B*T, C, D)
        xc_ln = self.ln_c(xc)
        attn_c, _ = self.chan_attn(xc_ln, xc_ln, xc_ln, need_weights=False)
        xc = xc + attn_c
        x2 = xc.view(B, T, C, D)

        # FFN
        x3 = self.ln_f(x2)
        x3 = x2 + self.ffn(x3)
        return x3


class NFSTNDTModel(nn.Module):
    """
    Input:  (B, T=20, C)
    Output: (B, T=20, C)
    """
    def __init__(
        self,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_T: int = 20,
        max_C: int = 512,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        self.d_model = hidden_size
        self.max_T = max_T

        self.in_proj = nn.Linear(1, hidden_size)

        self.time_pos = nn.Parameter(torch.zeros(1, max_T, 1, hidden_size))
        self.chan_pos_base = nn.Parameter(torch.zeros(1, 1, max_C, hidden_size))

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _AxialBlock(d_model=hidden_size, n_heads=n_heads, dropout=dropout)
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
        # x: (B,T,C)
        B, T, C = x.shape
        if T > self.time_pos.shape[1]:
            raise ValueError(f"T={T} exceeds max_T={self.time_pos.shape[1]}")

        h = self.in_proj(x.unsqueeze(-1))  # (B,T,C,1)->(B,T,C,D)
        h = h + self.time_pos[:, :T, :, :] + self._channel_pos(C)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h)

        y = self.out_proj(h).squeeze(-1)   # (B,T,C)
        return y


# -------------------------
# Codabench submission wrapper
# -------------------------
class Model:
    """
    Codabench entry:
      - load(): load weights based on monkey_name
      - predict(X): X is numpy (N,20,C,9) (or occasionally (N,20,C))
                   return numpy (N,20,C)
    """
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name.lower().strip()

        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_file = "model_beignet.pth"
            # If you trained with different hyperparams, change them here accordingly:
            self.hidden_size = 256
            self.n_heads = 8
            self.n_layers = 6
            self.dropout = 0.1
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi.pth"
            self.hidden_size = 256
            self.n_heads = 8
            self.n_layers = 6
            self.dropout = 0.1
        else:
            raise ValueError(f"No such a monkey: {self.monkey_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NFSTNDTModel(
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_T=20,
            max_C=max(self.input_size, 512),
        ).to(self.device)

        self._is_loaded = False

    def load(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, self.weight_file)
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    @torch.no_grad()
    def predict(self, X):
        """
        X: numpy array
          - expected: (N,20,C,9) where only first 10 steps are meaningful
          - allowed:  (N,20,C)
        return:
          - (N,20,C) float32
        """
        if not self._is_loaded:
            self.load()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim == 4:
            # use feature0 only
            x0 = X[..., 0]  # (N,20,C)
        elif X.ndim == 3:
            x0 = X
        else:
            raise ValueError(f"Expected X with shape (N,20,C,9) or (N,20,C). Got {X.shape}")

        # basic shape check
        if x0.shape[1] != 20:
            raise ValueError(f"Expected T=20. Got {x0.shape[1]}")
        if x0.shape[2] != self.input_size:
            raise ValueError(f"Expected C={self.input_size} for {self.monkey_name}. Got {x0.shape[2]}")

        # enforce masking rule for inference: future repeats step9
        init_steps = 10
        x0 = x0.astype(np.float32, copy=False)
        x0_masked = x0.copy()
        x0_masked[:, init_steps:, :] = x0_masked[:, init_steps-1:init_steps, :]

        xt = torch.from_numpy(x0_masked).to(self.device)  # (N,20,C)

        y_hat = self.model(xt)  # (N,20,C)

        # enforce competition output definition: first 10 steps are given
        y_hat[:, :init_steps, :] = xt[:, :init_steps, :]

        return y_hat.detach().cpu().numpy().astype(np.float32, copy=False)
