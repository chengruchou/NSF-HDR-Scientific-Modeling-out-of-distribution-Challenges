import os
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext

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
        # 聚合鄰居資訊: (C, C) @ (B*T, C, D) -> (B*T, C, D)
        support = torch.matmul(adj, x_flat) 
        return self.projection(support).view(B, T, C, D)

class GatedAxialBlock(nn.Module):
    """
    Axial attention block with a depthwise temporal conv branch and LayerScale-style gating.
    - Temporal path: per-channel MHA + local depthwise Conv1d (captures short-range trends)
    - Channel path: per-timestep MHA across channels
    - FFN: GEGLU-style feedforward with residual gating
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.chan_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # GCN代替Channel Attention
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

        # 舊版 MHA 預設輸入是 (L, N, E)，所以轉置為 (T, B*C, D)
        xt_norm = xt_norm_org.transpose(0, 1)

        attn_t, _ = self.time_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        attn_t = attn_t.transpose(0, 1) # 轉回 (B*C, T, D) 加回殘差

        conv_out = self.temporal_conv(xt_norm_org.transpose(1, 2)).transpose(1, 2)  # depthwise conv along time
        # print('xt',xt.shape, 'attn_t',attn_t.shape,'conv_out',conv_out.shape)
        # xt torch.Size([956, 20, 1024]) attn_t torch.Size([956, 20, 1024]) conv_out torch.Size([20, 956, 1024])
        xt = xt + self.gamma_t * self.res_dropout(attn_t + conv_out)
        xt = xt.view(B, C, T, D).permute(0, 2, 1, 3).contiguous()  # back to (B, T, C, D)

        # --- Channel (per timestep) ---
        # xc = xt.view(B * T, C, D)
        # xc_norm = self.ln_c(xc)
        # attn_c, _ = self.chan_attn(xc_norm, xc_norm, xc_norm, need_weights=False)
        # xc = xc + self.gamma_c * self.res_dropout(attn_c)
        # x2 = xc.view(B, T, C, D)
        xg = self.ln_c(x)
        x2 = x + self.gcn(xg, adj)

        # --- FFN ---
        x3 = self.ln_f(x2)
        x3 = x2 + self.gamma_f * self.ffn(x3)
        return x3


class NFSTNDTModelV4(nn.Module):
    """
    Drop-in replacement for NFBaseModel, with feature gating + gated axial blocks.
    input:  (B, T, C, F)
    output: (B, T, C) for feature0 forecasting
    """
    def __init__(
        self,
        input_size: int = 96,     # C (kept for API compatibility; not hard-coded into weights)
        hidden_size: int = 256,   # d_model
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_T: int = 20,
        max_C: int = 512,
        num_features: int = 9,
        adj: torch.Tensor = None
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size (d_model) must be divisible by n_heads. Got {hidden_size} and {n_heads}.")

        self.d_model = hidden_size
        self.max_T = max_T

        if adj is None:
            # 如果沒給，預設為單位矩陣（各通道獨立）
            adj = torch.eye(input_size)
        self.register_buffer('adj', adj)

        self.feature_gate = FeatureGate(num_features)
        self.in_proj = nn.Linear(num_features, hidden_size)

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
        # base is (1,1,max_C,D). If C > max_C, repeat.
        baseC = self.chan_pos_base.shape[2]
        if C <= baseC:
            return self.chan_pos_base[:, :, :C, :]
        reps = (C + baseC - 1) // baseC
        pos = self.chan_pos_base.repeat(1, 1, reps, 1)[:, :, :C, :]
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, F) float32
        return: (B, T, C)
        """
        B, T, C, F = x.shape

        h = self.feature_gate(x)
        h = self.in_proj(h)  # (B,T,C,D)
        h = h + self.time_pos[:, :T, :, :] + self._channel_pos(C)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h, self.adj)

        y = self.out_proj(h).squeeze(-1)    # (B,T,C)
        return y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)



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
        
        self.hidden_size = 256
        self.n_heads = 4
        self.n_layers = 3
        self.dropout = 0.1

        if self.monkey_name == "beignet":
            self.input_size = 89
            self.weight_file = "model_beignet.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
        elif self.monkey_name == "affi":
            self.input_size = 239
            self.weight_file = "model_affi.pth"
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
        self.model = NFSTNDTModelV4(
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

    # @torch.inference_mode()
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

            with autocast_ctx:
                yb = self.model(xb)  # (B,20,C)

            # enforce competition output definition: first 10 steps are given
            yb[:, :init_steps, :] = xb[:, :init_steps, :, 0]
            yb = self._denorm_feature0_torch(yb)
            outputs.append(yb.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return np.array(y_np)
