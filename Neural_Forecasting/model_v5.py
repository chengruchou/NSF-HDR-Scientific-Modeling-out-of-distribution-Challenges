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


class TimeAwareNFSTNDTModelV5(nn.Module):
    """
    v5 = v4 backbone + time-aware feature fusion.
    - Accepts x: (B, T, C, F)
    - Optionally accepts t: (B, T), absolute timestamps
    - Output remains (B, T, C) to preserve training loop expectations.

    Time is encoded (Time2Vec + delta_t) and concatenated to signal features
    before the temporal backbone.
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

        # v5 difference: inject time features before the axial backbone
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
