import os
import numpy as np
import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """
    Time2Vec 時間編碼層
    將純量時間 t 轉換為向量表示 [lin, sin, sin, ...]
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T) -> (B, T, k+1)
        lin = self.w0 * t + self.b0
        per = torch.sin(t.unsqueeze(-1) * self.w.view(1, 1, -1) + self.b.view(1, 1, -1))
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)


class TimeAwareGCN(nn.Module):
    """
    GCN 層：利用鄰接矩陣 (adj) 在空間維度上聚合特徵
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, adj):
        """
        x: (B, T, C, H) - Batch, Time, Channel, Hidden
        adj: (C, C) - Adjacency Matrix
        """
        B, T, C, H = x.shape
        
        # 將 (B, T) 視為一個大的 Batch 進行空間操作
        # 使用 reshape 避免 view 在非連續記憶體上報錯
        x_flat = x.reshape(B * T, C, H) 
        
        # GCN 運算: Output = Adj * X * W
        support = torch.matmul(adj, x_flat) # (B*T, C, H)
        output = self.proj(support)
        
        # 還原形狀
        output = output.reshape(B, T, C, H)
        output = self.act(output)
        output = self.dropout(output)
        
        # Residual Connection + LayerNorm
        return self.norm(x + output)


class NFSTNDTModelV15(nn.Module):
    """
    v9 + GCN 模型整合版
    - 輸入: (B, T, C, F)
    - 特徵: 原始值 + 一階差分(dx) + 二階差分(ddx) + 時間編碼
    - 骨幹: Transformer Encoder + GCN
    - 輸出: Quantiles + Change prediction (dy)
    """
    def __init__(
        self,
        input_size: int, # 通道數 (Channel Count)
        hidden_size: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        num_features: int = 9,
        use_time2vec: bool = True,
        time2vec_k: int = 8,
        use_delta_t: bool = True,
        quantiles: list = None,
        adj: torch.Tensor = None, # 必須傳入鄰接矩陣
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        
        # 註冊 Adjacency Matrix (不作為訓練參數，但隨模型保存)
        if adj is None:
            # 如果沒傳入，預設為單位矩陣 (無空間聚合)
            adj = torch.eye(input_size)
        self.register_buffer('adj', adj)

        self.num_features = num_features
        
        # 計算特徵維度
        time_dim = 0
        if use_delta_t: time_dim += 1
        if use_time2vec:
            time_dim += (time2vec_k + 1)
            self.time2vec = Time2Vec(time2vec_k)
        else:
            self.time2vec = None
        self.use_delta_t = use_delta_t
        self.time_dim = time_dim

        # 輸入投影層: 原始(F) + dx(F) + ddx(F) + Time(time_dim)
        self.in_proj = nn.Linear(self.num_features * 3 + time_dim, hidden_size)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # GCN 模組
        self.gcn = TimeAwareGCN(hidden_size, dropout)

        # 輸出層
        out_dim = len(quantiles) if quantiles else 1
        self.value_head = nn.Linear(hidden_size, out_dim) # 預測數值分位數
        self.change_head = nn.Linear(hidden_size, 1)      # 預測變化量 (dy)

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
        # 計算一階與二階差分特徵
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = torch.cat([dx[:, :1, :], dx], dim=1)
        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddx = torch.cat([ddx[:, :1, :], ddx], dim=1)
        return dx, ddx

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, C, F)
        B, T, C, F = x.shape

        # 1. 建構特徵 (Signal + Derivatives)
        dx, ddx = self._build_change_features(x)
        h = torch.cat([x, dx, ddx], dim=-1)

        # 2. 建構時間特徵
        if self.time_dim > 0:
            if t is None:
                t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1)
            time_feats = self._build_time_features(t) # (B, T, T_dim)
            time_feats = time_feats.unsqueeze(2).repeat(1, 1, C, 1) # 廣播到所有通道
            h = torch.cat([h, time_feats], dim=-1)

        # 3. 投影 (Projection)
        h = self.in_proj(h) # (B, T, C, Hidden)

        # 4. 時序注意力 (Temporal Attention - Transformer)
        # 需轉換為 (T, B*C, Hidden) 格式
        # 使用 reshape 替代 view 以防止 contiguous 錯誤
        h_flat = h.reshape(B * C, T, self.hidden_size).transpose(0, 1)
        
        h_temp = self.encoder(h_flat) 
        
        h_temp = h_temp.transpose(0, 1) # (B*C, T, Hidden)
        
        # 5. 空間聚合 (Spatial Mixing - GCN)
        # 恢復為 (B, T, C, Hidden) 格式以便利用 adj
        # 【修正點】: 這裡將 view 改為 reshape，解決 RuntimeError
        h_spatial_in = h_temp.reshape(B, T, C, self.hidden_size)
        
        h_final = self.gcn(h_spatial_in, self.adj)

        # 6. 輸出預測
        y = self.value_head(h_final) # (B, T, C, Q)
        dy = self.change_head(h_final).squeeze(-1) # (B, T, C)

        # 拼接輸出: [q10, q50, q90, dy]
        return torch.cat([y, dy.unsqueeze(-1)], dim=-1)
    
# ==========================================
# Wrapper for Competition
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
            self.weight_file = "model_beignet_v15.pth"
            self.stats_file = "train_data_average_std_beignet.npz"
        else:
            self.channel_count = 239
            self.input_size = 239
            self.weight_file = "model_affi_v15.pth"
            self.stats_file = "train_data_average_std_affi.npz"

        # Load Global Stats
        base = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(base, self.stats_file)
        
        print(f"DEBUG: Looking for stats at: {stats_path}")
        if not os.path.exists(stats_path):
             if os.path.exists(self.stats_file):
                 stats_path = self.stats_file
             else:
                 print(f"DEBUG: Files in dir: {os.listdir(base)}")
                 raise FileNotFoundError(f"CRITICAL: Stats file not found: {self.stats_file}")

        stats = np.load(stats_path)
        self.average = stats["average"].astype(np.float32, copy=False)
        self.std = stats["std"].astype(np.float32, copy=False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Init Model with Identity Adjacency (will be overwritten by load_state_dict)
        self.model = NFSTNDTModelV15(
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
        
        # Remove prefixes if present
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
        if self.average is None or self.std is None:
            return 
        num_feats = self.average.shape[1] // self.channel_count
        avg_c_f = self.average.reshape(1, self.channel_count, num_feats)
        std_c_f = self.std.reshape(1, self.channel_count, num_feats)
        # Store for Global Z-Score
        self._denorm_params = (avg_c_f, std_c_f)

    def _global_normalize_torch(self, x):
        if self._denorm_params is None:
            self._load_denorm_params()
        if self._denorm_params is None:
            return x
            
        avg, std = self._denorm_params
        avg_t = torch.from_numpy(avg).to(x.device)
        std_t = torch.from_numpy(std).to(x.device)
        
        # Global Z-Score: (x - mean) / std
        return (x - avg_t) / (std_t + 1e-6)

    def _global_denorm_feature0_torch(self, x):
        if self._denorm_params is None:
            self._load_denorm_params()
        if self._denorm_params is None:
            return x

        avg, std = self._denorm_params
        f0_avg = torch.from_numpy(avg[..., 0]).to(x.device)
        f0_std = torch.from_numpy(std[..., 0]).to(x.device)
        
        # Global Z-Score Inverse: x * std + mean
        return x * f0_std + f0_avg

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

            # 1. Global Normalize
            xb_global = self._global_normalize_torch(xb)
            
            # Masking future
            obs = xb_global[:, :init_steps, :, :]
            last_obs = xb_global[:, init_steps-1:init_steps, :, :]
            future_steps = xb.shape[1] - init_steps
            future_mask = last_obs.repeat(1, future_steps, 1, 1)
            xb_masked = torch.cat([obs, future_mask], dim=1)

            t_in = self._build_time_tensor(xb_masked)
            
            # 2. Model Inference
            out_tensor = self.model(xb_masked, t_in)
            
            # Median is index 1
            yb_pred = out_tensor[..., 1] 

            # 3. Inverse Global Norm Only (Feature 0)
            yb_raw = self._global_denorm_feature0_torch(yb_pred)
            
            # Overwrite past with ground truth
            f0_history = torch.from_numpy(x[start:end, :init_steps, :, 0]).to(self.device)
            yb_raw[:, :init_steps, :] = f0_history

            outputs.append(yb_raw.detach().cpu())

        y_np = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        return y_np