import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import sys
from tqdm import tqdm

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_v18 import NeuroEnsembleModelV18
from trainer import Trainer # Optional, but we use CustomTrainer

# ============================================================
# Normalized Dataset
# ============================================================
class NormalizedDataset(Dataset):
    def __init__(self, raw_data, stats_file):
        """
        raw_data: (N, T, C, F)
        """
        self.raw_data = raw_data
        
        # Load stats
        stats = np.load(stats_file)
        self.avg_flat = stats["average"] # (1, C*F)
        self.std_flat = stats["std"]     # (1, C*F)
        
        # Reshape to (1, 1, C, F) for broadcasting
        # raw_data.shape[2] is C
        C = raw_data.shape[2]
        F = raw_data.shape[3]
        
        self.avg = self.avg_flat.reshape(1, 1, C, F)
        self.std = self.std_flat.reshape(1, 1, C, F)
        
        # Compute denom params (Min-Max like V16)
        self.combine_min = self.avg - 4 * self.std
        self.combine_max = self.avg + 4 * self.std
        
        # Normalize Data (Pre-calculate)
        print("Pre-normalizing data...")
        self.norm_data = ((self.raw_data - self.combine_min) / (self.combine_max - self.combine_min)) * 2 - 1
        self.norm_data = np.clip(self.norm_data, -10, 10) 
        
        self.norm_data = torch.from_numpy(self.norm_data.astype(np.float32))
        self.raw_data_torch = torch.from_numpy(self.raw_data.astype(np.float32))

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.norm_data[index], self.raw_data_torch[index]


class DenormWrapper(nn.Module):
    def __init__(self, model, stats_file, num_channels):
        super().__init__()
        self.model = model
        
        stats = np.load(stats_file)
        avg_flat = stats["average"]
        std_flat = stats["std"]
        
        avg = avg_flat.reshape(num_channels, 9)
        std = std_flat.reshape(num_channels, 9)
        
        # Feature 0
        avg_0 = avg[:, 0:1] # (C, 1)
        std_0 = std[:, 0:1] # (C, 1)
        
        combine_min_0 = avg_0 - 4 * std_0
        combine_max_0 = avg_0 + 4 * std_0
        
        self.register_buffer('f0_min', torch.from_numpy(combine_min_0).float()) 
        self.register_buffer('f0_max', torch.from_numpy(combine_max_0).float()) 
        
    def forward(self, x_norm, t=None):
        out_dict = self.model(x_norm)
        y_pred_norm = out_dict["y_pred"] # (B, T, C, Q) or (B, T, C)
        dy_pred_norm = out_dict["dy_pred"] # (B, T, C)
        
        if y_pred_norm.dim() == 3:
            y_pred_norm = y_pred_norm.unsqueeze(-1)
            
        # Expand buffers
        f0_min = self.f0_min.unsqueeze(0).unsqueeze(0)
        f0_max = self.f0_max.unsqueeze(0).unsqueeze(0)
        
        y_raw = ((y_pred_norm + 1) / 2) * (f0_max - f0_min) + f0_min
        
        scale = (f0_max - f0_min) / 2
        dy_raw = dy_pred_norm.unsqueeze(-1) * scale
        
        return torch.cat([y_raw, dy_raw], dim=-1)

# Loss Function utils
def pinball_loss(y_hat, y, q):
    e = y - y_hat
    return torch.maximum(q * e, (q - 1.0) * e)

def quantile_order_penalty(q10, q50, q90):
    return (F.relu(q10 - q50).mean() + F.relu(q50 - q90).mean())

def spike_aware_loss(pred_tensor, target, quantiles, tau, spike_weight=2.0, lambda_change=0.2):
    """
    pred_tensor: (B, T, C, Q+1)
    target: (B, T, C) - Already Feature 0 Only
    """
    Q = len(quantiles)
    y_pred = pred_tensor[..., :Q]
    dy_pred = pred_tensor[..., Q]
    
    y_true = target # Already (B, T, C)
    
    # delta y true
    dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    dy_true = torch.cat([dy_true[:, :1, :], dy_true], dim=1)

    # spike mask
    spike_mask = (torch.abs(dy_true) > tau).float()
    w = 1.0 + spike_mask * (spike_weight - 1.0)

    # Quantile Losses
    q_losses = []
    for k, q in enumerate(quantiles):
        q_loss = pinball_loss(y_pred[..., k], y_true, q)
        q_losses.append((q_loss * w).mean())
    quant_loss = sum(q_losses) / len(q_losses)

    # Order penalty
    q10, q50, q90 = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]
    ord_loss = quantile_order_penalty(q10, q50, q90)

    # Change loss
    change_loss = (F.l1_loss(dy_pred.squeeze(-1), dy_true, reduction='none') * w).mean()

    lambda_quant = 1.0
    lambda_ord = 5.0

    total_loss = (
        lambda_quant * quant_loss +
        lambda_change * change_loss +
        lambda_ord * ord_loss
    )
    return total_loss

class CustomTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, loss_fn, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.best_val = float('inf')
        self.init_steps = 10
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            for x_norm, x_raw in pbar:
                x_norm = x_norm.to(self.device)
                x_raw = x_raw.to(self.device)
                
                # Mask Future in Input
                x_norm_masked = x_norm.clone()
                x_norm_masked[:, self.init_steps:, :, :] = x_norm_masked[:, self.init_steps-1:self.init_steps, :, :]
                
                self.optimizer.zero_grad()
                out = self.model(x_norm_masked)
                
                # Slice Future for Loss
                y_pred_future = out[:, self.init_steps:, :, :]
                y_true_future = x_raw[:, self.init_steps:, :, 0] # Feature 0
                
                loss = self.loss_fn(y_pred_future, y_true_future)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            mean_train_loss = np.mean(train_losses)
            if self.scheduler:
                self.scheduler.step()
                
            val_loss = self.validate()
            print(f"Epoch {epoch+1} | Train: {mean_train_loss:.4f} | Val: {val_loss:.4f}")
            
            if val_loss < self.best_val:
                self.best_val = val_loss
                print(f"Saving best model to {self.save_path}...")
                # Save just the base model weights
                torch.save(self.model.model.state_dict(), self.save_path)
                
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        for x_norm, x_raw in self.val_loader:
            x_norm = x_norm.to(self.device)
            x_raw = x_raw.to(self.device)
            x_norm[:, self.init_steps:, :, :] = x_norm[:, self.init_steps-1:self.init_steps, :, :]
            
            out = self.model(x_norm)
            y_pred_future = out[:, self.init_steps:, :, :]
            y_true_future = x_raw[:, self.init_steps:, :, 0]
            loss = self.loss_fn(y_pred_future, y_true_future)
            losses.append(loss.item())
        return np.mean(losses)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affi', choices=['affi', 'beignet'])
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    dataset_name = args.dataset
    num_epochs = args.epochs

    if dataset_name == 'beignet':
        num_channels = 89
    else:
        num_channels = 239

    batch_size = 16
    learning_rate = 3e-4
    hidden_size = 256
    n_heads = 4
    n_layers = 3
    dropout = 0.1
    
    print(f"Loading data for {dataset_name}...")
    try:
        raw = np.load(f'dataset/train_data_{dataset_name}.npz')['arr_0']
    except FileNotFoundError:
        raw = np.load(f'train_data_{dataset_name}.npz')['arr_0']
        
    num_total = len(raw)
    train_end = int(num_total * 0.8)
    val_end = int(num_total * 0.9)

    train_data = raw[:train_end]
    val_data = raw[train_end:val_end]
    test_data = raw[val_end:]

    stats_file = f"train_data_average_std_{dataset_name}.npz"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Run gen_stats.py first to create {stats_file}")

    train_ds = NormalizedDataset(train_data, stats_file)
    val_ds = NormalizedDataset(val_data, stats_file)
    test_ds = NormalizedDataset(test_data, stats_file)

    train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Calculate Tau
    f0 = train_data[..., 0]
    dy = f0[:, 1:, :] - f0[:, :-1, :]
    abs_dy = np.abs(dy.reshape(-1))
    tau = float(np.percentile(abs_dy, 95))
    print(f"Spike threshold tau: {tau:.6f}")

    # Initialize Model
    quantiles = [0.1, 0.5, 0.9]
    base_model = NeuroEnsembleModelV18(
        input_size=num_channels,
        num_features=27, # 9 * 3
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        quantiles=quantiles,
        history_steps=10
    ).to(torch.device('cuda'))

    model = DenormWrapper(base_model, stats_file, num_channels).to(torch.device('cuda'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** (epoch // 20))
    save_path = f'model_{dataset_name}_v18.pth'

    print("Starting training V18...")
    trainer = CustomTrainer(
        model, 
        train_data_loader, 
        val_data_loader, 
        optimizer, 
        scheduler, 
        torch.device('cuda'), 
        num_epochs, 
        lambda y_hat, y_true: spike_aware_loss(y_hat, y_true, quantiles, tau),
        save_path
    )
    trainer.train()
    print("Best val:", trainer.best_val)
    print("Training Complete.")

if __name__ == "__main__":
    train()
