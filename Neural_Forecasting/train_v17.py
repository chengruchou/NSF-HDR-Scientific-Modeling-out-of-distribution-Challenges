
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import sys

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_v17 import NeuroEnsembleModel
from utils import compute_adj_matrix
from trainer import Trainer

# ============================================================
# Raw Dataset (No normalization, leave it to RevIN)
# ============================================================
class RawNeuroForcastDataset(Dataset):
    def __init__(self, neural_data):
        """
        neural_data: N*T*C*F
        No normalization applied here.
        """
        self.data = neural_data
        # Compute dummy stats just in case something needs them, but we won't use them for normalization
        # reshaping to (N*T, C*F) to match utils.normalize approach roughly
        flat_data = self.data.reshape(-1, self.data.shape[-2] * self.data.shape[-1])
        self.average = np.mean(flat_data, axis=0, keepdims=True)
        self.std = np.std(flat_data, axis=0, keepdims=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return torch.tensor(data, dtype=torch.float32)

# ============================================================
# Wrapper for NeuroEnsemble to match Trainer expectation
# ============================================================
class NeuroEnsembleWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, t=None):
        # x: (B, T, C, F)
        # t: (B, T) ignored by NeuroEnsembleModel (it uses positional encoding)
        
        out_dict = self.base_model(x)
        y_pred = out_dict["y_pred"]   # (B, T, C, Q)
        dy_pred = out_dict["dy_pred"] # (B, T, C)

        # Output format expected by loss function: (B, T, C, Q+1)
        dy_pred = dy_pred.unsqueeze(-1)
        return torch.cat([y_pred, dy_pred], dim=-1)

# ============================================================
# Main Training Script
# ============================================================
import argparse

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
    input_size = num_channels
    
    print(f"Loading data for {dataset_name}...")
    try:
        raw = np.load(f'dataset/train_data_{dataset_name}.npz')['arr_0']
    except FileNotFoundError:
        # Fallback for running from parent dir
        raw = np.load(f'Neural_Forecasting/dataset/train_data_{dataset_name}.npz')['arr_0']
        
    num_total = len(raw)
    train_end = int(num_total * 0.8)
    val_end = int(num_total * 0.9)

    train_data = raw[:train_end]
    val_data = raw[train_end:val_end]
    test_data = raw[val_end:]

    print(f"Temporal split: train {len(train_data)}, val {len(val_data)}, test {len(test_data)}")

    # Use Raw Dataset
    train_ds = RawNeuroForcastDataset(train_data)
    val_ds = RawNeuroForcastDataset(val_data)
    test_ds = RawNeuroForcastDataset(test_data)

    train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize Model
    quantiles = [0.1, 0.5, 0.9]
    base_model = NeuroEnsembleModel(
        input_size=num_channels,
        num_features=9,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        quantiles=quantiles,
        history_steps=10
    ).to(torch.device('cuda'))

    model = NeuroEnsembleWrapper(base_model).to(torch.device('cuda'))

    # Loss Function (Same as v9)
    # Spike threshold calc
    f0 = train_data[..., 0]
    dy = f0[:, 1:, :] - f0[:, :-1, :]
    abs_dy = np.abs(dy.reshape(-1))
    tau = float(np.percentile(abs_dy, 95))
    print(f"Spike threshold tau: {tau:.6f}")

    spike_weight = 2.0
    lambda_change = 0.2

    change_loss_fn = torch.nn.L1Loss(reduction="none")

    def pinball_loss(y_hat, y, q):
        e = y - y_hat
        return torch.maximum(q * e, (q - 1.0) * e)

    def quantile_order_penalty(q10, q50, q90):
        return (F.relu(q10 - q50).mean() + F.relu(q50 - q90).mean())

    def spike_aware_loss(pred_tensor, target, init_steps=10):
        """
        pred_tensor: (B, H, C, Q+1)
        target:      (B, H, C)
        """
        Q = len(quantiles)
        y_pred = pred_tensor[..., :Q]
        dy_pred = pred_tensor[..., Q]
        y_true = target

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
        change_loss = (change_loss_fn(dy_pred, dy_true) * w).mean()

        lambda_quant = 1.0
        lambda_ord = 5.0

        total_loss = (
            lambda_quant * quant_loss +
            lambda_change * change_loss +
            lambda_ord * ord_loss
        )
        return total_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** (epoch // 20))
    
    save_path = f'model_{dataset_name}_v17.pth'
    
    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        val_data_loader=val_data_loader,
        loss_fn=lambda y_hat, y_true: spike_aware_loss(y_hat, y_true, init_steps=10),
        optimizer=optimizer,
        device=torch.device('cuda'),
        scheduler=scheduler,
        forecasting_mode='multi_step',
        init_steps=10,
        save_path=save_path,
        ckpt_path=None,
        max_grad_norm=1.0,
        validate_every=1,
        save_best=True
    )

    print("Starting training...")
    trainer.train(num_epochs)
    print("Best val:", trainer.best_val)
    print("Training Complete.")

if __name__ == "__main__":
    train()
