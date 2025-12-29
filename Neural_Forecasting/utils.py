from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import numpy as np
import torch

def compute_adj_matrix(data, threshold=0.1):
    """
    data: (N, T, C, F) 原始數據
    threshold: 門檻值，低於此相關性的邊設為 0
    """
    # 壓縮數據為 (N*T, C)，使用第 0 個特徵 (LFP) 計算通道間相關性
    c_data = data[:, :, :, 0].reshape(-1, data.shape[2]) 
    
    # 計算皮爾森相關矩陣 (C, C)
    adj = np.corrcoef(c_data.T)
    adj = np.nan_to_num(adj) # 處理常數訊號導致的 NaN
    
    # 只保留正相關且大於門檻值的邊
    adj[adj < threshold] = 0
    
    # 歸一化 (Degree Normalization)
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree!=0)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return torch.tensor(adj_normalized, dtype=torch.float32)

def load_dataset(filename,input_dir):
    """
    Load test dataset from file.

    Args:
        filename: Name of the test data file

    Returns:
        train_data: Samples to train with shape
                  (num_samples, num_timestep, num_channels, num_bands)
        test_data: Samples to test with shape
                  (num_samples, num_timestep, num_channels, num_bands)
        val_data: Samples to validate with shape
                  (num_samples, num_timestep, num_channels, num_bands)
    """
    test_file = os.path.join(input_dir, filename)
    # Open the file and load the data
    # split into train(80%), test(10%), val(10%)
    data = np.load(test_file)['arr_0']
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):int(len(data)*0.9)]
    val_data = data[int(len(data)*0.9):]

    return train_data, test_data, val_data


def normalize(data, average=[], std=[]):
      
    n, t, c, f = data.shape
    # 重新排列以便對每個通道(C)和特徵(F)單獨做標準化
    flat_data = data.reshape(n * t, c * f) 
    
    if len(average) == 0:
        average = np.mean(flat_data, axis=0, keepdims=True) # (1, C*F)
        std = np.std(flat_data, axis=0, keepdims=True) + 1e-6
        
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    
    # 進行標準化
    norm_data = 2 * (flat_data - combine_min) / (combine_max - combine_min + 1e-6) - 1
    norm_data = norm_data.reshape(n, t, c, f)
    return norm_data, average, std


class NeuroForcastDataset(Dataset):
  def __init__(self, neural_data, use_graph=False, average=[], std=[]):
    """
    neural_data: N*T*C*F (sampe size * total time steps * channel *feature dimension)
    f_window: T' the length of prediction window
    batch_size: batch size
    """
    self.data = neural_data
    self.use_graph = use_graph
    if len(average) == 0:
      self.data, self.average, self.std = normalize(self.data)
    else:
      self.data, self.average, self.std = normalize(self.data, average, std)

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index):
    data = self.data[index]
    # if not self.use_graph:
    #   data = data[:, :, 0]

    data = torch.tensor(data, dtype=torch.float32)
    return data