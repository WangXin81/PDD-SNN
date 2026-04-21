import torch
from torch import nn
import torch.nn.functional as F  # [新增] 用于动态缩放矩阵
import numpy as np

backend = 'torch'
import torch._dynamo as _dynamo

_dynamo.config.suppress_errors = True

def normalize_haar_matrix(H, device):
    # Compute norms of rows
    norms = torch.linalg.norm(H, axis=1, keepdims=True)
    # Divide each row by its norm
    H = H / norms
    return H.to(device).to(torch.float32)

def haar_matrix(N, device):
    H = torch.tensor(haar_1d_matrix(N)).to(torch.float32)
    return normalize_haar_matrix(H, device)

def haar_1d_matrix(n):
    # This function generates an nxn Haar matrix
    # n must be a power of 2
    if np.log2(n) % 1 > 0:
        raise ValueError("n must be a power of 2")

    if n == 1:
        return np.array([[1]])
    else:
        H_next = haar_1d_matrix(n // 2)
        upper = np.kron(H_next, [1, 1])
        lower = np.kron(np.eye(len(H_next)), [1, -1])
        H = np.vstack((upper, lower))
        return H

class Haar1DForward(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_neuron = neuron_type(v_threshold=vth, v_reset=None)
        self.register_buffer('H', None)

    def build(self, N, device):
        matrix = haar_matrix(N, device)
        self.register_buffer('H', matrix)

    def _get_adapted_matrix(self, target_size):
        """[新增] 自动适配输入维度的辅助方法"""
        if self.H is None: return None
        if self.H.shape[0] == target_size:
            return self.H
        
        # 动态插值适配：[1, 1, N, N] -> [1, 1, target, target]
        h_tmp = self.H.unsqueeze(0).unsqueeze(0)
        h_resized = F.interpolate(h_tmp, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
        return h_resized.squeeze(0).squeeze(0)

    def haar_1d(self, x):
        # x.shape 最后一维即为运算维度
        H_adapted = self._get_adapted_matrix(x.shape[-1])
        return torch.matmul(H_adapted, x)

    def forward(self, x):
        haar = self.haar_1d(x)
        return self.haar_neuron(haar)

class Haar1DInverse(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_inv_neu = neuron_type(v_threshold=vth, v_reset=None)
        self.register_buffer('H', None)

    def build(self, N, device):
        matrix = haar_matrix(N, device)
        self.register_buffer('H', matrix)

    def _get_adapted_matrix(self, target_size):
        if self.H is None: return None
        if self.H.shape[0] == target_size:
            return self.H
        h_tmp = self.H.unsqueeze(0).unsqueeze(0)
        h_resized = F.interpolate(h_tmp, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
        return h_resized.squeeze(0).squeeze(0)

    def haar_1d_inverse(self, x):
        H_adapted = self._get_adapted_matrix(x.shape[-1])
        return torch.matmul(H_adapted.T, x)

    def forward(self, x):
        haar_inverse = self.haar_1d_inverse(x)
        return self.haar_inv_neu(haar_inverse)

class Haar2DForward(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.row_haar_neuron = neuron_type(v_threshold=vth, backend=backend, v_reset=None)
        self.col_haar_neuron = neuron_type(v_threshold=vth, backend=backend, v_reset=None)
        self.register_buffer('H', None)

    def build(self, N, device):
        matrix = haar_matrix(N, device)
        self.register_buffer('H', matrix)

    def _get_adapted_matrix(self, target_size):
        if self.H is None: return None
        if self.H.shape[0] == target_size:
            return self.H
        h_tmp = self.H.unsqueeze(0).unsqueeze(0)
        h_resized = F.interpolate(h_tmp, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
        return h_resized.squeeze(0).squeeze(0)

    def forward(self, x):
        # x shape: [T, B, C, H, W]
        h_dim = x.shape[-2]
        w_dim = x.shape[-1]

        # Apply to each row (Width dimension)
        x = self.row_haar_neuron(x)
        H_w = self._get_adapted_matrix(w_dim)
        x = torch.matmul(x, H_w.T)

        # Apply to each column (Height dimension)
        x = self.col_haar_neuron(x)
        H_h = self._get_adapted_matrix(h_dim)
        x = torch.matmul(H_h, x)
        return x

class Haar2DInverse(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.row_haar_neuron = neuron_type(v_threshold=vth, v_reset=None)
        self.col_haar_neuron = neuron_type(v_threshold=vth, v_reset=None)
        self.register_buffer('H', None)

    def build(self, N, device):
        matrix = haar_matrix(N, device)
        self.register_buffer('H', matrix)

    def _get_adapted_matrix(self, target_size):
        if self.H is None: return None
        if self.H.shape[0] == target_size:
            return self.H
        h_tmp = self.H.unsqueeze(0).unsqueeze(0)
        h_resized = F.interpolate(h_tmp, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
        return h_resized.squeeze(0).squeeze(0)

    def forward(self, x):
        h_dim = x.shape[-2]
        w_dim = x.shape[-1]

        # Inverse Rows
        x = self.row_haar_neuron(x)
        H_w = self._get_adapted_matrix(w_dim)
        x = torch.matmul(x, H_w)

        # Inverse Columns
        x = self.col_haar_neuron(x)
        H_h = self._get_adapted_matrix(h_dim)
        x = torch.matmul(H_h.T, x)
        return x