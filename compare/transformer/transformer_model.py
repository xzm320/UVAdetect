import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
纯Transformer模型用于故障分类
处理时序信号数据 (N, 41, 32)
"""

class SignalDataset(Dataset):
    """时序信号数据集"""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # (N, T, C)
        self.y = data["y"].astype(np.int64)
        assert self.X.ndim == 3, "X应为三维数组 (N, T, C)"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """纯Transformer故障分类器"""
    
    def __init__(
        self,
        input_channels: int = 32,
        seq_len: int = 41,
        embed_dim: int = 256,        # 适中的嵌入维度
        num_heads: int = 16,         # 保持注意力头数
        num_layers: int = 8,         # 保持层数
        ff_dim: int = 512,           # 适中的前馈网络维度
        num_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_channels, embed_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头 - 增加更多层和参数
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),        # 512 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),   # 512 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),  # 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)     # 128 -> 4
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_channels)
        return: (batch_size, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)  # (B, L, embed_dim)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)  # (B, L, embed_dim)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # 分类
        x = self.classifier(x)
        return x


def get_data_loaders(
    data_path: str = "ipt/dataset.npz",
    batch_size: int = 64,
    split_ratio: float = 0.8,
    num_workers: int = 2
):
    """获取数据加载器"""
    dataset = SignalDataset(data_path)
    
    # 数据分割
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * split_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试模型
    model = TransformerClassifier()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Transformer模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 测试前向传播
    x = torch.randn(2, 41, 32)  # (batch_size, seq_len, channels)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
