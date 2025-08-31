import math
import os
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

"""cnn_transformer_model.py

该文件实现一个多模态分类网络：
1. SignalTransformer     —— 处理 (N, T, C_sig) 的多通道时序信号 (mini 数据集)。
2. CWT_CNN_Transformer   —— 处理 (N, C_img, H, W) 的时频图 (wavelet_dataset)。
3. MultiModalClassifier  —— 将两路特征拼接后进行分类。

使用 PyTorch >=1.10。
"""

# ----------------------------- 数据集 -----------------------------
class MiniSignalDataset(Dataset):
    """加载 mini/dataset.npz (三维信号)。返回 (signal, label)。"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # (N,T,C)
        self.y = data["y"].astype(np.int64)
        assert self.X.ndim == 3, "mini X 应为 (N,T,C)"
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WaveletDataset(Dataset):
    """加载 fig/wavelet_dataset.npz，返回 (cwt_img, label)。"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # (N,C,H,W)
        self.y = data["y"].astype(np.int64)
        assert self.X.ndim == 4, "wavelet X 应为 (N,C,H,W)"
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class JointDataset(Dataset):
    """把 MiniSignalDataset 和 WaveletDataset 按索引对齐组合。"""

    def __init__(self, mini_npz: str, wavelet_npz: str):
        self.ds_sig = MiniSignalDataset(mini_npz)
        self.ds_img = WaveletDataset(wavelet_npz)
        assert len(self.ds_sig) == len(self.ds_img)

    def __len__(self):
        return len(self.ds_sig)

    def __getitem__(self, idx):
        sig, y1 = self.ds_sig[idx]
        img, y2 = self.ds_img[idx]
        assert y1 == y2
        return sig, img, y1


# -------------------------- 模型部件 --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class SignalTransformer(nn.Module):
    """处理多通道时序信号的 Transformer 编码器。"""

    def __init__(
        self,
        input_channels: int = 32,
        seq_len: int = 41,
        embed_dim: int = 64,        # 增大嵌入维度，适合小数据集
        num_heads: int = 8,         # 增加注意力头数
        ff_dim: int = 128,          # 增大前馈网络维度
        num_layers: int = 3,        # 保持适中层数，避免过拟合
        dropout: float = 0.3,       # 增大dropout，防止过拟合
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_channels, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, input_channels)"""
        x = self.input_proj(x)  # (B, L, embed_dim)
        x = self.pos_enc(x)
        x = self.encoder(x)     # (B, L, embed_dim)
        return x


class CWT_CNN_Transformer(nn.Module):
    """先用 CNN 提取时频图特征，再用 Transformer 编码。"""

    def __init__(
        self,
        in_channels: int = 31,
        cnn_channels: List[int] | Tuple[int, ...] = (64, 128, 256, 512),  # 增大CNN通道数，与单CNN一致
        embed_dim: int = 96,        # 适中的嵌入维度
        num_heads: int = 8,         # 增加注意力头数
        ff_dim: int = 192,          # 增大前馈网络维度
        num_layers: int = 3,        # 保持适中层数
    ):
        super().__init__()
        layers = []
        c_prev = in_channels
        h_reduction = 1
        
        # 构建与单CNN模型相似的架构
        for i, c_out in enumerate(cnn_channels):
            # 每个块包含双卷积层
            layers += [
                nn.Conv2d(c_prev, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            
            # 前三个块有池化，最后一个块不池化
            if i < len(cnn_channels) - 1:
                layers += [nn.MaxPool2d(2, 2)]
                h_reduction *= 2
            
            c_prev = c_out
        
        self.cnn = nn.Sequential(*layers)

        # 假设输入 H=W=41，经过两次池化变为 41//4≈10
        self.out_h = math.ceil(41 / h_reduction)
        self.out_w = self.out_h
        self.cnn_out_dim = c_prev

        # 将 CNN 输出展平成序列 (B, L, C)
        self.input_proj = nn.Linear(self.cnn_out_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=self.out_h * self.out_w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W)"""
        x = self.cnn(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, L, C)
        x = self.input_proj(x)  # (B, L, embed_dim)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, L, embed_dim)
        return x


class MultiModalClassifier(nn.Module):
    """整合两路 Transformer 特征并分类。"""

    def __init__(self, num_classes: int = 4, **kwargs):
        super().__init__()
        self.signal_backbone = SignalTransformer(**kwargs.get("signal", {}))
        self.cwt_backbone = CWT_CNN_Transformer(**kwargs.get("cwt", {}))

        total_embed = self.signal_backbone.embed_dim + self.cwt_backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_embed, 256),    # 增大隐藏层，但用更强的正则化
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                # 增大dropout
            nn.Linear(256, 128),            # 添加中间层
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                # 再次dropout
            nn.Linear(128, num_classes),
        )

    def global_pool(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """x: (B, L, D) -> (B, D)"""
        if mode == "mean":
            return x.mean(dim=1)
        else:
            return x[:, 0]  # 取 CLS token 等

    def forward(self, sig: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # sig: (B, T, C_sig)
        # img: (B, C_img, H, W)
        feat_sig = self.signal_backbone(sig)          # (B, L1, D1)
        feat_img = self.cwt_backbone(img)             # (B, L2, D2)

        pooled_sig = self.global_pool(feat_sig)
        pooled_img = self.global_pool(feat_img)

        fused = torch.cat([pooled_sig, pooled_img], dim=1)  # (B, D1+D2)
        logits = self.classifier(fused)
        return logits


# ------------------------- 训练脚本示例 -------------------------

def get_data_loaders(
    mini_npz: str = os.path.join("ipt", "dataset.npz"),
    wavelet_npz: str = os.path.join("ipt", "wavelet_dataset.npz"),
    batch_size: int = 32,
    split: float = 0.8,
):
    ds = JointDataset(mini_npz, wavelet_npz)
    n = len(ds)
    idx = np.random.permutation(n)
    split_idx = int(n * split)
    train_idx, val_idx = idx[:split_idx], idx[split_idx:]
    train_set = torch.utils.data.Subset(ds, train_idx)
    val_set = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


if __name__ == "__main__":
    # 示例：只构造网络并打印参数量
    model = MultiModalClassifier(
        num_classes=4,
        signal=dict(input_channels=32, seq_len=41),
        cwt=dict(in_channels=31),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total trainable params: {total_params / 1e6:.2f} M") 