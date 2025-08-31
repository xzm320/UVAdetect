import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
纯CNN模型用于故障分类
处理小波变换后的时频图数据 (N, 31, 41, 41)
"""

class WaveletDataset(Dataset):
    """小波变换图像数据集"""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # (N, C, H, W)
        self.y = data["y"].astype(np.int64)
        assert self.X.ndim == 4, "X应为四维数组 (N, C, H, W)"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNClassifier(nn.Module):
    """纯CNN故障分类器"""
    
    def __init__(
        self,
        in_channels: int = 31,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 41x41 -> 20x20
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 20x20 -> 10x10
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 10x10 -> 5x5
            
            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, channels, height, width)
        return: (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def get_data_loaders(
    data_path: str = "ipt/wavelet_dataset.npz",
    batch_size: int = 64,
    split_ratio: float = 0.8,
    num_workers: int = 2
):
    """获取数据加载器"""
    dataset = WaveletDataset(data_path)
    
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
    model = CNNClassifier()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CNN模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 测试前向传播
    x = torch.randn(2, 31, 41, 41)  # (batch_size, channels, height, width)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
