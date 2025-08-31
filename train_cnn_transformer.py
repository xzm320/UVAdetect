import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from cnn_transformer_model import MultiModalClassifier, get_data_loaders


def train(args):
    train_loader, val_loader = get_data_loaders(
        batch_size=128,  # 统一使用128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    model = MultiModalClassifier(
        num_classes=4,
        signal=dict(input_channels=32, seq_len=41),
        cwt=dict(in_channels=31),
    ).to(device)

    # 针对数据不平衡的加权损失函数
    # 标签分布: [1760, 870, 72, 480] - 标签2样本很少
    class_weights = torch.FloatTensor([1.0/1760, 1.0/870, 1.0/72, 1.0/480])
    class_weights = class_weights / class_weights.sum() * 4  # 归一化
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 使用AdamW优化器，更好的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        running_loss = 0.0
        for sig, img, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            sig = sig.to(device)
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sig, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sig, img, labels in tqdm(val_loader, desc=f"Val   Epoch {epoch}"):
                sig = sig.to(device)
                img = img.to(device)
                labels = labels.to(device)
                outputs = model(sig, img)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 更新学习率
        scheduler.step()
        
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # ---- 可视化 ----
    os.makedirs("fig", exist_ok=True)
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("fig", "loss_curve.png"), dpi=150)
    plt.close()
    print("[INFO] Loss 曲线已保存至 fig/loss_curve.png")

    # ---- 保存模型权重 ----
    torch.save(model.state_dict(), "cnn_transformer_checkpoint.pth")
    print("[INFO] 模型权重已保存至 cnn_transformer_checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-Transformer model on mini dataset")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")  
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    args = parser.parse_args()

    train(args) 