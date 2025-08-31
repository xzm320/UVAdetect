import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compare.cnn.cnn_model import CNNClassifier, get_data_loaders

def train_cnn(args):
    """训练CNN模型"""
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")
    
    # 数据加载器
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        split_ratio=args.split_ratio
    )
    
    print(f"[INFO] 训练样本数: {len(train_loader.dataset)}")
    print(f"[INFO] 验证样本数: {len(val_loader.dataset)}")
    
    # 模型
    model = CNNClassifier(
        in_channels=31,
        num_classes=4,
        dropout=args.dropout
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 损失函数 - 处理数据不平衡
    class_weights = torch.FloatTensor([1.0/1760, 1.0/870, 1.0/72, 1.0/480])
    class_weights = class_weights / class_weights.sum() * 4
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = "compare/cnn/best_cnn.pth"
    
    print(f"[INFO] 开始训练 {args.epochs} 个epoch...")
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
        
        # 打印epoch结果
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n[INFO] 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存训练曲线
    save_training_curves(train_losses, val_losses, train_accs, val_accs, args.epochs)
    
    # 最终评估
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    final_evaluation(model, val_loader, device)
    
    return model, best_val_acc


def save_training_curves(train_losses, val_losses, train_accs, val_accs, epochs):
    """保存训练曲线 - 统一风格与CNN-Transformer一致"""
    os.makedirs("compare/cnn/results", exist_ok=True)
    
    # 只绘制损失曲线，与CNN-Transformer风格一致
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('compare/cnn/results/loss_curve.png', dpi=150)
    plt.close()
    
    print("[INFO] Loss曲线已保存至 compare/cnn/results/loss_curve.png")


def final_evaluation(model, val_loader, device):
    """最终评估"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Final Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 分类报告
    class_names = ['正常', '螺旋桨故障', '低电压故障', '电机故障']
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    
    print("\n=== CNN 分类报告 ===")
    print(report)
    
    # 保存报告
    os.makedirs("compare/cnn/results", exist_ok=True)
    with open("compare/cnn/results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("CNN模型分类报告\n")
        f.write("="*40 + "\n\n")
        f.write(report)
    
    print("[INFO] 分类报告已保存至 compare/cnn/results/classification_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练CNN故障分类模型")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout比例")
    
    # 数据参数
    parser.add_argument("--split_ratio", type=float, default=0.8, help="训练集比例")
    
    args = parser.parse_args()
    
    print("=== CNN模型训练配置 ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30)
    
    # 开始训练
    model, best_acc = train_cnn(args)
    print(f"\n[FINAL] CNN训练完成，最佳验证准确率: {best_acc:.2f}%")
