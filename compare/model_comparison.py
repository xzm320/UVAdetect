import os
import sys
import time
import gc
import psutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from compare.transformer.transformer_model import TransformerClassifier, SignalDataset
from compare.cnn.cnn_model import CNNClassifier, WaveletDataset
from cnn_transformer_model import MultiModalClassifier, JointDataset

"""
三模型对比工具
对比原始CNN-Transformer、单Transformer、单CNN三个模型的：
1. 正确率（准确率）- 主要对比指标

统一配置：batch_size=128, epochs=50
"""

class ModelComparator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def load_models(self):
        """加载三个模型"""
        models = {}
        
        # 1. CNN-Transformer模型
        try:
            cnn_transformer = MultiModalClassifier(
                num_classes=4,
                signal=dict(input_channels=32, seq_len=41),
                cwt=dict(in_channels=31),
            ).to(self.device)
            
            if os.path.exists("cnn_transformer_checkpoint.pth"):
                cnn_transformer.load_state_dict(torch.load("cnn_transformer_checkpoint.pth", map_location=self.device))
                print("[INFO] 成功加载CNN-Transformer模型")
            else:
                print("[WARNING] 未找到CNN-Transformer模型权重，使用随机初始化")
            
            models['CNN-Transformer'] = cnn_transformer
            
        except Exception as e:
            print(f"[ERROR] 加载CNN-Transformer模型失败: {e}")
        
        # 2. 单Transformer模型
        try:
            transformer = TransformerClassifier(
                input_channels=32,
                seq_len=41,
                embed_dim=256,      # 适中的参数量
                num_heads=16,       # 保持注意力头数
                num_layers=8,       # 保持层数
                ff_dim=512,         # 适中的前馈网络
                num_classes=4,
                dropout=0.1
            ).to(self.device)
            
            if os.path.exists("compare/transformer/best_transformer.pth"):
                checkpoint = torch.load("compare/transformer/best_transformer.pth", map_location=self.device)
                transformer.load_state_dict(checkpoint['model_state_dict'])
                print("[INFO] 成功加载Transformer模型")
            else:
                print("[WARNING] 未找到Transformer模型权重，使用随机初始化")
            
            models['Transformer'] = transformer
            
        except Exception as e:
            print(f"[ERROR] 加载Transformer模型失败: {e}")
        
        # 3. 单CNN模型
        try:
            cnn = CNNClassifier(
                in_channels=31,
                num_classes=4,
                dropout=0.3
            ).to(self.device)
            
            if os.path.exists("compare/cnn/best_cnn.pth"):
                checkpoint = torch.load("compare/cnn/best_cnn.pth", map_location=self.device)
                cnn.load_state_dict(checkpoint['model_state_dict'])
                print("[INFO] 成功加载CNN模型")
            else:
                print("[WARNING] 未找到CNN模型权重，使用随机初始化")
            
            models['CNN'] = cnn
            
        except Exception as e:
            print(f"[ERROR] 加载CNN模型失败: {e}")
        
        return models
    
    def get_model_params(self, model):
        """获取模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    

    
    def evaluate_accuracy(self, model, data_loader, model_name):
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {model_name} Accuracy"):
                if model_name == 'CNN-Transformer':
                    sig, img, labels = batch
                    sig, img, labels = sig.to(self.device), img.to(self.device), labels.to(self.device)
                    outputs = model(sig, img)
                elif model_name == 'Transformer':
                    signals, labels = batch
                    signals, labels = signals.to(self.device), labels.to(self.device)
                    outputs = model(signals)
                else:  # CNN
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # 计算每个类别的准确率
        class_accuracies = {}
        class_names = ['正常', '螺旋桨故障', '低电压故障', '电机故障']
        
        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_labels) == i
            if np.sum(class_mask) > 0:
                class_correct = np.sum((np.array(all_predictions)[class_mask] == np.array(all_labels)[class_mask]))
                class_total = np.sum(class_mask)
                class_acc = 100. * class_correct / class_total
                class_accuracies[class_name] = class_acc
            else:
                class_accuracies[class_name] = 0.0
        
        return {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total,
            'correct_samples': correct,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def prepare_data_loaders(self, batch_size=64):
        """准备数据加载器"""
        data_loaders = {}
        
        try:
            # CNN-Transformer数据（多模态）
            joint_dataset = JointDataset("ipt/dataset.npz", "ipt/wavelet_dataset.npz")
            n = len(joint_dataset)
            indices = np.random.permutation(n)
            test_indices = indices[:min(1000, n)]  # 使用1000个样本进行测试
            joint_test = torch.utils.data.Subset(joint_dataset, test_indices)
            data_loaders['CNN-Transformer'] = torch.utils.data.DataLoader(
                joint_test, batch_size=batch_size, shuffle=False
            )
            
            # Transformer数据（时序信号）
            signal_dataset = SignalDataset("ipt/dataset.npz")
            signal_test = torch.utils.data.Subset(signal_dataset, test_indices)
            data_loaders['Transformer'] = torch.utils.data.DataLoader(
                signal_test, batch_size=batch_size, shuffle=False
            )
            
            # CNN数据（小波图像）
            wavelet_dataset = WaveletDataset("ipt/wavelet_dataset.npz")
            wavelet_test = torch.utils.data.Subset(wavelet_dataset, test_indices)
            data_loaders['CNN'] = torch.utils.data.DataLoader(
                wavelet_test, batch_size=batch_size, shuffle=False
            )
            
            print(f"[INFO] 准备了 {len(test_indices)} 个测试样本")
            
        except Exception as e:
            print(f"[ERROR] 准备数据加载器失败: {e}")
            return None
        
        return data_loaders
    
    def compare_models(self, batch_size=128):
        """对比三个模型的准确率"""
        print("开始三模型准确率对比...")
        print(f"配置: batch_size={batch_size}, epochs=50")
        
        # 加载模型
        models = self.load_models()
        if not models:
            print("[ERROR] 没有成功加载任何模型")
            return
        
        # 准备数据
        data_loaders = self.prepare_data_loaders(batch_size)
        if not data_loaders:
            print("[ERROR] 数据加载器准备失败")
            return
        
        print(f"\n=== 模型参数量和准确率对比 ===")
        print(f"{'模型':<20} {'参数量(M)':<15} {'准确率(%)':<12}")
        print("-" * 50)
        
        # 对比每个模型
        for model_name, model in models.items():
            print(f"\n正在评估 {model_name}...")
            
            # 参数量统计
            total_params = self.get_model_params(model)
            print(f"{model_name:<20} {total_params/1e6:<14.2f} ", end="")
            
            # 准确率评估
            if model_name in data_loaders:
                accuracy_info = self.evaluate_accuracy(
                    model, data_loaders[model_name], model_name
                )
                
                # 保存结果
                self.results[model_name] = {
                    'params': total_params,
                    'accuracy': accuracy_info
                }
                
                print(f"{accuracy_info['overall_accuracy']:<11.2f}")
                
                # 显示各类别准确率
                print(f"  各类别准确率:")
                for class_name, class_acc in accuracy_info['class_accuracies'].items():
                    print(f"    {class_name}: {class_acc:.2f}%")
            else:
                print("数据加载器缺失")
        
        # 生成简化的对比报告
        self.generate_simplified_report()
        self.plot_accuracy_comparison()
    
    def generate_simplified_report(self):
        """生成简化的准确率对比报告"""
        if not self.results:
            print("[ERROR] 没有结果数据")
            return
        
        print("\n" + "="*60)
        print("                三模型准确率对比报告")
        print("="*60)
        
        # 简化对比表
        print(f"\n{'模型':<20} {'参数量(M)':<15} {'准确率(%)':<12}")
        print("-" * 50)
        
        for model_name, results in self.results.items():
            params_m = results['params'] / 1e6
            accuracy = results['accuracy']['overall_accuracy']
            print(f"{model_name:<20} {params_m:<14.2f} {accuracy:<11.2f}")
        
        # 准确率排名
        accuracy_ranking = sorted(self.results.items(), 
                                key=lambda x: x[1]['accuracy']['overall_accuracy'], 
                                reverse=True)
        print(f"\n{'='*60}")
        print("准确率排名:")
        print(f"{'='*60}")
        for i, (model_name, results) in enumerate(accuracy_ranking, 1):
            accuracy = results['accuracy']['overall_accuracy']
            print(f"{i}. {model_name}: {accuracy:.2f}%")
        
        # 各类别准确率对比
        print(f"\n{'='*60}")
        print("各类别准确率对比:")
        print(f"{'='*60}")
        class_names = ['正常', '螺旋桨故障', '低电压故障', '电机故障']
        print(f"{'模型':<20} ", end="")
        for class_name in class_names:
            print(f"{class_name:<12} ", end="")
        print()
        print("-" * (20 + 12 * len(class_names)))
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} ", end="")
            for class_name in class_names:
                class_acc = results['accuracy']['class_accuracies'][class_name]
                print(f"{class_acc:<11.2f}% ", end="")
            print()
        
        # 保存简化报告
        self.save_simplified_report()
    
    def save_simplified_report(self):
        """保存简化的准确率报告"""
        os.makedirs("compare/results", exist_ok=True)
        
        with open("compare/results/accuracy_comparison_report.txt", "w", encoding="utf-8") as f:
            f.write("三模型准确率对比报告\n")
            f.write("="*40 + "\n\n")
            
            f.write("统一配置:\n")
            f.write("- Epochs: 50\n")
            f.write("- Batch Size: 128\n")
            f.write("- 优化器: AdamW\n")
            f.write("- 学习率调度: CosineAnnealingLR\n\n")
            
            f.write("准确率对比:\n")
            f.write(f"{'模型':<20} {'参数量(M)':<15} {'准确率(%)':<12}\n")
            f.write("-" * 50 + "\n")
            
            for model_name, results in self.results.items():
                params_m = results['params'] / 1e6
                accuracy = results['accuracy']['overall_accuracy']
                f.write(f"{model_name:<20} {params_m:<14.2f} {accuracy:<11.2f}\n")
            
            f.write(f"\n各类别详细准确率:\n")
            class_names = ['正常', '螺旋桨故障', '低电压故障', '电机故障']
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}:\n")
                for class_name, class_acc in results['accuracy']['class_accuracies'].items():
                    f.write(f"  {class_name}: {class_acc:.2f}%\n")
        
        print(f"\n[INFO] 简化报告已保存至 compare/results/accuracy_comparison_report.txt")
    
    def plot_accuracy_comparison(self):
        """绘制准确率对比图表"""
        if not self.results:
            return
        
        os.makedirs("compare/results", exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        models = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 创建准确率对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 整体准确率对比
        accuracies = [self.results[model]['accuracy']['overall_accuracy'] for model in models]
        bars1 = ax1.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        ax1.set_title('三模型整体准确率对比', fontweight='bold', fontsize=16)
        ax1.set_ylabel('准确率 (%)', fontsize=12)
        ax1.tick_params(axis='x', rotation=15)
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 各类别准确率雷达图
        class_names = ['正常', '螺旋桨故障', '低电压故障', '电机故障']
        angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合圆圈
        
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        
        for i, model_name in enumerate(models):
            class_accs = [self.results[model_name]['accuracy']['class_accuracies'][class_name] 
                         for class_name in class_names]
            class_accs += class_accs[:1]  # 闭合圆圈
            
            ax2.plot(angles, class_accs, 'o-', linewidth=3, label=model_name, 
                    color=colors[i], markersize=8)
            ax2.fill(angles, class_accs, alpha=0.25, color=colors[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(class_names, fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_title('各类别准确率对比', fontweight='bold', fontsize=16, pad=30)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('compare/results/accuracy_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[INFO] 准确率对比图表已保存至 compare/results/accuracy_comparison_charts.png")


def main():
    """主函数"""
    print("=== 三模型准确率对比工具 ===")
    print("对比模型: CNN-Transformer, Transformer, CNN")
    print("主要指标: 准确率对比")
    print("统一配置: batch_size=128, epochs=50\n")
    
    comparator = ModelComparator()
    comparator.compare_models(batch_size=128)
    
    print("\n=== 对比完成 ===")
    print("结果文件:")
    print("- compare/results/accuracy_comparison_report.txt")
    print("- compare/results/accuracy_comparison_charts.png")


if __name__ == "__main__":
    main()
