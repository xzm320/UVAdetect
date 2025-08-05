import os
from typing import Tuple

import numpy as np
import pywt

"""wavelet_transform.py

将 ipt/dataset.npz 中的三维数组 X (N, T, F) 转换为小波时频图：
1. 忽略首列时间 (F0)，对其余 F-1 个传感器通道做连续小波变换(CWT)。
2. 结果为 (F-1, S, T) 的二维矩阵，其中 S == T == 时间步长 (41)。
3. 输出标准化后的小波数据集保存到 ipt/wavelet_dataset.npz。

依赖：pywt
安装：pip install PyWavelets
"""

def ensure_dirs() -> str:
    """确保 ipt 目录存在，返回其路径"""
    data_dir = os.path.join("ipt")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_mini_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """加载 ipt/dataset.npz 并返回 (X, y)。

    X: (N, T, F) 三维数组；首列为时间，不参与后续小波变换。
    y: (N,) 标签数组。
    """
    mini_path = os.path.join("ipt", "dataset.npz")
    if not os.path.isfile(mini_path):
        raise FileNotFoundError(f"未找到 {mini_path}，请先生成 ipt 数据集")

    data = np.load(mini_path)
    if "X" not in data or "y" not in data:
        raise KeyError("dataset.npz 中需同时包含键 'X' 与 'y'")

    X = data["X"]
    y = data["y"]
    if X.ndim != 3:
        raise ValueError(
            "期望 X 形状为 (N, T, F) 三维数组；如之前生成的是二维，请重新生成三维格式"
        )
    if y.ndim != 1:
        raise ValueError("y 需为一维标签数组")
    return X.astype(np.float32), y.astype(np.int64)


def cwt_single_channel(signal: np.ndarray, wavelet: str = "morl") -> np.ndarray:
    """对单通道序列做 CWT，返回 |系数| (S, T)。"""
    T = signal.shape[0]
    scales = np.arange(1, T + 1)
    coeffs, _ = pywt.cwt(signal, scales, wavelet)
    # 取绝对值表示功率幅度
    return np.abs(coeffs).astype(np.float32)


# 图片保存功能已移除


def main():
    print("[INFO] 开始加载完整数据集 …")
    X_raw, y = load_mini_dataset()  # X_raw: (N, T, F)
    N, T, F = X_raw.shape
    print(f"[INFO] 数据形状: N={N}, T={T}, F={F}")

    if F <= 1:
        raise ValueError("特征维度需 >1，首列为时间，其余为传感器信号")

    C = F - 1  # 有效传感器通道数

    data_dir = ensure_dirs()

    cwt_chunks = []  # 用于最终堆叠 (N,C,T,T)

    print(f"[INFO] 开始处理 {N} 个样本的小波变换...")
    for i in range(N):
        sample = X_raw[i]            # (T, F)
        signals = sample[:, 1:]      # 跳过时间列 (T,C)

        # 对每个通道做 CWT
        cwt_imgs = np.empty((C, T, T), dtype=np.float32)
        for ch in range(C):
            cwt_imgs[ch] = cwt_single_channel(signals[:, ch])

        cwt_chunks.append(cwt_imgs)

        if (i + 1) % 100 == 0 or i == N - 1:
            print(f"[INFO] 样本 {i + 1}/{N} 完成 ({100*(i+1)/N:.1f}%)")

    # -------- 整合并标准化 --------
    print("[INFO] 开始数据标准化...")
    X_cwt = np.stack(cwt_chunks, axis=0)  # (N, C, T, T)

    # 计算通道级 mean/std
    flat = X_cwt.transpose(1, 0, 2, 3).reshape(C, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    std[std == 0] = 1.0

    mean_b = mean.reshape(1, C, 1, 1)
    std_b = std.reshape(1, C, 1, 1)
    X_std = (X_cwt - mean_b) / std_b

    # 保存统一数据集到 ipt 目录
    output_npz = os.path.join("ipt", "wavelet_dataset.npz")
    print("[INFO] 正在保存小波数据集...")
    np.savez_compressed(
        output_npz,
        X=X_std.astype(np.float32),
        y=y,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    print(
        f"[DONE] 已保存小波数据集 {output_npz} → 形状 {X_std.shape}，标签 {y.shape[0]} 条"
    )


if __name__ == "__main__":
    main() 