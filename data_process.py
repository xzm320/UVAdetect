import os
import re
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- 列索引常量 ----------
# 0-based 索引
FEATURE_COLS = list(range(2, 11)) + list(range(19, 32)) + list(range(35, 45))  # C-K, T-AF, AJ-AS
AU_IDX = 46  # AU 列
READ_COLS = sorted(set(FEATURE_COLS + [AU_IDX]))

# 直接用 0 基索引以 range 指定要读取的列
# C-K  -> 2-10 (含)
# T-AF -> 19-31 (含)
# AJ-AS-> 35-44 (含)
USECOLS = (
    list(range(2, 11)) +
    list(range(19, 32)) +
    list(range(35, 45))
)

# 故障类型映射
FAULT_CODE_TO_LABEL = {
    "00": 0,
    "01": 1,
    "02": 2,
    "10": 3,
}

# 正则用于提取文件名中的数字部分
DIGIT_RE = re.compile(r"(\d+)")


def extract_label_from_filename(filename: str) -> Tuple[bool, int]:
    """从文件名中提取故障代码并返回 (是否有效, label)。"""
    # 取文件名里的第一段数字
    m = DIGIT_RE.search(filename)
    if not m:
        return False, -1
    digits = m.group(1)
    if len(digits) < 4:
        return False, -1
    code = digits[2:4]  # 第三、第四位数字(0 基)
    if code in FAULT_CODE_TO_LABEL:
        return True, FAULT_CODE_TO_LABEL[code]
    return False, -1


def main():
    data_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    csv_paths = glob.glob(os.path.join("raw_data", "*.csv"))
    if not csv_paths:
        print("未找到任何 CSV 文件，请确认文件路径是否正确。")
        return

    for path in csv_paths:
        valid, label = extract_label_from_filename(os.path.basename(path))
        if not valid:
            continue  # 跳过不在指定故障代码内的文件

        try:
            df = pd.read_csv(
                path,
                usecols=READ_COLS,
            )

            # ---------- 基于 AU 列提取窗口 ----------
            au_series = df.iloc[:, READ_COLS.index(AU_IDX)].fillna(0)
            au_values = pd.to_numeric(au_series, errors='coerce').fillna(0).astype(int).values

            # 找到首次 0->1 跃迁位置
            trans_idxs = np.flatnonzero((au_values[:-1] == 0) & (au_values[1:] == 1))
            if trans_idxs.size > 0:
                center = int(trans_idxs[0] + 1)
            else:
                # 若 AU 全 1，无 0，则取文件中间
                center = len(df) // 2

            start = max(center - 20, 0)
            end = min(center + 20, len(df) - 1)
            if end - start + 1 < 41:
                # 若不足41行，通过两端补齐
                if start == 0:
                    end = min(40, len(df) - 1)
                else:
                    start = max(len(df) - 41, 0)

            df_win = df.iloc[start:end + 1].reset_index(drop=True)

            # 仅保留特征列
            feat_cols = [df.columns[READ_COLS.index(idx)] for idx in FEATURE_COLS]
            df_feat = df_win[feat_cols].copy()

            # 定义字符串转浮点的辅助函数
            def _str_to_float(val):
                if isinstance(val, str):
                    fragment = val.split(',')[0].strip()
                    try:
                        return float(fragment)
                    except ValueError:
                        return np.nan
                return val

            obj_cols = df_feat.select_dtypes(include="object").columns
            for col in obj_cols:
                converted = df_feat[col].map(_str_to_float)
                success_ratio = converted.notna().mean()
                if success_ratio == 1.0:
                    print(f"[转换成功] 文件 {os.path.basename(path)} 列 {col} 全部转换为数字")
                elif success_ratio > 0:
                    print(
                        f"[部分转换] 文件 {os.path.basename(path)} 列 {col} 转换率 {success_ratio:.0%}，剩余 NaN 将用均值填充"
                    )
                else:
                    print(f"[转换失败] 文件 {os.path.basename(path)} 列 {col} 无法转换为数字，整列 NaN")
                df_feat[col] = converted

            # 将所有列转换为数值，无法解析的保持 NaN
            df_feat = df_feat.apply(pd.to_numeric, errors='coerce')
            # 以列均值填充 NaN
            df_feat = df_feat.fillna(df_feat.mean())
            df_feat = df_feat.astype(np.float32, errors='ignore')
        except Exception as e:
            print(f"读取文件 {path} 失败: {e}")
            continue

        if len(df_feat) == 0:
            continue

        # 删除全为 NaN 的列
        df_feat = df_feat.dropna(axis=1, how="all")

        # 保留 41×feature 的窗口，不再把 41 行拍平成单行
        values = df_feat.to_numpy(dtype=np.float32)  # (41, feature_dim)

        data_chunks.append(values)         # 追加到窗口列表
        label_chunks.append(label)         # 每个窗口只对应一个标签

        print(f"[完成] 已处理文件 {os.path.basename(path)}，样本数 {values.shape[0]}")

    if not data_chunks:
        print("没有符合条件的数据被读取，检查故障代码或文件格式。")
        return

    # 以 (num_windows, 41, feature_dim) 形式堆叠
    X = np.stack(data_chunks, axis=0).astype(np.float32)  # (N, 41, F)
    y = np.array(label_chunks, dtype=np.int64)            # (N,)

    # -------- 标准化 --------
    # 按特征维度计算均值/方差，时间与样本维度全部展开
    flat = X.reshape(-1, X.shape[-1])                     # (N*41, F)
    mean = flat.mean(axis=0, keepdims=True)               # (1, F)
    std = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0                                   # 防止除零

    # 重新 reshape 成可广播形状 (1,1,F)
    mean_3d = mean.reshape(1, 1, -1)
    std_3d = std.reshape(1, 1, -1)
    X_std = (X - mean_3d) / std_3d

    # 保存一维形式的 mean/std 便于后续反标准化
    mean = mean.squeeze()
    std = std.squeeze()

    os.makedirs("ipt", exist_ok=True)
    np.save(os.path.join("ipt", "X.npy"), X_std)
    np.save(os.path.join("ipt", "y.npy"), y)
    # 也保存压缩 npz
    np.savez_compressed(
        os.path.join("ipt", "dataset.npz"),
        X=X_std,
        y=y,
        mean=mean,
        std=std,
    )

    print(
        f"数据转换完成，共生成 {X.shape[0]} 个样本窗口，窗口长度 {X.shape[1]}，"
        f"特征维度 {X.shape[2]}，已保存至 ipt 目录。"
    )


if __name__ == "__main__":
    main()
