# 基于CNN-Transformer的多模态无人机故障检测系统

## 项目简介

本项目实现了一个基于深度学习的多模态无人机故障检测系统，使用CNN-Transformer混合架构对无人机飞行数据进行故障分类。系统能够同时处理多通道时序信号和小波变换生成的时频图，通过多模态特征融合实现高精度的故障检测。

## 数据集介绍

本项目使用的数据来源于[RflyMAD数据集](https://rfly-openha.github.io/documents/4_resources/dataset.html)，这是北航可靠飞行控制研究组（Rfly）基于RflySim平台和真实飞行收集的专业无人机故障检测数据集。RflyMAD数据集包含5629个飞行案例，涵盖软件在环仿真(SIL)、硬件在环仿真(HIL)和真实飞行数据，是目前规模最大、最全面的无人机故障检测数据集之一。

**数据集特点**：
- **多平台覆盖**：包含仿真数据（SIL/HIL）和真实飞行数据，确保算法的泛化性
- **丰富故障类型**：涵盖11种常见故障类型，包括执行器、传感器故障和环境影响
- **多飞行状态**：包含悬停、航点飞行、速度控制、圆周飞行、加速/减速等6种飞行状态
- **高质量标注**：每个飞行案例都包含精确的故障注入时间点和故障参数
- **多模态数据**：提供ULog、TLog、Ground Truth和ROS Bag等多种数据格式

本项目处理的raw_data目录包含3000+个CSV文件，这些文件是从RflyMAD数据集中提取的预处理数据，专门用于深度学习模型训练。每个CSV文件包含47列数据，涵盖无人机的执行器控制、传感器读数、飞行状态等完整的飞行数据记录。

## 核心技术

- **多模态学习**：同时处理时序信号和时频图像数据
- **CNN-Transformer架构**：结合卷积神经网络的局部特征提取能力和Transformer的全局建模能力
- **小波变换**：将时序信号转换为时频域表示，增强特征表达能力
- **注意力机制**：通过多头自注意力机制捕获时序数据中的长距离依赖关系

## 文件结构

```
code/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
├── raw_data/                           # RflyMAD数据集提取的CSV文件 (3000+ 文件)
│   ├── Case_1000000000.csv           # 正常飞行数据
│   ├── Case_1001000001.csv           # 电机故障数据  
│   ├── Case_1002000001.csv           # 螺旋桨故障数据
│   └── Case_1010000001.csv           # 传感器故障数据
├── ipt/                               # 预处理后的数据
│   ├── dataset.npz                    # 标准化后的时序数据 (N, 41, 32)
│   ├── wavelet_dataset.npz            # 小波变换时频图数据 (N, 31, 41, 41)
│   ├── X.npy                         # 时序特征数据
│   └── y.npy                         # 标签数据
├── fig/                               # 训练过程可视化
│   └── loss_curve.png                # 训练和验证损失曲线
├── compare/                           # 模型对比工具
│   ├── transformer/                   # 单Transformer模型
│   │   ├── transformer_model.py       # Transformer模型定义
│   │   ├── train_transformer.py       # Transformer训练脚本
│   │   ├── best_transformer.pth       # 最佳模型权重
│   │   └── results/
│   │       └── loss_curve.png        # 训练损失曲线
│   ├── cnn/                          # 单CNN模型
│   │   ├── cnn_model.py              # CNN模型定义
│   │   ├── train_cnn.py              # CNN训练脚本
│   │   ├── best_cnn.pth              # 最佳模型权重
│   │   └── results/
│   │       └── loss_curve.png        # 训练损失曲线
│   ├── model_comparison.py           # 三模型准确率对比工具
│   └── results/                      # 对比结果
│       ├── accuracy_comparison_report.txt
│       └── accuracy_comparison_charts.png
├── data_process.py                    # 数据预处理脚本
├── wavelet_transform.py               # 小波变换处理脚本
├── cnn_transformer_model.py           # 多模态模型架构定义
├── train_cnn_transformer.py           # 多模态模型训练脚本
└── cnn_transformer_checkpoint.pth     # 训练好的模型权重
```

## 核心模块说明

### 1. 数据预处理 (`data_process.py`)

**功能**：处理原始CSV格式的无人机飞行数据，提取特征并进行标准化。

**输入数据详情**：

#### 特征列 (FEATURE_COLS) - 共31列

**C-K 列组 (索引 2-10)：执行器控制和输出数据**
- `trueTime` - 真实时间戳
- `_actuator_controls_0_0_control[0-3]` - 飞控的虚拟控制输入
- `_actuator_outputs_0_output[0-3]` - 飞控给出的电机PWM输出

**T-AF 列组 (索引 19-31)：传感器数据**
- `_sensor_combined_0_gyro_rad[0-2]` - 陀螺仪数据 (3轴，弧度/秒)
- `_sensor_combined_0_accelerometer_m_s2[0-2]` - 加速度计数据 (3轴，米/秒²)
- `_vehicle_air_data_0_baro_alt_meter` - 气压高度 (米)
- `_vehicle_air_data_0_baro_temp_celcius` - 气压温度 (摄氏度)
- `_vehicle_air_data_0_baro_pressure_pa` - 气压值 (帕斯卡)
- `_vehicle_attitude_0_q[0-3]` - 飞行器姿态四元数 (4个分量)

**AJ-AS 列组 (索引 35-44)：位置、速度和电机数据**
- `_vehicle_local_position_0_vx/vy/vz` - 本地坐标系速度 (3轴，米/秒)
- `_vehicle_magnetometer_0_magnetometer_ga[0-2]` - 磁力计数据 (3轴，高斯)
- `TrueState_data_motorRPMs[1-4]` - 电机转速 (4个电机的RPM)

#### 辅助列 (AU_IDX) - 用于时间窗口定位
- `UAVState_data_fault_state` (索引 46) - 无人机故障状态标志

**技术实现**：
- 从CSV文件名中解析故障代码，映射为4类故障标签
- 基于AU列的状态跃迁（0→1）确定关键时间窗口，提取41个时间步的数据段
- 选择上述31个关键传感器特征列（排除时间列后为31列）
- 对字符串格式数据进行清洗和类型转换
- 使用Z-score标准化处理特征数据

**输出**：`(N, 41, 32)`形状的三维数组，其中N为样本数，41为时间步长，32为特征维度（包含时间列）

### 2. 小波变换 (`wavelet_transform.py`)

**功能**：将时序信号转换为时频域表示，生成小波变换时频图。

**输入数据**：`(N, 41, 32)`形状的时序数据（来自data_process.py）

**技术实现**：
- 使用PyWavelets库的连续小波变换(CWT)，采用Morlet小波
- 排除时间列，对31个传感器通道分别进行小波变换
- 每个传感器通道生成41×41的时频图
- 通过逐通道标准化确保数据分布的一致性

**输出**：`(N, 31, 41, 41)`形状的四维数组，其中：
- N: 样本数
- 31: 传感器通道数（排除时间列）
- 41×41: 时频分辨率（频率×时间）

### 3. 模型架构 (`cnn_transformer_model.py`)

**核心组件**：

#### SignalTransformer
- 处理多通道时序信号的Transformer编码器
- 通过线性投影将32维特征映射到64维嵌入空间
- 使用位置编码和多头自注意力机制捕获时序依赖
- 采用3层Transformer编码器，8个注意力头

#### CWT_CNN_Transformer
- 先用CNN提取时频图的空间特征
- 两层卷积+池化操作，通道数逐步增加到32
- 将CNN输出展平为序列，输入Transformer进行全局建模
- 嵌入维度96，3层编码器结构

#### MultiModalClassifier
- 整合两路特征进行多模态融合
- 全局平均池化聚合序列特征
- 三层全连接分类器，包含Dropout正则化
- 输出4类故障预测结果

**数据流**：
```
时序信号 (B,41,32) → SignalTransformer → 特征1 (B,64)
                                              ↓
时频图 (B,31,41,41) → CWT_CNN_Transformer → 特征2 (B,96) → 拼接 → 分类器 → 预测 (B,4)
```

### 4. 训练脚本 (`train_cnn_transformer.py`)

**训练策略**：
- 针对数据不平衡问题使用加权交叉熵损失函数
- AdamW优化器，学习率2e-4，权重衰减1e-4
- 余弦退火学习率调度，动态调整学习率
- 批大小128，训练50个epoch

**正则化技术**：
- Dropout层防止过拟合（比例0.3-0.4）
- 批标准化稳定训练过程
- 权重衰减控制模型复杂度

## 故障类型和数据分布

基于RflyMAD数据集的故障分类编码，本项目将无人机故障分为4个主要类别。根据RflyMAD数据集的编码规则，故障代码的第3-4位数字表示具体的故障类型：

| 故障代码 | 标签 | 故障类型 | 描述 | 样本数量 |
|---------|------|----------|------|----------|
| 00      | 0    | 正常状态 | 无故障的正常飞行状态 | 1,760 |
| 01      | 1    | 螺旋桨故障 | 螺旋桨损坏、脱落或效率下降 | 870 |
| 02      | 2    | 低电压故障 | 电池电压不足影响飞行性能 | 72 |
| 10      | 3    | 电机故障 | 电机失效、转速异常或功率损失 | 480 |

**数据特点**：
- **数据不平衡**：各故障类型样本数量差异较大，低电压故障样本最少(72个)
- **故障注入精确**：每个故障案例都有明确的故障注入时间点，通过UAVState_data_fault_state字段标识
- **飞行状态多样**：涵盖悬停、航点飞行、速度控制等多种飞行模式下的故障场景
- **传感器丰富**：包含46个维度的飞行数据，涵盖姿态、位置、速度、电机转速等关键参数

本项目针对数据不平衡问题采用加权交叉熵损失函数，确保模型对少样本故障类型也能有效学习。

## 环境要求

- Python 3.8+
- CUDA 11.0+（推荐，用于GPU加速训练）
- 推荐配置：服务器级GPU（本项目在A100-40G上进行训练和验证）

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 处理原始CSV数据，生成标准化的时序数据
python data_process.py

# 生成小波变换时频图数据
python wavelet_transform.py
```

### 3. 模型训练

#### 训练CNN-Transformer多模态模型
```bash
# 使用默认参数训练
python train_cnn_transformer.py

# 自定义训练参数
python train_cnn_transformer.py --epochs 100 --batch_size 64 --lr 1e-4
```

#### 训练单模型进行对比
```bash
# 训练单Transformer模型
python compare/transformer/train_transformer.py

# 训练单CNN模型
python compare/cnn/train_cnn.py
```

### 4. 模型对比

```bash
# 运行三模型准确率对比
python compare/model_comparison.py
```

对比工具会自动：
- 加载三个训练好的模型
- 在相同的测试集上评估准确率
- 生成详细的对比报告和可视化图表

### 5. 训练监控

每个模型训练过程中会自动生成：
- 训练和验证损失曲线 (`loss_curve.png`)
- 最佳模型权重文件 (`.pth`)
- 分类报告 (各类别的精确率、召回率、F1分数)

## 技术特色

1. **多模态融合**：结合时域和频域信息，提升故障检测的鲁棒性
2. **注意力机制**：自动学习关键时间点和特征通道的重要性
3. **端到端训练**：从原始传感器数据到故障分类的完整pipeline
4. **数据增强**：通过小波变换增加数据的特征表达维度
5. **不平衡处理**：针对故障样本稀少的实际情况设计加权损失

## 模型对比研究

本项目实现了三种不同架构的故障检测模型，并进行了详细的性能对比研究：

### 三种模型架构

#### 1. CNN-Transformer（多模态模型）
- **输入数据**：
  - 时序信号：`(N, 41, 32)` - 包含31个传感器特征 + 时间戳
  - 小波图像：`(N, 31, 41, 41)` - 31个传感器通道的时频图
- **数据来源**：同时使用 `dataset.npz` 和 `wavelet_dataset.npz`
- **架构特点**：信号Transformer分支 + 图像CNN分支 + 特征融合
- **参数量**：5.16M

#### 2. 单CNN模型
- **输入数据**：`(N, 31, 41, 41)` - 小波变换时频图
- **数据来源**：`wavelet_dataset.npz`
- **数据内容**：31个传感器通道（陀螺仪、加速度计、气压、姿态、速度、磁力计、电机转速）的时频表示
- **架构特点**：4层卷积块 + 全局平均池化 + 分类器
- **参数量**：4.87M

#### 3. 单Transformer模型
- **输入数据**：`(N, 41, 32)` - 时序信号
- **数据来源**：`dataset.npz`
- **数据内容**：41个时间步 × 32个特征（31个传感器特征 + 时间戳）
- **架构特点**：8层Transformer编码器 + 多头自注意力
- **参数量**：4.33M

### 实验配置

**统一训练配置**：
- **训练轮数**：50 epochs
- **批大小**：128
- **优化器**：AdamW (lr=2e-4, weight_decay=1e-4)
- **学习率调度**：余弦退火调度
- **损失函数**：加权交叉熵（处理数据不平衡）

**数据分割**：
- 训练集：80%
- 验证集：20%

### 对比结果

| 模型 | 参数量 | 整体准确率 | 正常 | 螺旋桨故障 | 低电压故障 | 电机故障 |
|------|--------|------------|------|------------|------------|----------|
| CNN-Transformer | 5.16M | 97.90% | 97.70% | 99.64% | 80.77% | 98.52% |
| Transformer | 4.33M | 95.10% | 96.28% | 99.64% | 46.15% | 90.37% |
| CNN | 4.87M | 93.40% | 90.62% | 99.27% | 69.23% | 97.78% |

### 实验结果分析

**整体准确率排序**：
1. CNN-Transformer: 97.90%
2. Transformer: 95.10%
3. CNN: 93.40%

**各类别准确率分析**：
- **正常状态**: CNN-Transformer (97.70%) > Transformer (96.28%) > CNN (90.62%)
- **螺旋桨故障**: 三个模型表现相近，均超过99%
- **低电压故障**: CNN-Transformer (80.77%) > CNN (69.23%) > Transformer (46.15%)
- **电机故障**: CNN-Transformer (98.52%) > CNN (97.78%) > Transformer (90.37%)

**参数量对比**：
- CNN-Transformer: 5.16M (最多)
- CNN: 4.87M (中等)
- Transformer: 4.33M (最少)

**模型特性总结**：
- CNN-Transformer模型在整体准确率和各类别准确率上均表现最佳
- Transformer模型在低电压故障检测上准确率较低 (46.15%)
- CNN模型在电机故障检测上表现接近CNN-Transformer模型
- 三个模型在螺旋桨故障检测上表现相当

## 性能指标

- **最佳模型**：CNN-Transformer多模态模型
- **整体准确率**：97.90%
- **模型参数量**：5.16M
- **训练配置**：50 epochs, batch_size=128

## 模型对比结论

### 性能对比总结

**准确率表现**：
- CNN-Transformer模型在整体准确率(97.90%)和大部分类别准确率上表现最佳
- Transformer模型整体准确率为95.10%，在低电压故障检测上表现不佳(46.15%)
- CNN模型整体准确率为93.40%，在电机故障检测上表现良好(97.78%)

**参数量对比**：
- 三个模型的参数量相近，均在4-5M范围内
- CNN-Transformer: 5.16M
- CNN: 4.87M  
- Transformer: 4.33M

**模型特点**：
- CNN-Transformer: 多模态融合，各类别表现均衡
- Transformer: 序列建模能力强，但在低电压故障检测上存在局限
- CNN: 图像特征提取，在电机故障检测上表现突出

## 扩展说明

本系统设计具有良好的可扩展性，可以通过以下方式进行改进：
- 增加更多传感器通道的数据
- 尝试不同的小波基函数
- 调整网络架构的深度和宽度
- 引入更多的数据增强技术
- 集成更多的故障类型
- 探索模型集成和知识蒸馏技术

## 技术参考

- **RflyMAD数据集**：[RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment](https://rfly-openha.github.io/documents/4_resources/dataset.html)
- **Transformer架构**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **小波变换**：PyWavelets库文档
- **多模态学习**：深度学习中的特征融合技术
- **北航可靠飞行控制研究组**：RflySim仿真平台和数据处理工具