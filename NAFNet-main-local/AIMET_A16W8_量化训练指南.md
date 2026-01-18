# AIMET A16W8 量化训练指南

> **NAFNet A16W8 量化感知训练 (QAT) 完整指南**
>
> - **激活精度**: 16-bit
> - **权重精度**: 8-bit
> - **框架**: AIMET (AI Model Efficiency Toolkit)

---

## 📋 目录

1. [A16W8 量化方案概述](#a16w8-量化方案概述)
2. [环境配置](#环境配置)
3. [文件说明](#文件说明)
4. [训练流程](#训练流程)
5. [关键参数说明](#关键参数说明)
6. [A16W8 vs A8W16 对比](#a16w8-vs-a8w16-对比)
7. [常见问题与解决方案](#常见问题与解决方案)
8. [性能优化建议](#性能优化建议)

---

## A16W8 量化方案概述

### 什么是 A16W8？

A16W8 表示：
- **A16**: Activation 16-bit（激活值使用 16 位量化）
- **W8**: Weight 8-bit（权重使用 8 位量化）

### 为什么选择 A16W8？

#### ✅ 优势

1. **更小的模型体积**
   - 权重占模型参数的绝大部分，8-bit 权重可显著减小模型大小（相比 16-bit 减半）
   - 适合部署到存储受限的设备

2. **更高的推理速度**
   - 8-bit 权重在硬件上计算更快
   - 内存带宽需求降低

3. **保持激活精度**
   - 16-bit 激活保证了中间特征的表达能力
   - 对暗区、细节等敏感区域的处理更精确

4. **适合边缘设备**
   - NPU/DSP 等硬件通常对 8-bit 权重有更好的支持
   - 功耗更低

#### ⚠️ 挑战

1. **权重量化损失**
   - 8-bit 权重相比 16-bit 会有精度损失
   - 需要更精细的量化策略（Per-channel quantization）

2. **训练难度**
   - QAT 训练需要更长时间收敛
   - 需要权重增强等技术辅助

3. **超参数调优**
   - 学习率、训练轮数等需要重新调整

---

## 环境配置

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision
pip install opencv-python numpy matplotlib
pip install pyyaml addict tqdm tensorboard

# AIMET 安装（Ubuntu 20.04, Python 3.8+）
# 方法1: 使用 pip（推荐）
pip install aimet-torch

# 方法2: 从源码安装
# 参考: https://github.com/quic/aimet
```

### 2. 验证安装

```python
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
print("AIMET installed successfully!")
```

---

## 文件说明

### 核心文件结构

```
NAFNet-main-local/
├── basicsr/
│   └── train_aimet_a16w8.py          # A16W8 训练主脚本
├── options/
│   └── train/
│       └── quantization/
│           ├── NAFNet_AIMET_a16w8.yml              # 训练配置文件
│           └── aimet_config_a16w8_perchannel.json  # AIMET 量化配置
├── scripts/
│   └── run_aimet_a16w8_training.sh   # 训练启动脚本
└── AIMET_A16W8_量化训练指南.md        # 本文档
```

### 文件功能说明

| 文件 | 功能 | 关键配置 |
|------|------|----------|
| `train_aimet_a16w8.py` | A16W8 QAT 训练主程序 | 数据增强、权重增强、训练循环 |
| `NAFNet_AIMET_a16w8.yml` | 训练超参数配置 | 学习率、batch size、数据集路径 |
| `aimet_config_a16w8_perchannel.json` | AIMET 量化配置 | Per-channel 量化、对称/非对称 |
| `run_aimet_a16w8_training.sh` | 训练启动脚本 | 环境变量、命令行参数 |

---

## 训练流程

### Step 1: 准备数据集

```bash
# 示例：SIDD 数据集结构
datasets/
└── SIDD/
    ├── train/
    │   ├── gt/     # 高质量图像
    │   └── noisy/  # 噪声图像
    └── val/
        ├── gt/
        └── noisy/
```

### Step 2: 准备预训练模型

```bash
# 下载 FP32 预训练模型
mkdir -p experiments/pretrained_models
# 将 NAFNet-SIDD-width32.pth 放入该目录
```

### Step 3: 修改配置文件

编辑 `options/train/quantization/NAFNet_AIMET_a16w8.yml`：

```yaml
# 数据集路径
datasets:
  train:
    dataroot_gt: datasets/SIDD/train/gt  # 修改为实际路径
    dataroot_lq: datasets/SIDD/train/noisy

  val:
    dataroot_gt: datasets/SIDD/val/gt
    dataroot_lq: datasets/SIDD/val/noisy

# 预训练模型路径
path:
  pretrain_network_g: experiments/pretrained_models/NAFNet-SIDD-width32.pth
```

### Step 4: 启动训练

#### 方法 1: 使用脚本启动（推荐）

```bash
# 赋予执行权限
chmod +x scripts/run_aimet_a16w8_training.sh

# 启动训练
bash scripts/run_aimet_a16w8_training.sh
```

#### 方法 2: 直接命令行启动

```bash
python -u basicsr/train_aimet_a16w8.py \
    -opt options/train/quantization/NAFNet_AIMET_a16w8.yml \
    --launcher none \
    --quantization_config options/train/quantization/aimet_config_a16w8_perchannel.json \
    --activation_bw 16 \
    --param_bw 8 \
    --calibration_batches 100 \
    --pretrained_model experiments/pretrained_models/NAFNet-SIDD-width32.pth \
    --weight_aug \
    --mixed_precision_aug
```

### Step 5: 监控训练

```bash
# TensorBoard 监控
tensorboard --logdir experiments/NAFNet_AIMET_a16w8_QAT/tb_logger

# 查看日志
tail -f experiments/NAFNet_AIMET_a16w8_QAT/log/train_aimet_a16w8_*.log
```

---

## 关键参数说明

### 1. 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--activation_bw` | 16 | 激活值位宽（16-bit） |
| `--param_bw` | 8 | 权重位宽（8-bit） |
| `--quantization_config` | `aimet_config_a16w8_perchannel.json` | AIMET 配置文件（Per-channel 量化） |
| `--calibration_batches` | 100 | 校准批次数（用于计算量化参数） |

### 2. 训练参数

| 参数 | 配置文件位置 | 推荐值 | 说明 |
|------|------------|--------|------|
| `learning_rate` | `train.optim_g.lr` | 8e-5 | 比 A8W16 更小（8-bit 权重更敏感） |
| `total_iter` | `train.total_iter` | 250000 | 比 A8W16 多 25%（需要更长收敛） |
| `warmup_iter` | `train.warmup_iter` | 2000 | 更长的预热期 |
| `batch_size_per_gpu` | `datasets.train.batch_size_per_gpu` | 8 | 根据显存调整 |

### 3. 数据增强参数

#### 3.1 BLC 噪声增强

```python
AdvancedBLCNoiseAugmentation(
    blc_std=2.5,        # BLC 误差标准差
    fpn_row_std=1.5,    # 行固定模式噪声
    fpn_col_std=1.5,    # 列固定模式噪声
    aug_prob=0.5        # 增强概率
)
```

#### 3.2 混合精度增强

启用参数：`--mixed_precision_aug`

```python
MixedPrecisionAugmentation(
    aug_prob=0.4  # 增强概率
)
```

增强类型：
- **亮度调整**: 0.5 ~ 1.5x
- **对比度调整**: 0.7 ~ 1.3x
- **Gamma 校正**: 0.8 ~ 1.2

#### 3.3 权重增强

启用参数：`--weight_aug`

```python
WeightAugmentation(
    noise_std=0.001,  # 噪声标准差
    aug_prob=0.3      # 增强概率（每5个iter）
)
```

**作用**: 对 8-bit 权重添加微小扰动，增强量化鲁棒性

---

## A16W8 vs A8W16 对比

### 量化方案对比

| 维度 | A16W8 | A8W16 |
|------|-------|-------|
| **激活位宽** | 16-bit | 8-bit |
| **权重位宽** | 8-bit | 16-bit |
| **模型大小** | ⭐⭐⭐⭐⭐ 更小 | ⭐⭐⭐ 中等 |
| **推理速度** | ⭐⭐⭐⭐ 快 | ⭐⭐⭐ 中等 |
| **激活精度** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中等 |
| **权重精度** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 高 |
| **训练难度** | ⭐⭐⭐⭐ 较难 | ⭐⭐⭐ 中等 |
| **硬件支持** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐ 良好 |

### 适用场景

#### A16W8 适合：
- ✅ 需要小模型体积（存储受限）
- ✅ 需要快速推理（边缘设备）
- ✅ 激活精度要求高（图像处理）
- ✅ 硬件支持 8-bit 权重加速

#### A8W16 适合：
- ✅ 需要高权重精度（复杂模型）
- ✅ 激活量化不敏感
- ✅ 训练时间有限
- ✅ 权重复用率高的架构

### 性能预期

基于 NAFNet-SIDD-width32：

| 指标 | FP32 | A8W16 | A16W8 |
|------|------|-------|-------|
| **PSNR (dB)** | 39.96 | 39.82 (-0.14) | 39.75 (-0.21) |
| **模型大小 (MB)** | 12.5 | ~6.8 | ~4.2 |
| **推理速度** | 1.0x | 1.3x | 1.6x |

> 注：实际性能取决于具体硬件和优化

---

## 常见问题与解决方案

### Q1: AIMET 安装失败

**问题**：`ImportError: No module named 'aimet_torch'`

**解决方案**：
```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 使用国内源安装
pip install aimet-torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或从源码安装
git clone https://github.com/quic/aimet.git
cd aimet
# 按照官方文档编译安装
```

### Q2: 校准阶段卡死

**问题**：Calibration 阶段长时间无响应

**解决方案**：
```bash
# 减少校准批次
--calibration_batches 50

# 检查数据加载器
# 确保 prefetch_mode 设置正确
# datasets.train.prefetch_mode: cpu 或 cuda
```

### Q3: 训练 loss 不收敛

**问题**：Loss 震荡或不下降

**解决方案**：

1. **降低学习率**
   ```yaml
   train:
     optim_g:
       lr: !!float 5e-5  # 从 8e-5 降低
   ```

2. **增加 warmup**
   ```yaml
   train:
     warmup_iter: 3000  # 从 2000 增加
   ```

3. **检查预训练模型**
   ```bash
   # 确保 FP32 预训练模型加载成功
   # 查看日志：Loading pretrained model from ...
   ```

4. **启用所有数据增强**
   ```bash
   --weight_aug --mixed_precision_aug
   ```

### Q4: 量化后精度下降严重

**问题**：PSNR 下降 > 0.5 dB

**解决方案**：

1. **使用 Per-channel 量化**（已默认启用）
   ```json
   // aimet_config_a16w8_perchannel.json
   "per_channel_quantization": "True"
   ```

2. **增加训练轮数**
   ```yaml
   train:
     total_iter: 300000  # 从 250000 增加
   ```

3. **调整量化配置**
   ```json
   // 对特定层使用更高精度
   "op_name": {
     "model.encoder.conv_first": {
       "params": {
         "weight": {
           "is_quantized": "False"  // 首层不量化
         }
       }
     }
   }
   ```

### Q5: 导出的量化编码如何使用？

**问题**：如何使用导出的 `quantsim_a16w8_encodings_final.json`？

**解决方案**：

```python
# 部署时加载量化编码
from aimet_torch.quantsim import QuantizationSimModel

# 1. 加载 FP32 模型
model = NAFNet(...)

# 2. 创建 QuantSim
quantsim = QuantizationSimModel(model, ...)

# 3. 加载量化编码
quantsim.load_encodings('quantsim_a16w8_encodings_final.json')

# 4. 加载 QAT 训练的权重
checkpoint = torch.load('net_g_latest.pth')
model.load_state_dict(checkpoint['params'])

# 5. 导出 ONNX 等格式
quantsim.export(...)
```

---

## 性能优化建议

### 1. 训练加速

#### 使用混合精度训练（AMP）

虽然是量化训练，但可以结合 AMP 加速：

```python
# 在 train_aimet_a16w8.py 中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 数据加载优化

```yaml
datasets:
  train:
    num_worker_per_gpu: 16  # 增加数据加载线程
    prefetch_mode: cuda     # 使用 CUDA 预取
    pin_memory: true        # 启用 pin_memory
```

### 2. 内存优化

#### Gradient Checkpointing

对于大模型，启用梯度检查点：

```python
# 在模型中添加
import torch.utils.checkpoint as checkpoint

def forward(self, x):
    x = checkpoint.checkpoint(self.encoder, x)
    ...
```

#### 降低 Batch Size

```yaml
datasets:
  train:
    batch_size_per_gpu: 4  # 从 8 降低到 4
```

### 3. 量化优化

#### 层级量化策略

对不同层使用不同量化精度：

```json
{
  "op_name": {
    "model.encoder.conv_first": {
      "params": {
        "weight": {"is_quantized": "False"}  // 首层 FP32
      }
    },
    "model.decoder.conv_last": {
      "params": {
        "weight": {"is_quantized": "False"}  // 尾层 FP32
      }
    }
  }
}
```

#### 渐进式量化

先训练高精度，再逐步降低：

1. **阶段 1**: A32W16 (50k iters)
2. **阶段 2**: A16W12 (50k iters)
3. **阶段 3**: A16W8 (150k iters)

---

## 训练监控指标

### 关键指标

| 指标 | 监控方式 | 正常范围 |
|------|---------|----------|
| **训练 Loss** | TensorBoard | 逐步下降，最终 < 0.01 |
| **验证 PSNR** | 日志文件 | 逐步上升，最终 > 39.5 dB |
| **学习率** | TensorBoard | 余弦退火，从 8e-5 降到 1e-7 |
| **GPU 显存** | `nvidia-smi` | < 24GB (单卡) |
| **数据加载时间** | 日志 `data_time` | < 0.1s |

### 异常情况处理

| 异常 | 可能原因 | 解决方案 |
|------|---------|----------|
| Loss 突然飙升 | 学习率过大 | 降低学习率或重启训练 |
| PSNR 不提升 | 数据增强过强 | 降低 aug_prob |
| OOM 错误 | 显存不足 | 降低 batch_size 或 gt_size |
| 校准失败 | 数据路径错误 | 检查数据集路径配置 |

---

## 部署与转换

### 1. 导出 ONNX

```python
import torch
from aimet_torch.quantsim import QuantizationSimModel

# 加载量化模型
model = ...  # 加载训练好的模型
quantsim = QuantizationSimModel(model, ...)
quantsim.load_encodings('quantsim_a16w8_encodings_final.json')

# 导出 ONNX
dummy_input = torch.randn(1, 3, 256, 256).cuda()
torch.onnx.export(
    quantsim.model,
    dummy_input,
    'nafnet_a16w8.onnx',
    opset_version=13,
    input_names=['input'],
    output_names=['output']
)
```

### 2. 部署到 TensorRT

```python
import tensorrt as trt

# 加载 ONNX
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open('nafnet_a16w8.onnx', 'rb') as model:
    parser.parse(model.read())

# 设置精度
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)  # 8-bit 量化
config.set_flag(trt.BuilderFlag.FP16)  # 16-bit 激活

# 构建引擎
engine = builder.build_engine(network, config)
```

### 3. 部署到移动端（NCNN）

```bash
# 转换为 NCNN 格式
onnx2ncnn nafnet_a16w8.onnx nafnet_a16w8.param nafnet_a16w8.bin

# 量化为 int8
ncnnoptimize nafnet_a16w8.param nafnet_a16w8.bin \
    nafnet_a16w8_int8.param nafnet_a16w8_int8.bin 0
```

---

## 总结

### A16W8 量化训练要点

✅ **关键成功因素**
1. 使用高质量 FP32 预训练模型
2. Per-channel 权重量化
3. 启用权重增强和混合精度增强
4. 更长的训练时间（250k iters）
5. 较小的学习率（8e-5）

⚠️ **注意事项**
1. 8-bit 权重对超参数敏感，需仔细调优
2. 校准阶段需要代表性数据
3. 监控训练过程，及时调整策略
4. 保存多个 checkpoint，选择最佳模型

📊 **性能预期**
- 模型大小：减小约 65%（相比 FP32）
- 推理速度：提升约 1.6x
- PSNR 损失：< 0.3 dB

---

## 参考资料

- [AIMET Documentation](https://quic.github.io/aimet-pages/releases/latest/user_guide/index.html)
- [NAFNet Paper](https://arxiv.org/abs/2204.04676)
- [Quantization-Aware Training](https://arxiv.org/abs/1712.05877)

---

## 更新日志

- **2026-01-18**: 初始版本，完整 A16W8 QAT 训练流程

---

**如有问题，欢迎提交 Issue 或联系开发者！**

