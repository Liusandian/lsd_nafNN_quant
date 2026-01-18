# AIMET A8W8 量化训练指南

> **NAFNet A8W8 标准 INT8 量化感知训练 (QAT) 完整指南**
>
> - **激活精度**: 8-bit
> - **权重精度**: 8-bit
> - **框架**: AIMET (AI Model Efficiency Toolkit)
> - **特点**: 标准 INT8 量化，硬件兼容性最佳

---

## 📋 目录

1. [A8W8 量化方案概述](#a8w8-量化方案概述)
2. [环境配置](#环境配置)
3. [文件说明](#文件说明)
4. [训练流程](#训练流程)
5. [关键参数说明](#关键参数说明)
6. [A8W8 vs A8W16 vs A16W8 对比](#a8w8-vs-a8w16-vs-a16w8-对比)
7. [数据增强策略](#数据增强策略)
8. [常见问题与解决方案](#常见问题与解决方案)
9. [性能优化建议](#性能优化建议)
10. [部署与转换](#部署与转换)

---

## A8W8 量化方案概述

### 什么是 A8W8？

A8W8 表示：
- **A8**: Activation 8-bit（激活值使用 8 位量化）
- **W8**: Weight 8-bit（权重使用 8 位量化）

这是 **标准 INT8 量化**，激活和权重均使用 8 位整数表示，是最常见的深度学习量化配置。

### 为什么选择 A8W8？

#### ✅ 优势

1. **最佳硬件兼容性**
   - 几乎所有 AI 加速器都支持 INT8（NVIDIA TensorRT, Intel OpenVINO, Qualcomm SNPE 等）
   - NPU、DSP、专用 ASIC 的原生支持
   - 无需特殊硬件指令

2. **最小模型体积**
   - 相比 FP32 模型减小约 75%
   - 内存占用最低
   - 适合边缘设备和嵌入式系统

3. **最快推理速度**
   - INT8 计算比 FP32 快 2-4x
   - 内存带宽需求最低
   - 功耗最小

4. **工业标准**
   - 最成熟的量化方案
   - 广泛的工具链支持
   - 大量参考资料和最佳实践

#### ⚠️ 挑战

1. **精度损失风险**
   - 8-bit 激活 + 8-bit 权重可能导致较大精度损失
   - 暗区和细节处理可能受影响
   - 需要更精细的量化策略

2. **训练难度最高**
   - 需要更长的 QAT 训练时间
   - 需要更激进的数据增强
   - 超参数调优更复杂

3. **敏感层处理**
   - 某些层可能需要保持高精度
   - 需要仔细分析和调整量化配置

---

## 环境配置

### 1. 系统要求

- **操作系统**: Ubuntu 18.04/20.04/22.04 或 Windows 10/11
- **Python**: 3.8+
- **CUDA**: 11.0+
- **GPU**: NVIDIA GPU with 16GB+ VRAM（推荐）

### 2. 安装依赖

```bash
# 基础依赖
pip install torch>=1.10.0 torchvision
pip install opencv-python numpy matplotlib scipy
pip install pyyaml addict tqdm tensorboard

# AIMET 安装（Ubuntu）
# 方法1: 使用 pip（推荐）
pip install aimet-torch

# 方法2: 使用 conda
conda install -c aimet aimet-torch

# 方法3: 从源码安装（最新功能）
# 参考: https://github.com/quic/aimet
git clone https://github.com/quic/aimet.git
cd aimet
pip install -e .
```

### 3. 验证安装

```python
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("AIMET installed successfully!")

# 测试量化模拟
model = torch.nn.Conv2d(3, 64, 3, padding=1)
dummy_input = torch.randn(1, 3, 64, 64)

quantsim = QuantizationSimModel(
    model=model,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    dummy_input=dummy_input,
    default_output_bw=8,
    default_param_bw=8
)
print("QuantSim created successfully!")
```

---

## 文件说明

### 核心文件结构

```
NAFNet-main-local/
├── basicsr/
│   └── train_aimet_a8w8.py                   # A8W8 训练主脚本
├── options/
│   └── train/
│       └── quantization/
│           ├── NAFNet_AIMET_a8w8.yml         # 训练配置文件
│           └── aimet_config_a8w8_perchannel.json  # AIMET 量化配置
├── scripts/
│   └── run_aimet_a8w8_training.sh            # 训练启动脚本
└── AIMET_A8W8_量化训练指南.md                 # 本文档
```

### 文件功能说明

| 文件 | 功能 | 关键配置 |
|------|------|----------|
| `train_aimet_a8w8.py` | A8W8 QAT 训练主程序 | 激进数据增强、量化感知增强、训练循环 |
| `NAFNet_AIMET_a8w8.yml` | 训练超参数配置 | 学习率(5e-5)、总迭代(300k)、batch size(6) |
| `aimet_config_a8w8_perchannel.json` | AIMET 量化配置 | Per-channel 量化、对称权重、运算融合 |
| `run_aimet_a8w8_training.sh` | 训练启动脚本 | 环境变量、命令行参数 |

---

## 训练流程

### Step 1: 准备数据集

```bash
# 示例：SIDD 数据集结构
datasets/
└── SIDD/
    ├── train/
    │   ├── gt/     # 高质量图像 (Ground Truth)
    │   └── noisy/  # 噪声图像 (Low Quality)
    └── val/
        ├── gt/
        └── noisy/
```

### Step 2: 准备预训练模型

```bash
# 创建目录
mkdir -p experiments/pretrained_models

# 下载或复制 FP32 预训练模型到该目录
# 例如: NAFNet-SIDD-width32.pth
```

### Step 3: 修改配置文件

编辑 `options/train/quantization/NAFNet_AIMET_a8w8.yml`：

```yaml
# 数据集路径（修改为实际路径）
datasets:
  train:
    dataroot_gt: /path/to/your/SIDD/train/gt
    dataroot_lq: /path/to/your/SIDD/train/noisy

  val:
    dataroot_gt: /path/to/your/SIDD/val/gt
    dataroot_lq: /path/to/your/SIDD/val/noisy

# 预训练模型路径
path:
  pretrain_network_g: experiments/pretrained_models/NAFNet-SIDD-width32.pth
```

### Step 4: 启动训练

#### 方法 1: 使用脚本启动（推荐）

```bash
# 赋予执行权限
chmod +x scripts/run_aimet_a8w8_training.sh

# 启动训练
bash scripts/run_aimet_a8w8_training.sh
```

#### 方法 2: 直接命令行启动

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python -u basicsr/train_aimet_a8w8.py \
    -opt options/train/quantization/NAFNet_AIMET_a8w8.yml \
    --launcher none \
    --quantization_config options/train/quantization/aimet_config_a8w8_perchannel.json \
    --activation_bw 8 \
    --param_bw 8 \
    --calibration_batches 200 \
    --pretrained_model experiments/pretrained_models/NAFNet-SIDD-width32.pth \
    --aggressive_aug \
    --weight_aug \
    --quant_aware_aug
```

#### 方法 3: 多 GPU 分布式训练

```bash
# 2 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    basicsr/train_aimet_a8w8.py \
    -opt options/train/quantization/NAFNet_AIMET_a8w8.yml \
    --launcher pytorch \
    --quantization_config options/train/quantization/aimet_config_a8w8_perchannel.json \
    --activation_bw 8 \
    --param_bw 8 \
    --calibration_batches 200 \
    --pretrained_model experiments/pretrained_models/NAFNet-SIDD-width32.pth \
    --aggressive_aug \
    --weight_aug \
    --quant_aware_aug
```

### Step 5: 监控训练

```bash
# TensorBoard 监控
tensorboard --logdir experiments/NAFNet_AIMET_a8w8_QAT/tb_logger

# 实时查看日志
tail -f experiments/NAFNet_AIMET_a8w8_QAT/log/train_aimet_a8w8_*.log

# GPU 监控
watch -n 1 nvidia-smi
```

---

## 关键参数说明

### 1. 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--activation_bw` | 8 | 激活值位宽（标准 INT8） |
| `--param_bw` | 8 | 权重位宽（标准 INT8） |
| `--quantization_config` | `aimet_config_a8w8_perchannel.json` | AIMET 量化配置（Per-channel） |
| `--calibration_batches` | 200 | 校准批次数（比 A8W16/A16W8 更多） |

### 2. 训练参数（相比其他量化方案的调整）

| 参数 | A8W8 推荐值 | A8W16 值 | A16W8 值 | 说明 |
|------|------------|----------|----------|------|
| `learning_rate` | **5e-5** | 1e-4 | 8e-5 | 更小，因为双 8-bit 更敏感 |
| `total_iter` | **300000** | 200000 | 250000 | 更长，需要更多时间收敛 |
| `warmup_iter` | **3000** | 1500 | 2000 | 更长，稳定初始训练 |
| `batch_size_per_gpu` | **6** | 8 | 8 | 稍小，提高稳定性 |
| `calibration_batches` | **200** | 100 | 100 | 更多，提高量化精度 |

### 3. 数据增强参数

#### 3.1 激进暗区增强

```python
AggressiveDarkRegionAugmentation(
    dark_threshold=0.25,      # 暗区阈值
    aug_prob=0.6,             # 增强概率（比其他方案更高）
    brightness_range=(0.2, 0.8)  # 亮度范围（更激进）
)
```

#### 3.2 增强版 BLC 噪声

```python
EnhancedBLCNoiseAugmentation(
    blc_std=3.0,          # BLC 误差标准差（更强）
    fpn_row_std=2.0,      # 行噪声（更强）
    fpn_col_std=2.0,      # 列噪声（更强）
    aug_prob=0.6          # 增强概率
)
```

#### 3.3 量化感知增强

```python
QuantizationAwareAugmentation(
    aug_prob=0.5  # 增强概率
)
```

增强类型：
- **亮度调整**: 0.4 ~ 1.6x（更激进）
- **对比度调整**: 0.6 ~ 1.4x
- **Gamma 校正**: 0.7 ~ 1.3
- **模拟量化**: 6/7/8-bit 随机量化

#### 3.4 权重量化增强

```python
WeightQuantizationAugmentation(
    noise_std=0.002,  # 噪声标准差（比其他方案更大）
    aug_prob=0.4      # 增强概率（每3个iter）
)
```

---

## A8W8 vs A8W16 vs A16W8 对比

### 量化方案对比表

| 维度 | A8W8 | A8W16 | A16W8 |
|------|------|-------|-------|
| **激活位宽** | 8-bit | 8-bit | 16-bit |
| **权重位宽** | 8-bit | 16-bit | 8-bit |
| **模型大小** | ⭐⭐⭐⭐⭐ 最小 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较小 |
| **推理速度** | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ 快 | ⭐⭐⭐⭐ 快 |
| **硬件兼容** | ⭐⭐⭐⭐⭐ 最佳 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 良好 |
| **精度保持** | ⭐⭐⭐ 有损失 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 良好 |
| **训练难度** | ⭐⭐⭐⭐⭐ 最难 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较难 |
| **功耗效率** | ⭐⭐⭐⭐⭐ 最佳 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 良好 |

### 性能预期（基于 NAFNet-SIDD-width32）

| 指标 | FP32 | A8W16 | A16W8 | A8W8 |
|------|------|-------|-------|------|
| **PSNR (dB)** | 39.96 | 39.82 (-0.14) | 39.75 (-0.21) | **39.60 (-0.36)** |
| **模型大小 (MB)** | 12.5 | ~6.8 | ~4.2 | **~3.2** |
| **推理速度** | 1.0x | 1.3x | 1.6x | **2.0x** |
| **功耗** | 1.0x | 0.7x | 0.6x | **0.5x** |

> 注：实际性能取决于具体硬件、数据集和优化程度

### 适用场景

#### A8W8 适合：
- ✅ 最严格的模型大小限制
- ✅ 需要最快推理速度
- ✅ 功耗敏感的边缘设备
- ✅ 需要广泛硬件兼容性
- ✅ 精度损失可接受（< 0.5 dB）

#### A8W16 适合：
- ✅ 需要较高权重精度
- ✅ 复杂模型结构
- ✅ 训练时间有限

#### A16W8 适合：
- ✅ 需要较高激活精度
- ✅ 图像处理任务（暗区、细节敏感）
- ✅ 需要小模型但保持特征精度

---

## 数据增强策略

A8W8 量化需要最激进的数据增强策略来弥补精度损失。

### 增强策略优先级

1. **量化感知增强**（最重要）
   - 模拟量化效果，让模型适应低精度
   - `--quant_aware_aug` 启用

2. **暗区增强**
   - INT8 在暗区特别容易出现量化误差
   - `--aggressive_aug` 启用

3. **权重扰动**
   - 增强权重对量化的鲁棒性
   - `--weight_aug` 启用

4. **BLC 噪声**
   - 模拟真实传感器噪声
   - 默认启用

### 增强概率配置

```python
# A8W8 推荐增强概率（比其他方案更高）
dark_aug_prob = 0.6       # 暗区增强
blc_aug_prob = 0.6        # BLC 噪声
quant_aug_prob = 0.5      # 量化感知增强
weight_aug_prob = 0.4     # 权重扰动（每3个iter）
```

---

## 常见问题与解决方案

### Q1: 训练不收敛，Loss 持续震荡

**问题描述**：Loss 在训练初期就开始震荡，不下降

**解决方案**：

1. **进一步降低学习率**
   ```yaml
   train:
     optim_g:
       lr: !!float 3e-5  # 从 5e-5 降低到 3e-5
   ```

2. **增加 warmup 迭代**
   ```yaml
   train:
     warmup_iter: 5000  # 从 3000 增加到 5000
   ```

3. **降低数据增强强度**
   ```python
   # 临时降低增强概率
   dark_aug_prob = 0.4   # 从 0.6 降低
   blc_aug_prob = 0.4    # 从 0.6 降低
   ```

4. **检查预训练模型**
   ```bash
   # 确认模型加载成功
   # 查看日志: "Loading pretrained model from ..."
   ```

### Q2: PSNR 下降超过 0.5 dB

**问题描述**：量化后精度损失过大

**解决方案**：

1. **增加训练迭代次数**
   ```yaml
   train:
     total_iter: 400000  # 从 300000 增加
   ```

2. **对敏感层使用更高精度**
   ```json
   // 在 aimet_config_a8w8_perchannel.json 中添加
   "op_name": {
     "intro": {
       "is_output_quantized": "False"
     },
     "ending": {
       "is_output_quantized": "False"
     }
   }
   ```

3. **使用渐进式量化**
   ```bash
   # 先训练 A8W16，再微调到 A8W8
   # Step 1: A8W16 (100k iters)
   # Step 2: A8W8 (200k iters) 从 A8W16 checkpoint 开始
   ```

### Q3: 校准阶段非常慢或卡死

**问题描述**：Calibration 阶段超过 30 分钟还没完成

**解决方案**：

1. **减少校准批次**
   ```bash
   --calibration_batches 100  # 从 200 减少到 100
   ```

2. **检查数据加载**
   ```yaml
   datasets:
     train:
       num_worker_per_gpu: 4  # 减少 worker 数量
       prefetch_mode: cpu     # 使用 CPU prefetch
   ```

3. **减小图像尺寸**
   ```yaml
   datasets:
     train:
       gt_size: 128  # 校准时用更小的尺寸
   ```

### Q4: GPU 显存不足 (OOM)

**问题描述**：`CUDA out of memory`

**解决方案**：

1. **减小 batch size**
   ```yaml
   datasets:
     train:
       batch_size_per_gpu: 4  # 从 6 减少到 4
   ```

2. **减小图像尺寸**
   ```yaml
   datasets:
     train:
       gt_size: 192  # 从 256 减少
   ```

3. **使用梯度检查点**（需要修改代码）
   ```python
   import torch.utils.checkpoint as checkpoint
   ```

### Q5: 量化编码导出失败

**问题描述**：`export_encodings` 报错

**解决方案**：

```python
# 确保在导出前完成校准
quantsim.compute_encodings(forward_pass_callback, None)

# 导出编码
try:
    quantsim.export(
        path='./export',
        filename_prefix='nafnet_a8w8',
        dummy_input=dummy_input
    )
except Exception as e:
    print(f"Export error: {e}")
    # 尝试只导出编码 JSON
    import json
    encodings = quantsim.get_all_encodings()
    with open('encodings.json', 'w') as f:
        json.dump(encodings, f)
```

---

## 性能优化建议

### 1. 训练加速

#### 混合精度训练

```python
# 在训练循环中使用 AMP
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
    num_worker_per_gpu: 12    # 增加数据加载线程
    prefetch_mode: cuda       # GPU 预取
    pin_memory: true          # 固定内存
```

### 2. 精度优化

#### 关键层保护

对首层和尾层使用更高精度：

```json
{
  "op_name": {
    "model.intro.conv": {
      "is_output_quantized": "False",
      "params": {
        "weight": {"is_quantized": "False"}
      }
    },
    "model.ending.conv": {
      "is_output_quantized": "False",
      "params": {
        "weight": {"is_quantized": "False"}
      }
    }
  }
}
```

#### 渐进式量化训练

```bash
# 阶段 1: A32W8 (权重先量化)
python train_aimet_a8w8.py --activation_bw 32 --param_bw 8 ...

# 阶段 2: A16W8 (激活逐步降低)
python train_aimet_a8w8.py --activation_bw 16 --param_bw 8 ...

# 阶段 3: A8W8 (最终配置)
python train_aimet_a8w8.py --activation_bw 8 --param_bw 8 ...
```

### 3. 量化配置优化

#### Per-channel vs Per-tensor

| 配置 | Per-channel | Per-tensor |
|------|-------------|------------|
| 精度 | 更高 | 较低 |
| 速度 | 稍慢 | 更快 |
| 推荐 | ✅ A8W8 权重 | 激活 |

```json
{
  "defaults": {
    "per_channel_quantization": "True"  // 权重使用 per-channel
  },
  "op_type": {
    "Conv": {
      "per_channel_quantization": "True"  // 卷积层 per-channel
    }
  }
}
```

---

## 部署与转换

### 1. 导出 ONNX

```python
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.onnx_utils import OnnxExportApiArgs

# 加载训练好的模型
model = NAFNet(...)
model.load_state_dict(torch.load('net_g_latest.pth')['params'])

# 创建 QuantSim 并加载编码
quantsim = QuantizationSimModel(
    model=model,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    dummy_input=torch.randn(1, 3, 256, 256).cuda(),
    default_output_bw=8,
    default_param_bw=8
)

# 加载量化编码
quantsim.load_encodings('quantsim_a8w8_encodings_final.json')

# 导出 ONNX
dummy_input = torch.randn(1, 3, 256, 256).cuda()
torch.onnx.export(
    quantsim.model,
    dummy_input,
    'nafnet_a8w8.onnx',
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    }
)

print("ONNX model exported!")
```

### 2. 部署到 TensorRT

```python
import tensorrt as trt

# 创建 builder
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# 解析 ONNX
parser = trt.OnnxParser(network, logger)
with open('nafnet_a8w8.onnx', 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# 配置 INT8
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 构建引擎
engine = builder.build_serialized_network(network, config)

# 保存引擎
with open('nafnet_a8w8.trt', 'wb') as f:
    f.write(engine)

print("TensorRT engine built!")
```

### 3. 部署到 ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# 创建 INT8 推理会话
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 使用 TensorRT EP 或 CUDA EP
providers = [
    ('TensorrtExecutionProvider', {
        'trt_int8_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession(
    'nafnet_a8w8.onnx',
    sess_options,
    providers=providers
)

# 推理
input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = session.run(None, {'input': input_data})
print(f"Output shape: {output[0].shape}")
```

### 4. 部署到移动端 (NCNN)

```bash
# 转换为 NCNN
onnx2ncnn nafnet_a8w8.onnx nafnet_a8w8.param nafnet_a8w8.bin

# INT8 量化优化
ncnnoptimize nafnet_a8w8.param nafnet_a8w8.bin \
    nafnet_a8w8_int8.param nafnet_a8w8_int8.bin 1
```

---

## 训练监控清单

### 关键指标监控

| 指标 | 监控方式 | 正常范围 | 异常处理 |
|------|---------|----------|---------|
| 训练 Loss | TensorBoard | 逐步下降，最终 < 0.015 | 降低 lr，增加 warmup |
| 验证 PSNR | 日志 | 逐步上升，最终 > 39.5 dB | 延长训练，调整增强 |
| 学习率 | TensorBoard | 余弦衰减 5e-5 → 1e-7 | 检查 scheduler |
| GPU 显存 | nvidia-smi | < 90% | 降低 batch size |
| data_time | 日志 | < 0.2s | 增加 workers |

### 训练阶段检查点

| 阶段 | 迭代次数 | 预期 PSNR | 检查项 |
|------|---------|-----------|--------|
| 校准完成 | 0 | N/A | 量化编码生成成功 |
| 早期 | 10k | > 35 dB | Loss 稳定下降 |
| 中期 | 100k | > 38 dB | 无震荡，稳定提升 |
| 后期 | 250k | > 39 dB | 收敛趋势明显 |
| 完成 | 300k | > 39.5 dB | 达到目标精度 |

---

## 总结

### A8W8 量化训练要点

**✅ 关键成功因素**

1. 高质量 FP32 预训练模型作为起点
2. Per-channel 权重量化（必须）
3. 激进数据增强（暗区、量化感知、权重扰动）
4. 充足的训练时间（300k iterations）
5. 较小的学习率（5e-5）和长 warmup（3000）
6. 更多的校准批次（200）

**⚠️ 注意事项**

1. A8W8 是最激进的量化方案，精度损失可能较大
2. 对超参数非常敏感，需要仔细调优
3. 建议从 A8W16 或 A16W8 模型微调
4. 监控训练过程，及时调整策略
5. 保存多个 checkpoint，选择最佳模型

**📊 性能预期**

- 模型大小：减小约 75%（相比 FP32）
- 推理速度：提升约 2x
- PSNR 损失：< 0.4 dB（经过充分训练）
- 硬件兼容：几乎所有 AI 加速器

---

## 参考资料

- [AIMET Documentation](https://quic.github.io/aimet-pages/releases/latest/user_guide/index.html)
- [NAFNet Paper](https://arxiv.org/abs/2204.04676)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [TensorRT INT8 Quantization](https://developer.nvidia.com/tensorrt)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)

---

## 更新日志

- **2026-01-18**: 初始版本，完整 A8W8 标准 INT8 QAT 训练流程

---

**如有问题，欢迎提交 Issue 或联系开发者！**

