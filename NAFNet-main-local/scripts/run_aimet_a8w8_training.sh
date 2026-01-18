#!/bin/bash
# ------------------------------------------------------------------------
# AIMET A8W8 Quantization-Aware Training Script
# 8-bit activation, 8-bit weight quantization (Standard INT8)
# 
# This is the standard INT8 quantization configuration, providing:
# - Maximum hardware compatibility (TensorRT, ONNX Runtime INT8, etc.)
# - Smallest model size
# - Fastest inference speed
# - Best power efficiency
# ------------------------------------------------------------------------

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 配置文件路径
CONFIG_FILE="options/train/quantization/NAFNet_AIMET_a8w8.yml"
QUANT_CONFIG="options/train/quantization/aimet_config_a8w8_perchannel.json"

# 预训练模型路径（FP32 模型）
PRETRAINED_MODEL="experiments/pretrained_models/NAFNet-SIDD-width32.pth"

# AIMET 量化参数
ACTIVATION_BW=8     # 8-bit activation (Standard INT8)
PARAM_BW=8          # 8-bit weight (Standard INT8)
CALIBRATION_BATCHES=200  # A8W8 需要更多校准批次

# 训练参数
LAUNCHER="none"  # 可选: none, pytorch, slurm
LOCAL_RANK=0

# ------------------------------------------------------------------------
# 训练命令
# ------------------------------------------------------------------------

echo "=========================================="
echo "AIMET A8W8 QAT Training (Standard INT8)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Quantization Config: $QUANT_CONFIG"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo "Activation Bitwidth: $ACTIVATION_BW"
echo "Weight Bitwidth: $PARAM_BW"
echo "Calibration Batches: $CALIBRATION_BATCHES"
echo "=========================================="
echo ""
echo "特性说明:"
echo "- 标准 INT8 量化 (激活 8-bit + 权重 8-bit)"
echo "- Per-channel 权重量化提升精度"
echo "- 激进数据增强增强量化鲁棒性"
echo "- 适合 TensorRT/ONNX Runtime INT8 部署"
echo "=========================================="

# 基础训练命令
python -u basicsr/train_aimet_a8w8.py \
    -opt "$CONFIG_FILE" \
    --launcher "$LAUNCHER" \
    --local_rank "$LOCAL_RANK" \
    --quantization_config "$QUANT_CONFIG" \
    --activation_bw "$ACTIVATION_BW" \
    --param_bw "$PARAM_BW" \
    --calibration_batches "$CALIBRATION_BATCHES" \
    --pretrained_model "$PRETRAINED_MODEL" \
    --aggressive_aug \
    --weight_aug \
    --quant_aware_aug

# 训练完成
echo "=========================================="
echo "A8W8 Training completed!"
echo "=========================================="
echo ""
echo "输出文件位置:"
echo "- 模型: experiments/NAFNet_AIMET_a8w8_QAT/models/"
echo "- 日志: experiments/NAFNet_AIMET_a8w8_QAT/log/"
echo "- 量化编码: experiments/NAFNet_AIMET_a8w8_QAT/models/quantsim_a8w8_encodings_final.json"
echo "=========================================="

