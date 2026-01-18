#!/bin/bash
# ------------------------------------------------------------------------
# AIMET A16W8 Quantization-Aware Training Script
# 16-bit activation, 8-bit weight quantization
# ------------------------------------------------------------------------

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 配置文件路径
CONFIG_FILE="options/train/quantization/NAFNet_AIMET_a16w8.yml"
QUANT_CONFIG="options/train/quantization/aimet_config_a16w8_perchannel.json"

# 预训练模型路径（FP32 模型）
PRETRAINED_MODEL="experiments/pretrained_models/NAFNet-SIDD-width32.pth"

# AIMET 量化参数
ACTIVATION_BW=16    # 16-bit activation
PARAM_BW=8          # 8-bit weight
CALIBRATION_BATCHES=100

# 训练参数
LAUNCHER="none"  # 可选: none, pytorch, slurm
LOCAL_RANK=0

# ------------------------------------------------------------------------
# 训练命令
# ------------------------------------------------------------------------

echo "=========================================="
echo "AIMET A16W8 QAT Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Quantization Config: $QUANT_CONFIG"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo "Activation Bitwidth: $ACTIVATION_BW"
echo "Weight Bitwidth: $PARAM_BW"
echo "Calibration Batches: $CALIBRATION_BATCHES"
echo "=========================================="

# 基础训练命令
python -u basicsr/train_aimet_a16w8.py \
    -opt "$CONFIG_FILE" \
    --launcher "$LAUNCHER" \
    --local_rank "$LOCAL_RANK" \
    --quantization_config "$QUANT_CONFIG" \
    --activation_bw "$ACTIVATION_BW" \
    --param_bw "$PARAM_BW" \
    --calibration_batches "$CALIBRATION_BATCHES" \
    --pretrained_model "$PRETRAINED_MODEL" \
    --weight_aug \
    --mixed_precision_aug

# 训练完成
echo "=========================================="
echo "Training completed!"
echo "=========================================="

