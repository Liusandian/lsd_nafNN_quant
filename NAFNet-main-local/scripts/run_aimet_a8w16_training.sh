#!/bin/bash
# NAFNet AIMET 量化训练启动脚本
# 使用方法: bash scripts/run_aimet_training.sh

set -e

echo "========================================="
echo "NAFNet AIMET 量化训练 (a8w16)"
echo "========================================="

# 配置参数
CONFIG_FILE="options/train/quantization/NAFNet_AIMET_a8w16.yml"
QUANTIZATION_CONFIG="options/train/quantization/aimet_config_perchannel.json"
PRETRAINED_MODEL="experiments/pretrained_models/NAFNet-SIDD-width32.pth"
ACTIVATION_BW=8
PARAM_BW=16
CALIBRATION_BATCHES=100

# GPU 配置
NUM_GPUS=1
GPU_IDS="0"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$QUANTIZATION_CONFIG" ]; then
    echo "错误: AIMET 配置文件不存在: $QUANTIZATION_CONFIG"
    exit 1
fi

# 检查预训练模型
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "警告: 预训练模型不存在: $PRETRAINED_MODEL"
    echo "将从随机初始化开始训练"
    PRETRAINED_MODEL=""
fi

# 显示配置
echo ""
echo "训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  量化配置: $QUANTIZATION_CONFIG"
echo "  预训练模型: $PRETRAINED_MODEL"
echo "  激活位宽: ${ACTIVATION_BW}-bit"
echo "  权重位宽: ${PARAM_BW}-bit"
echo "  校准批次: $CALIBRATION_BATCHES"
echo "  GPU 数量: $NUM_GPUS"
echo "  GPU IDs: $GPU_IDS"
echo ""

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python"

if [ $NUM_GPUS -gt 1 ]; then
    # 多 GPU 训练
    CMD="$CMD -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29500"
    LAUNCHER="--launcher pytorch"
else
    # 单 GPU 训练
    LAUNCHER="--launcher none"
fi

CMD="$CMD basicsr/train_aimet_quantization.py"
CMD="$CMD -opt $CONFIG_FILE"
CMD="$CMD $LAUNCHER"
CMD="$CMD --activation_bw $ACTIVATION_BW"
CMD="$CMD --param_bw $PARAM_BW"
CMD="$CMD --calibration_batches $CALIBRATION_BATCHES"
CMD="$CMD --quantization_config $QUANTIZATION_CONFIG"

if [ -n "$PRETRAINED_MODEL" ]; then
    CMD="$CMD --pretrained_model $PRETRAINED_MODEL"
fi

# 显示命令
echo "执行命令:"
echo "$CMD"
echo ""

# 确认执行
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始训练..."
    eval $CMD
else
    echo "已取消"
    exit 0
fi

