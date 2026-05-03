#!/bin/bash
# WAM-ACT Finetune Script
# 微调启动脚本

set -e

# 默认配置
DATA_DIR=${DATA_DIR:-"./data/robot"}
PRETRAINED_PATH=${PRETRAINED_PATH:-"./outputs/pretrain/checkpoints/best.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/finetune"}
DEVICE=${DEVICE:-"cuda"}

# 检查预训练权重是否存在
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAINED_PATH"
    echo "Please run pretraining first: ./run_pretrain.sh"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行微调
echo "Starting WAM-ACT Finetuning..."
echo "Data dir: $DATA_DIR"
echo "Pretrained: $PRETRAINED_PATH"
echo "Output dir: $OUTPUT_DIR"

python main.py \
    --mode finetune \
    --config "./configs/finetune_config.yaml" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "Finetuning completed!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
