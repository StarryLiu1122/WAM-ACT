#!/bin/bash
# WAM-ACT Evaluation Script
# 评估启动脚本

set -e

# 默认配置
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./outputs/finetune/checkpoints/best.pt"}
DATA_DIR=${DATA_DIR:-"./data/robot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/eval"}
DEVICE=${DEVICE:-"cuda"}

# 检查检查点是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行评估
echo "Starting WAM-ACT Evaluation..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

python main.py \
    --mode eval \
    --config "./configs/finetune_config.yaml" \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"