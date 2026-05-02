#!/bin/bash
# WAM-ACT Evaluation Script
# 评估启动脚本

set -e

# 默认配置
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./outputs/finetune/checkpoints/best.pt"}
DATA_DIR=${DATA_DIR:-"./data/robot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/eval"}
NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-10}
ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-8}
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
echo "Output dir: $OUTPUT_DIR"

python -m wam_act.evaluation.eval_policy \
    --checkpoint_path $CHECKPOINT_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --num_eval_batches $NUM_EVAL_BATCHES \
    --device $DEVICE

python -m wam_act.evaluation.eval_world_model \
    --checkpoint_path $CHECKPOINT_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --rollout_length $ROLLOUT_LENGTH \
    --device $DEVICE

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
