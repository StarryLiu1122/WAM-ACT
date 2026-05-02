#!/bin/bash
# WAM-ACT Pretrain Script
# 预训练启动脚本

set -e

# 默认配置
DATA_DIR=${DATA_DIR:-"./data/robot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/pretrain"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-100}
LR=${LR:-1e-4}
IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"cuda"}

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行预训练
echo "Starting WAM-ACT Pretraining..."
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"

python -m wam_act.training.pretrain \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --config "./configs/pretrain_config.yaml"

echo "Pretraining completed!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
