#!/bin/bash
# WAM-ACT Finetune Script
# 微调启动脚本

set -e

# 默认配置
DATA_DIR=${DATA_DIR:-"./data/robot"}
PRETRAINED_PATH=${PRETRAINED_PATH:-"./outputs/pretrain/checkpoints/best.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/finetune"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LR=${LR:-1e-5}
IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"cuda"}
FREEZE_ENCODER=${FREEZE_ENCODER:-"true"}

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
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"

python -m wam_act.training.finetune \
    --data_dir $DATA_DIR \
    --pretrained_path $PRETRAINED_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --freeze_encoder $FREEZE_ENCODER \
    --config "./configs/finetune_config.yaml"

echo "Finetuning completed!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
