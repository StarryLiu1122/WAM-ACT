# WAM-ACT Training Guide

## 环境准备

### 硬件要求
- GPU: NVIDIA A100/V100 (建议显存 >= 32GB)
- CPU: 16核心以上
- 内存: >= 64GB
- 存储: >= 500GB SSD

### 软件依赖
```bash
pip install -r requirements.txt
```

## 数据准备

### 数据格式
数据集应包含以下字段:
- `images`: [T, H, W, 3] RGB帧序列
- `actions`: [T, action_dim] 动作序列
- `states`: [T, state_dim] 状态序列
- `instruction`: str 语言指令

### 数据预处理
```python
from wam_act.data import ImagePreprocessor, ActionNormalizer

# 图像预处理
img_preprocessor = ImagePreprocessor(image_size=256)

# 动作归一化
action_normalizer = ActionNormalizer(method='minmax')
action_normalizer.fit(train_actions)
```

## 预训练

### 配置
编辑 `configs/pretrain_config.yaml`:
```yaml
model:
  image_size: 256
  latent_dim: 16
  transformer_dim: 768
  num_layers: 12

data:
  data_dir: "/path/to/data"
  batch_size: 32

train:
  pretrain_epochs: 100
  pretrain_lr: 1e-4
```

### 启动训练
```bash
# 方式1: 使用脚本
bash scripts/run_pretrain.sh

# 方式2: 使用Python
python main.py --mode pretrain --config configs/pretrain_config.yaml

# 方式3: 编程式
from wam_act.training import run_pretrain

trainer = run_pretrain(
    data_dir="/path/to/data",
    output_dir="./outputs/pretrain",
    batch_size=32,
    num_epochs=100,
)
```

### 监控训练
- 查看日志: `outputs/pretrain/train.log`
- TensorBoard: `tensorboard --logdir outputs/pretrain`

## 微调

### 配置
编辑 `configs/finetune_config.yaml`:
```yaml
train:
  pretrained_path: "./outputs/pretrain/checkpoints/best.pt"
  finetune_epochs: 50
  finetune_lr: 1e-5
  freeze_encoder: true
```

### 启动训练
```bash
# 方式1: 使用脚本
bash scripts/run_finetune.sh

# 方式2: 使用Python
python main.py --mode finetune --config configs/finetune_config.yaml
```

### 冻结策略
- `freeze_encoder=true`: 冻结VAE编码器、文本编码器、状态编码器和部分Transformer层
- 只训练动作头和未来帧预测头
- 适合小数据集

## 评估

### 策略评估
```bash
python main.py --mode eval --checkpoint outputs/finetune/checkpoints/best.pt
```

### 世界模型评估
```python
from wam_act.evaluation import WorldModelEvaluator

evaluator = WorldModelEvaluator(model)

# 单步评估
single_step = evaluator.evaluate_single_step(val_loader)

# 多步rollout
rollout = evaluator.evaluate_multi_step_rollout(val_loader, rollout_length=8)

# 动作条件评估
action_cond = evaluator.evaluate_action_conditioned(val_loader)
```

## 推理

### 单步推理
```python
from wam_act.models import WAMACT

model = WAMACT(...)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

predictions = model.predict(
    current_image=obs,
    state=proprio,
    num_denoising_steps=10,
)

actions = predictions['pred_actions']
future_images = predictions['pred_images']
```

### 流式推理
```python
# 初始化KV-Cache
model.transformer.blocks[0].attn.clear_cache()

for step in range(max_steps):
    pred = model.predict(current_image, num_denoising_steps=5)
    action = pred['pred_actions'][0, 0]

    # 执行动作
    obs = env.step(action)

    # 更新KV-Cache
    current_image = obs['image']
```

## 调参建议

### 预训练阶段
- 学习率: 1e-4 ~ 1e-3
- Batch size: 尽可能大 (受显存限制)
- 训练轮数: 100-500 epochs
- 数据量: >= 100K帧

### 微调阶段
- 学习率: 1e-5 ~ 1e-4 (比预训练小10倍)
- Batch size: 16-32
- 训练轮数: 50-200 epochs
- 数据量: >= 10K (帧, 动作)对

### 关键超参数
- `prediction_stride`: 稀疏预测步长 (4-8)
- `chunk_size`: Action Chunk长度 (8-32)
- `history_len`: 历史帧长度 (2-8)
- `num_flow_steps`: Flow Matching推理步数 (10-50)

## 常见问题

### Q: 显存不足怎么办?
A: 
1. 减小batch_size
2. 减小image_size (256 -> 128)
3. 减小transformer_dim (768 -> 512)
4. 使用梯度累积

### Q: 预训练损失不下降?
A:
1. 检查数据预处理是否正确
2. 降低学习率
3. 增加warmup步数
4. 检查VAE编码器是否正常工作

### Q: 微调时动作预测不准确?
A:
1. 确保预训练权重加载正确
2. 尝试不冻结编码器 (freeze_encoder=false)
3. 增加微调数据量
4. 调整action_loss_weight

### Q: 推理速度慢?
A:
1. 减少num_denoising_steps (10 -> 5)
2. 使用KV-Cache流式推理
3. 量化模型 (INT8/FP16)
4. 使用TensorRT加速
