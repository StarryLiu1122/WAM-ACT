
# WAM-ACT 项目代码文件清单

## 已创建的全部代码文件

### 模型架构 (models/)
1. **__init__.py** - 模型包初始化
2. **wam_act.py** - 主模型架构 (WAMACT类)
3. **adaptive_transformer.py** - 自适应因果Transformer (核心创新)
4. **diffusion_forcing.py** - Diffusion Forcing训练框架
5. **flow_matching_head.py** - Flow Matching动作预测头
6. **token_routing.py** - Action-Aware Token Routing
7. **vae_encoder.py** - VAE编码器/解码器

### 数据处理 (data/)
8. **__init__.py** - 数据包初始化
9. **robot_dataset.py** - 机器人数据集加载器
10. **data_preprocessing.py** - 数据预处理 (图像/动作/指令)
11. **data_augmentation.py** - 数据增强

### 训练 (training/)
12. **__init__.py** - 训练包初始化
13. **trainer.py** - 通用训练器基类
14. **pretrain.py** - 预训练脚本
15. **finetune.py** - 微调脚本
16. **lr_scheduler.py** - 学习率调度器

### 评估 (evaluation/)
17. **__init__.py** - 评估包初始化
18. **eval_policy.py** - 策略评估器
19. **eval_world_model.py** - 世界模型评估器
20. **metrics.py** - 通用评估指标

### 工具 (utils/)
21. **__init__.py** - 工具包初始化
22. **config.py** - 配置管理
23. **logging_utils.py** - 日志工具
24. **checkpoint.py** - 检查点管理

### 配置 (configs/)
25. **pretrain_config.yaml** - 预训练配置
26. **finetune_config.yaml** - 微调配置

### 脚本 (scripts/)
27. **run_pretrain.sh** - 预训练启动脚本
28. **run_finetune.sh** - 微调启动脚本
29. **run_eval.sh** - 评估启动脚本

### 其他
30. **main.py** - 主入口文件
31. **README.md** - 项目说明文档

## 核心创新点总结

### 1. Diffusion Forcing 预训练
- 每帧独立噪声调度
- 支持任意长度自回归rollout
- 避免固定长度限制

### 2. Action-Aware Token Routing
- 可学习路由门控
- 动态选择注意力头
- 动作-视觉深度耦合

### 3. Sparse Future Frame Prediction
- 稀疏关键帧预测
- 减少冗余监督
- 保留关键动态信息

### 4. Bidirectional Consistency Loss
- 动作-视觉互相约束
- 双向一致性

### 5. Streaming Inference with KV-Cache
- 实时流式推理
- KV-Cache压缩

## 架构图

架构图已保存至: /mnt/agents/output/wam_act_architecture_v2.png

## 使用方式

### 预训练
```bash
python main.py --mode pretrain --config configs/pretrain_config.yaml
```

### 微调
```bash
python main.py --mode finetune --config configs/finetune_config.yaml
```

### 评估
```bash
python main.py --mode eval --checkpoint outputs/finetune/checkpoints/best.pt
```

### 推理
```bash
python main.py --mode predict --checkpoint outputs/finetune/checkpoints/best.pt
```
