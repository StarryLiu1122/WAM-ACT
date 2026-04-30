# WAM-ACT Architecture Details

## 系统架构概览

```
Input Layer
    |
    v
[VAE Encoder] + [Text Encoder] + [State Encoder]
    |
    v
Adaptive Causal Transformer (ACT)
    |
    +---> [Pretrain Output] Next Frame Latent
    |         |
    |         v
    |     [VAE Decoder] -> Predicted Image
    |
    +---> [Finetune Output] Action Chunk + Future Latents
              |
              +---> [Flow Matching Head] -> Actions
              |
              +---> [Future Pred Head] -> Future Images
```

## 核心组件详解

### 1. VAE Encoder/Decoder

**功能**: 将RGB图像压缩到Latent空间

**输入**: [B, 3, H, W]
**输出**: [B, latent_dim, H/16, W/16]

**架构**:
- 4层下采样卷积 (stride=2)
- 每层: Conv -> BatchNorm -> SiLU -> Conv -> BatchNorm -> SiLU
- 最终投影到latent_dim通道
- 重参数化: mu + eps * exp(0.5 * logvar)

**多视角支持**:
- 支持num_views个视角拼接
- Compose(o_left, o_front, o_right) -> [B, 3*num_views, H, W]

### 2. Adaptive Causal Transformer

**功能**: 统一处理多模态Token序列

**Token序列格式**:
```
[Instruction Tokens] [History Vision Tokens] [Current Vision Tokens] [State Tokens] [Action Tokens] [Future Vision Tokens]
```

**关键特性**:
- **RoPE位置编码**: 旋转位置编码，支持外推
- **AdaLN-Zero**: 条件LayerNorm，使用噪声水平/时间步作为条件
- **Action-Aware Token Routing**: 动态选择注意力头
- **KV-Cache**: 流式推理支持

**Transformer Block结构**:
```
x -> AdaLN-Zero -> Causal Self-Attention -> Gate*Residual -> AdaLN-Zero -> MLP -> Gate*Residual
```

### 3. Diffusion Forcing

**核心思想**: 每帧独立噪声水平

**训练**:
- 历史帧: 低噪声/零噪声 (作为条件)
- 未来帧: 随机噪声水平
- 模型学习从混合噪声序列预测干净帧

**推理**:
- 自回归生成
- 从高噪声逐步去噪
- 支持任意长度rollout

**稀疏预测**:
- 只预测stride步长的关键帧
- 减少计算量和冗余监督

### 4. Flow Matching Action Head

**核心思想**: 学习从噪声到动作分布的向量场

**训练**:
- 采样流时间 t ~ Uniform(0, 1)
- 线性插值: x_t = (1-t)*x_0 + t*x_1
- 预测速度场: v_t = dx_t/dt
- 损失: MSE(v_pred, v_target)

**推理**:
- 欧拉ODE求解
- 通常10-50步即可
- 比扩散模型更快更稳定

### 5. Action-Aware Token Routing

**核心思想**: 动作Token动态路由到不同专家

**机制**:
- 路由网络: Linear -> GELU -> Linear -> Softmax
- Top-k选择: 选择最重要的k个专家
- 专家注意力: 每个专家有独立的Q/K/V投影
- 模态嵌入: 区分指令/视觉/状态/动作Token

## 训练流程

### 阶段一: 预训练

**目标**: 学习世界动态模型

**数据**: 视频帧序列
**损失**: MSE(预测Latent, 目标Latent)

**流程**:
1. 编码当前帧和历史帧
2. 对下一帧添加噪声
3. Transformer预测去噪后的Latent
4. 计算MSE损失
5. 反向传播更新所有参数

### 阶段二: 微调

**目标**: 学习动作-视觉耦合策略

**数据**: (帧, 动作)对
**损失**: 
  - 动作损失 (Flow Matching)
  - 未来帧损失 (稀疏MSE)
  - 一致性损失 (动作-视觉耦合)

**流程**:
1. 加载预训练权重
2. 冻结编码器和部分Transformer层
3. 添加动作Token和未来帧Token
4. 联合训练动作预测和帧预测
5. 使用复合损失优化

## 推理流程

### 单步推理

1. 编码当前观测
2. 构建Token序列 (包含动作查询Token)
3. 逐步去噪未来帧Token
4. 提取动作输出
5. 解码未来帧

### 流式推理

1. 维护KV-Cache
2. 只编码新帧
3. 复用历史KV
4. 快速预测下一步

## 创新点总结

1. **Diffusion Forcing**: 独立噪声调度，任意长度rollout
2. **Action-Aware Routing**: 动态注意力路由
3. **Sparse Prediction**: 稀疏关键帧预测
4. **Bidirectional Consistency**: 动作-视觉互相约束
5. **Streaming Inference**: KV-Cache实时推理
