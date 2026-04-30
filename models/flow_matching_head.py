"""
Flow Matching Action Head
核心创新: 使用Flow Matching替代传统扩散模型进行动作预测

参考:
- Flow Matching (Lipman et al., 2023): 直接学习向量场
- MOTUS (Bi et al., 2025): 流匹配VLA
- CogACT (Zhao et al., 2025): 基于DiT的动作预测

创新点:
1. 相比扩散模型，Flow Matching训练更稳定，推理步数更少
2. 动作Token与视觉Token在Transformer中深度耦合
3. 支持Action Chunk预测 (一次性预测K个动作)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching动作预测头

    架构:
    1. 将Transformer输出的动作Token映射为条件向量
    2. 使用Flow Matching学习从噪声到动作分布的向量场
    3. 通过ODE求解器生成最终动作

    相比扩散模型的优势:
    - 训练更稳定 (不需要噪声调度)
    - 推理更快 (通常10-50步即可)
    - 与Transformer架构更兼容
    """

    def __init__(
        self,
        token_dim: int = 768,
        action_dim: int = 7,
        chunk_size: int = 16,  # Action Chunk长度
        hidden_dim: int = 512,
        num_flow_steps: int = 50,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_flow_steps = num_flow_steps
        self.sigma_min = sigma_min

        # 条件编码器 - 从动作Token提取条件特征
        self.condition_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 时间嵌入 - 将流时间t编码
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 向量场网络 - 预测速度场 v_t(x)
        # 输入: [x_t, condition, time_embed]
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim * chunk_size + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim * chunk_size),
        )

        # 动作归一化参数 (可学习)
        self.action_scale = nn.Parameter(torch.ones(action_dim))
        self.action_bias = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        action_tokens: torch.Tensor,  # [B, chunk_size, token_dim]
        target_actions: Optional[torch.Tensor] = None,  # [B, chunk_size, action_dim] (训练时使用)
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            action_tokens: Transformer输出的动作Token
            target_actions: 目标动作 (训练时)
            is_training: 是否训练模式

        Returns:
            pred_actions: [B, chunk_size, action_dim] 预测的动作
            loss: 训练损失 (训练时)
        """
        B = action_tokens.shape[0]

        # 编码条件
        # 对chunk内的所有token取平均作为全局条件
        condition = action_tokens.mean(dim=1)  # [B, token_dim]
        condition = self.condition_encoder(condition)  # [B, hidden_dim]

        if is_training and target_actions is not None:
            # 训练: 使用条件流匹配损失
            loss = self._compute_flow_matching_loss(condition, target_actions)

            # 同时返回一个前向预测用于评估
            with torch.no_grad():
                pred_actions = self._sample_actions(condition)

            return pred_actions, loss
        else:
            # 推理: 通过ODE求解器生成动作
            pred_actions = self._sample_actions(condition)
            return pred_actions, None

    def _compute_flow_matching_loss(
        self,
        condition: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算条件流匹配损失

        目标: 学习向量场 u_t(x) = E[X_1 - X_0 | X_t = x]
        这里使用简单的线性插值: x_t = (1-t)*x_0 + t*x_1
        其中 x_0 ~ N(0, I), x_1 = target_actions
        """
        B = condition.shape[0]

        # 展平动作
        target_flat = target_actions.view(B, -1)  # [B, chunk_size * action_dim]

        # 采样流时间 t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=condition.device)

        # 采样源分布 x_0 ~ N(0, I)
        x_0 = torch.randn_like(target_flat)

        # 线性插值得到 x_t
        x_t = (1 - t) * x_0 + t * target_flat

        # 真实的速度场
        v_target = target_flat - x_0  # dx_t/dt = x_1 - x_0

        # 时间嵌入
        t_embed = self.time_embed(t.squeeze(-1))  # [B, hidden_dim]

        # 预测速度场
        v_pred = self._predict_velocity(x_t, condition, t_embed)

        # 流匹配损失: MSE
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t_embed: torch.Tensor,
    ) -> torch.Tensor:
        """预测速度场 v_t(x)"""
        # 拼接输入
        input_vec = torch.cat([x_t, condition, t_embed], dim=-1)
        v_pred = self.velocity_net(input_vec)
        return v_pred

    def _sample_actions(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        通过ODE求解器采样动作

        使用欧拉方法求解: dx/dt = v_t(x)
        """
        if num_steps is None:
            num_steps = self.num_flow_steps

        B = condition.shape[0]
        device = condition.device

        # 从源分布采样
        x = torch.randn(B, self.chunk_size * self.action_dim, device=device)

        # 欧拉积分
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_embed = self.time_embed(t)

            v = self._predict_velocity(x, condition, t_embed)
            x = x + dt * v

        # 重塑为动作格式
        actions = x.view(B, self.chunk_size, self.action_dim)

        # 反归一化
        actions = actions * self.action_scale + self.action_bias

        return actions

    @torch.no_grad()
    def predict_action_chunk(
        self,
        action_tokens: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        快速推理: 预测Action Chunk

        使用少量步数进行快速推理 (参考MOTUS的10步推理)
        """
        condition = action_tokens.mean(dim=1)
        condition = self.condition_encoder(condition)
        return self._sample_actions(condition, num_steps=num_steps)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ActionChunkConsistencyLoss(nn.Module):
    """
    Action Chunk一致性损失

    确保预测的动作块与预测的未来帧一致
    这是微调阶段的关键损失组件
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim

        # 简单的动作到视觉变化映射 (可学习)
        self.action_to_delta = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )

    def forward(
        self,
        pred_actions: torch.Tensor,  # [B, chunk_size, action_dim]
        pred_future_frames: torch.Tensor,  # [B, chunk_size, H, W, C] Latent帧
        target_future_frames: torch.Tensor,  # [B, chunk_size, H, W, C]
    ) -> torch.Tensor:
        """
        计算动作-视觉一致性损失

        核心思想: 动作应该导致视觉上的合理变化
        """
        B, K, H, W, C = pred_future_frames.shape

        # 计算帧间变化
        frame_deltas = pred_future_frames[:, 1:] - pred_future_frames[:, :-1]  # [B, K-1, H, W, C]
        target_deltas = target_future_frames[:, 1:] - target_future_frames[:, :-1]

        # 将动作映射到变化空间
        action_features = self.action_to_delta(pred_actions[:, :-1])  # [B, K-1, 256]

        # 将帧变化展平
        flat_deltas = frame_deltas.view(B, K-1, -1)  # [B, K-1, H*W*C]
        flat_target = target_deltas.view(B, K-1, -1)

        # 计算一致性: 动作特征应该能预测帧变化的方向
        # 使用余弦相似度
        action_norm = F.normalize(action_features, dim=-1)
        delta_norm = F.normalize(flat_deltas, dim=-1)
        target_delta_norm = F.normalize(flat_target, dim=-1)

        # 动作应该与真实变化方向一致
        consistency = F.cosine_embedding_loss(
            action_norm.view(-1, action_norm.shape[-1]),
            target_delta_norm.view(-1, target_delta_norm.shape[-1]),
            torch.ones(action_norm.shape[0] * action_norm.shape[1], device=action_norm.device),
        )

        return consistency
