"""
Action-Aware Token Routing (AATR)
核心创新: 动作感知的动态Token路由机制

灵感来源: Motus的MoT架构 + GigaWorld的共享Transformer设计
创新点: 通过可学习门控动态选择注意力头，实现动作-视觉深度耦合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ActionAwareTokenRouter(nn.Module):
    """
    动作感知Token路由器

    功能:
    1. 为动作Token和未来视觉Token分配不同的注意力路由策略
    2. 通过门控机制动态选择专家注意力头
    3. 实现跨模态信息的高效融合

    架构参考:
    - Motus (Bi et al., 2025): Mixture-of-Transformers思想
    - GigaWorld-Policy (2026): 共享Transformer + 模态特定位置编码
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = dim // num_heads

        # 路由门控网络 - 根据Token类型和特征动态选择专家
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, num_experts),
        )

        # 专家注意力参数 - 每个专家有独立的Q/K/V投影
        # 这比MoE更轻量，因为共享了大部分参数
        self.expert_q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.expert_k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.expert_v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])

        # 输出投影 - 所有专家共享
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 模态类型嵌入 - 区分动作/视觉/指令Token
        self.modal_type_embed = nn.Embedding(4, dim)  # 0:指令, 1:视觉, 2:状态, 3:动作

        # 可学习的温度参数控制路由锐度
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(
        self,
        x: torch.Tensor,
        modal_types: torch.Tensor,  # [batch, seq_len] 模态类型索引
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim] 输入Token
            modal_types: [batch, seq_len] 模态类型 (0=指令, 1=视觉, 2=状态, 3=动作)
            attn_mask: 注意力掩码
            is_causal: 是否使用因果掩码

        Returns:
            output: [batch, seq_len, dim] 路由后的输出
            routing_weights: [batch, seq_len, num_experts] 路由权重(用于分析)
        """
        batch_size, seq_len, dim = x.shape

        # 添加模态类型嵌入
        modal_embed = self.modal_type_embed(modal_types)  # [B, S, D]
        x_enhanced = x + modal_embed

        # 计算路由权重
        router_logits = self.router(x_enhanced)  # [B, S, num_experts]
        routing_weights = F.softmax(router_logits / self.temperature, dim=-1)  # [B, S, num_experts]

        # Top-k路由 - 只选择最重要的k个专家
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 重新归一化

        # 为每个专家计算注意力输出
        expert_outputs = []
        for i in range(self.num_experts):
            q = self.expert_q_projs[i](x_enhanced)
            k = self.expert_k_projs[i](x_enhanced)
            v = self.expert_v_projs[i](x_enhanced)

            # 重塑为多头格式
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 因果自注意力
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device), diagonal=1
                ).bool()
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
                if attn_mask is not None:
                    attn_mask = attn_mask | causal_mask
                else:
                    attn_mask = causal_mask

            # 缩放点积注意力
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            expert_out = torch.matmul(attn_probs, v)  # [B, H, S, D_h]
            expert_out = expert_out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            expert_outputs.append(expert_out)

        # 根据路由权重聚合专家输出
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, :, i]  # [B, S]
            weight = topk_weights[:, :, i:i+1]    # [B, S, 1]

            # 为每个位置选择对应的专家输出
            for b in range(batch_size):
                for s in range(seq_len):
                    e_idx = expert_idx[b, s].item()
                    output[b, s] += weight[b, s, 0] * expert_outputs[e_idx][b, s]

        # 最终投影
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, routing_weights


class ModalTypeEncoder:
    """
    模态类型编码器 - 为不同输入类型分配类型索引
    """
    INSTRUCTION = 0
    VISION = 1
    STATE = 2
    ACTION = 3

    @classmethod
    def create_modal_tensor(
        cls,
        instr_len: int,
        vision_len: int,
        state_len: int,
        action_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """创建模态类型张量"""
        types = []
        types.extend([cls.INSTRUCTION] * instr_len)
        types.extend([cls.VISION] * vision_len)
        types.extend([cls.STATE] * state_len)
        types.extend([cls.ACTION] * action_len)
        return torch.tensor(types, device=device)
