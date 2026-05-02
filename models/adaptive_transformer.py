"""
Adaptive Causal Transformer (ACT)
核心创新: 融合Diffusion Forcing + Flow Matching的多模态因果Transformer

参考:
- DiT (Peebles & Xie, 2023): 扩散Transformer
- MOTUS (Bi et al., 2025): 流匹配Transformer
- GigaWorld-Policy (2026): 共享Transformer + 模态特定编码
- CogACT (Zhao et al., 2025): 基于DiT的动作预测

创新点:
1. 多模态Token统一处理 (指令/视觉/状态/动作/未来)
2. AdaLN-Zero条件调制 (来自DiT)
3. 因果注意力 + KV-Cache流式推理
4. 稀疏未来帧Token表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (DiT)

    使用条件(如噪声水平、时间步)生成scale和shift参数
    并初始化残差连接的权重为0
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim

        # 条件投影到6*dim: scale, shift, gate (各2*dim用于MLP)
        self.scale_shift_table = nn.Linear(cond_dim, 6 * dim, bias=True)

        # 初始化gate为0，确保残差连接开始时不起作用
        nn.init.zeros_(self.scale_shift_table.weight)
        nn.init.zeros_(self.scale_shift_table.bias)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
            cond: [B, cond_dim]

        Returns:
            normed_x: 归一化后的x
            scale, shift, gate: 用于调制和门控
        """
        # 条件投影
        params = self.scale_shift_table(cond)  # [B, 6*D]

        # 分割参数
        B = x.shape[0]
        params = params.view(B, 6, self.dim)

        scale_msa = params[:, 0]  # [B, D] - MHA scale
        shift_msa = params[:, 1]  # [B, D] - MHA shift
        gate_msa = params[:, 2]   # [B, D] - MHA gate

        scale_mlp = params[:, 3]  # [B, D] - MLP scale
        shift_mlp = params[:, 4]  # [B, D] - MLP shift
        gate_mlp = params[:, 5]   # [B, D] - MLP gate

        # 归一化
        normed_x = self.norm(x)

        return normed_x, scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp


class MultiHeadCausalAttention(nn.Module):
    """
    多头因果自注意力

    支持:
    1. 标准因果掩码 (防止看到未来信息)
    2. 模态特定注意力 (通过ActionAwareTokenRouter)
    3. KV-Cache支持 (流式推理)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_router: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_router = use_router

        if use_router:
            from .token_routing import ActionAwareTokenRouter
            self.router = ActionAwareTokenRouter(
                dim=dim,
                num_heads=num_heads,
                num_experts=4,
                top_k=2,
                dropout=dropout,
            )
        else:
            # 标准注意力
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        # KV-Cache (用于流式推理)
        self.k_cache = None
        self.v_cache = None
        self.cache_length = 0

    def forward(
        self,
        x: torch.Tensor,
        modal_types: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, S, D]
            modal_types: [B, S] 模态类型
            attn_mask: 注意力掩码
            use_cache: 是否使用KV-Cache

        Returns:
            output: [B, S, D]
            routing_weights: [B, S, num_experts] (如果使用router)
        """
        if self.use_router and modal_types is not None:
            output, routing_weights = self.router(x, modal_types, attn_mask, is_causal=True)
            return output, routing_weights
        else:
            # 标准多头注意力
            B, S, D = x.shape

            q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            # KV-Cache
            if use_cache and self.k_cache is not None:
                k = torch.cat([self.k_cache, k], dim=2)
                v = torch.cat([self.v_cache, v], dim=2)
                self.k_cache = k
                self.v_cache = v
            elif use_cache:
                self.k_cache = k
                self.v_cache = v

            # 因果掩码
            if attn_mask is None:
                causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(attn_mask, float('-inf'))
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            out = torch.matmul(attn_probs, v)
            out = out.transpose(1, 2).contiguous().view(B, S, D)
            out = self.out_proj(out)
            out = self.dropout(out)

            return out, None

    def clear_cache(self):
        """清除KV-Cache"""
        self.k_cache = None
        self.v_cache = None
        self.cache_length = 0


class TransformerBlock(nn.Module):
    """
    Transformer块

    结构: AdaLN-Zero -> Attention -> Residual -> AdaLN-Zero -> MLP -> Residual
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cond_dim: int = 256,
        use_router: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # AdaLN-Zero调制
        self.adaln = AdaLNZero(dim, cond_dim)

        # 注意力
        self.attn = MultiHeadCausalAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_router=use_router,
        )

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        modal_types: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, S, D]
            cond: [B, cond_dim] 条件向量 (如噪声水平、时间步)
            modal_types: [B, S] 模态类型
            attn_mask: 注意力掩码
            use_cache: 是否使用KV-Cache

        Returns:
            output: [B, S, D]
            routing_weights: [B, S, num_experts]
        """
        # AdaLN-Zero调制
        normed_x, scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaln(x, cond)

        # 调制并应用注意力
        modulated = normed_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, routing_weights = self.attn(modulated, modal_types, attn_mask, use_cache)

        # 门控残差连接
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        normed_x2 = self.adaln.norm(x)  # 重新归一化
        modulated2 = normed_x2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(modulated2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x, routing_weights


class AdaptiveCausalTransformer(nn.Module):
    """
    自适应因果Transformer

    核心组件:
    1. 多模态Token嵌入层
    2. 位置编码
    3. 堆叠的TransformerBlock
    4. 模态特定的输出头
    """

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        num_modalities: int = 4,  # 指令, 视觉, 状态, 动作
        latent_token_len: int = 256,  # 每帧Latent的Token数 (16x16)
        use_router: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.latent_token_len = latent_token_len

        # 模态嵌入 - 为每种模态分配不同的嵌入空间
        self.modal_embeds = nn.ModuleDict({
            'instruction': nn.Embedding(50000, dim),  # 词汇表大小
            'vision': nn.Linear(16, dim),  # Latent通道 -> dim
            'state': nn.Linear(7, dim),  # 状态维度 -> dim (假设7DoF)
            'action': nn.Linear(7, dim),  # 动作维度 -> dim
        })

        # 位置编码 - 使用RoPE (旋转位置编码)
        self.pos_embed = RotaryPositionEmbedding(dim // num_heads, max_seq_len)

        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                cond_dim=256,
                use_router=use_router,
            )
            for _ in range(num_layers)
        ])

        # 最终归一化
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)

        # 条件编码器 - 将噪声水平/时间步编码为条件向量
        self.cond_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(
        self,
        tokens: torch.Tensor,  # [B, S, D] 或各种模态的拼接
        cond: torch.Tensor,  # [B, 1] 条件 (噪声水平/时间步)
        modal_types: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            tokens: [B, S, D] 输入Token
            cond: [B, 1] 条件值
            modal_types: [B, S] 模态类型
            attn_mask: 注意力掩码
            use_cache: 是否使用KV-Cache

        Returns:
            output: [B, S, D]
            aux_outputs: 辅助输出 (如路由权重)
        """
        B, S, D = tokens.shape
        device = tokens.device

        # 添加位置编码
        positions = torch.arange(S, device=device)
        tokens = self.pos_embed(tokens, positions)

        # 编码条件
        cond_vec = self.cond_encoder(cond)  # [B, 256]

        # 通过Transformer块
        routing_weights_all = []
        x = tokens

        for block in self.blocks:
            x, routing_weights = block(x, cond_vec, modal_types, attn_mask, use_cache)
            if routing_weights is not None:
                routing_weights_all.append(routing_weights)

        x = self.final_norm(x)

        # 收集辅助输出
        aux_outputs = {
            'routing_weights': torch.stack(routing_weights_all, dim=0) if routing_weights_all else None,
        }

        return x, aux_outputs

    def prepare_multimodal_tokens(
        self,
        instruction_tokens: Optional[torch.Tensor] = None,
        vision_tokens: Optional[torch.Tensor] = None,  # [B, num_frames, latent_token_len, C]
        state_tokens: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
        future_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备多模态Token序列

        拼接顺序: [指令, 历史视觉, 当前视觉, 状态, 动作, 未来视觉]

        Returns:
            tokens: [B, S, D]
            modal_types: [B, S]
        """
        all_tokens = []
        all_types = []
        type_map = {
            'instruction': 0,
            'vision': 1,
            'state': 2,
            'action': 3,
        }

        if instruction_tokens is not None:
            B = instruction_tokens.shape[0]
            instr_embed = self.modal_embeds['instruction'](instruction_tokens)  # [B, L, D]
            all_tokens.append(instr_embed)
            all_types.extend([type_map['instruction']] * instr_embed.shape[1])

        if vision_tokens is not None:
            B, F, T, C = vision_tokens.shape  # [B, num_frames, token_len, C]
            # 展平帧和token
            vision_flat = vision_tokens.view(B, F * T, C)
            vision_embed = self.modal_embeds['vision'](vision_flat)  # [B, F*T, D]
            all_tokens.append(vision_embed)
            all_types.extend([type_map['vision']] * vision_embed.shape[1])

        if state_tokens is not None:
            state_embed = self.modal_embeds['state'](state_tokens)  # [B, L, D]
            all_tokens.append(state_embed)
            all_types.extend([type_map['state']] * state_embed.shape[1])

        if action_tokens is not None:
            action_embed = self.modal_embeds['action'](action_tokens)  # [B, L, D]
            all_tokens.append(action_embed)
            all_types.extend([type_map['action']] * action_embed.shape[1])

        if future_tokens is not None:
            B, F, T, C = future_tokens.shape
            future_flat = future_tokens.view(B, F * T, C)
            future_embed = self.modal_embeds['vision'](future_flat)
            all_tokens.append(future_embed)
            all_types.extend([type_map['vision']] * future_embed.shape[1])

        tokens = torch.cat(all_tokens, dim=1)
        modal_types = torch.tensor(all_types, device=tokens.device).unsqueeze(0).expand(B, -1)

        return tokens, modal_types


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)

    参考: RoFormer (Su et al., 2021)
    优势: 更好的外推能力，适合变长序列
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
            positions: [S] 位置索引

        Returns:
            x_rotated: [B, S, D]
        """
        # 计算旋转矩阵
        angles = torch.outer(positions.float(), self.inv_freq)  # [S, head_dim//2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 应用旋转
        # 将x分成两半
        x1, x2 = x[..., ::2], x[..., 1::2]

        # 旋转
        rotated_x1 = x1 * cos.unsqueeze(0) - x2 * sin.unsqueeze(0)
        rotated_x2 = x1 * sin.unsqueeze(0) + x2 * cos.unsqueeze(0)

        # 交错合并
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

        return rotated
