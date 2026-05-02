"""
WAM-ACT: World-Action Model with Adaptive Causal Transformer
基于图像生成的世界动作模型

核心架构:
1. VAE编码器: 图像 -> Latent Tokens
2. Adaptive Causal Transformer: 多模态Token统一处理
3. Diffusion Forcing: 预训练阶段预测下一帧
4. Flow Matching Action Head: 微调阶段预测Action Chunk
5. VAE解码器: Latent -> 图像

训练策略:
- 阶段一 (预训练): 仅使用视觉重建损失，训练世界模型
- 阶段二 (微调): 添加动作预测和一致性损失，训练策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .vae_encoder import VAEEncoder, VAEDecoder
from .adaptive_transformer import AdaptiveCausalTransformer
from .diffusion_forcing import DiffusionForcingTrainer, NoiseScheduler
from .flow_matching_head import FlowMatchingActionHead, ActionChunkConsistencyLoss
from .token_routing import ModalTypeEncoder


class WAMACT(nn.Module):
    """
    WAM-ACT主模型

    输入:
    - 当前观测图像 $o_t$ (RGB)
    - 语言指令 $l$ (Text)
    - 本体感知状态 $s_t$ (Proprioception)
    - 历史帧序列 $\{o_{t-H:t}\}$

    预训练输出:
    - 下一帧Latent $\hat{z}_{t+1}$

    微调输出:
    - Action Chunk $\hat{A}_{t:t+K}$
    - 未来帧Latent $\{\hat{z}_{t+k}\}_{k=1}^{K}$
    """

    def __init__(
        self,
        # VAE参数
        image_size: int = 256,
        latent_dim: int = 16,
        vae_hidden_dims: List[int] = [128, 256, 512, 1024],

        # Transformer参数
        transformer_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 4096,

        # 任务参数
        action_dim: int = 7,
        chunk_size: int = 16,
        history_len: int = 4,
        prediction_stride: int = 4,

        # 训练参数
        num_diffusion_steps: int = 1000,
        num_flow_steps: int = 50,

        # 多视角
        num_views: int = 1,
    ):
        super().__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.history_len = history_len
        self.prediction_stride = prediction_stride
        self.num_views = num_views

        # ========== 编码器 ==========
        self.vae_encoder = VAEEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            image_size=image_size,
            num_views=num_views,
        )

        # 文本编码器 (简单版本，实际可用CLIP/T5)
        self.text_encoder = nn.Embedding(50000, transformer_dim)

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, transformer_dim),
        )

        # ========== Transformer核心 ==========
        self.transformer = AdaptiveCausalTransformer(
            dim=transformer_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_seq_len=max_seq_len,
            latent_token_len=(image_size // 16) ** 2,  # 假设16倍下采样
            use_router=True,
        )

        # ========== 预训练组件 ==========
        self.noise_scheduler = NoiseScheduler(
            num_steps=num_diffusion_steps,
            schedule='cosine',
        )

        # 视觉预测头 (用于预训练)
        self.vision_pred_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.SiLU(),
            nn.Linear(transformer_dim * 2, latent_dim),
        )

        # ========== 微调组件 ==========
        # 动作预测头 (Flow Matching)
        self.action_head = FlowMatchingActionHead(
            token_dim=transformer_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_flow_steps=num_flow_steps,
        )

        # 未来帧预测头 (用于微调时的视觉监督)
        self.future_pred_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.SiLU(),
            nn.Linear(transformer_dim * 2, latent_dim),
        )

        # 一致性损失
        self.consistency_loss = ActionChunkConsistencyLoss(action_dim=action_dim)

        # ========== 解码器 ==========
        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,
            out_channels=3,
            hidden_dims=vae_hidden_dims[::-1],
            image_size=image_size,
            num_views=num_views,
        )

        # 模态类型编码器
        self.modal_encoder = ModalTypeEncoder()

    def encode_inputs(
        self,
        current_image: torch.Tensor,  # [B, 3, H, W]
        instruction_tokens: Optional[torch.Tensor] = None,  # [B, L]
        state: Optional[torch.Tensor] = None,  # [B, action_dim]
        history_images: Optional[torch.Tensor] = None,  # [B, H, 3, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        编码所有输入为Token

        Returns:
            encoded: 包含各种Token的字典
        """
        B = current_image.shape[0]
        device = current_image.device

        encoded = {}

        # 编码当前图像 -> Latent Tokens
        current_latent, _, _ = self.vae_encoder(current_image)
        current_tokens = current_latent.view(B, self.latent_dim, -1).transpose(1, 2)  # [B, H*W, C]
        encoded['vision_current'] = current_tokens

        # 编码历史图像
        if history_images is not None:
            B, H, C, H_img, W = history_images.shape
            history_flat = history_images.view(B * H, C, H_img, W)
            history_latent, _, _ = self.vae_encoder(history_flat)
            history_tokens = history_latent.view(B, H, self.latent_dim, -1).transpose(-2, -1)  # [B, H, H*W, C]
            encoded['vision_history'] = history_tokens
        else:
            encoded['vision_history'] = None

        # 编码指令
        if instruction_tokens is not None:
            instr_embed = self.text_encoder(instruction_tokens)  # [B, L, D]
            encoded['instruction'] = instr_embed
        else:
            encoded['instruction'] = None

        # 编码状态
        if state is not None:
            state_tokens = self.state_encoder(state).unsqueeze(1)  # [B, 1, D]
            encoded['state'] = state_tokens
        else:
            encoded['state'] = None

        return encoded

    def forward_pretrain(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        instruction_tokens: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        history_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预训练前向传播

        目标: 预测下一帧图像

        Args:
            current_image: [B, 3, H, W] 当前帧
            next_image: [B, 3, H, W] 下一帧 (目标)
            instruction_tokens: [B, L] 指令
            state: [B, action_dim] 状态
            history_images: [B, H, 3, H, W] 历史帧

        Returns:
            pred_next_latent: [B, latent_dim, H', W'] 预测的下一帧Latent
            target_next_latent: [B, latent_dim, H', W'] 目标下一帧Latent
        """
        B = current_image.shape[0]
        device = current_image.device

        # 编码输入
        encoded = self.encode_inputs(current_image, instruction_tokens, state, history_images)

        # 编码目标帧 (用于计算损失)
        with torch.no_grad():
            target_latent, _, _ = self.vae_encoder(next_image)

        # 准备Transformer输入
        # 预训练时，未来帧位置用噪声填充
        noise_level = torch.randint(0, self.noise_scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(target_latent)
        noisy_target = self.noise_scheduler.add_noise(target_latent, noise, noise_level)
        noisy_tokens = noisy_target.view(B, self.latent_dim, -1).transpose(1, 2)

        # 构建Token序列
        tokens, modal_types = self.transformer.prepare_multimodal_tokens(
            instruction_tokens=encoded['instruction'],
            vision_tokens=encoded['vision_history'],
            state_tokens=encoded['state'],
            future_tokens=noisy_tokens.unsqueeze(1),  # [B, 1, H*W, C]
        )

        # 通过Transformer
        cond = (noise_level.float() / self.noise_scheduler.num_steps).unsqueeze(-1)  # [B, 1]
        output, _ = self.transformer(tokens, cond, modal_types)

        # 提取未来帧对应的输出
        # 未来帧Tokens在序列末尾
        future_output = output[:, -noisy_tokens.shape[1]:]  # [B, H*W, D]

        # 预测去噪后的Latent
        pred_next_flat = self.vision_pred_head(future_output)  # [B, H*W, latent_dim]
        pred_next_latent = pred_next_flat.transpose(1, 2).view(B, self.latent_dim, 
                                                               self.image_size // 16, 
                                                               self.image_size // 16)

        return pred_next_latent, target_latent

    def forward_finetune(
        self,
        current_image: torch.Tensor,
        target_actions: torch.Tensor,  # [B, chunk_size, action_dim]
        target_future_images: Optional[torch.Tensor] = None,  # [B, chunk_size, 3, H, W]
        instruction_tokens: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        history_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        微调前向传播

        目标: 同步预测Action Chunk和未来帧

        Args:
            current_image: [B, 3, H, W]
            target_actions: [B, chunk_size, action_dim]
            target_future_images: [B, chunk_size, 3, H, W] (用于监督)
            instruction_tokens: [B, L]
            state: [B, action_dim]
            history_images: [B, H, 3, H, W]

        Returns:
            outputs: 包含预测结果和损失的字典
        """
        B = current_image.shape[0]
        device = current_image.device

        # 编码输入
        encoded = self.encode_inputs(current_image, instruction_tokens, state, history_images)

        # 编码目标未来帧 (用于视觉监督)
        if target_future_images is not None:
            B, K, C, H, W = target_future_images.shape
            future_flat = target_future_images.view(B * K, C, H, W)
            with torch.no_grad():
                future_latents, _, _ = self.vae_encoder(future_flat)
            future_latents = future_latents.view(B, K, self.latent_dim, -1).transpose(-2, -1)  # [B, K, H*W, C]
        else:
            future_latents = None

        # 构建Token序列
        # 动作Token: 使用可学习的动作查询Token
        action_queries = nn.Parameter(torch.randn(1, self.chunk_size, self.action_dim, device=device))
        action_queries = action_queries.expand(B, -1, -1)

        # 未来帧Token: 用噪声初始化
        if future_latents is not None:
            noise_level = torch.randint(0, self.noise_scheduler.num_steps, (B, self.chunk_size), device=device)
            noise = torch.randn_like(future_latents)
            noisy_future = self.noise_scheduler.add_noise(future_latents, noise, noise_level)
        else:
            noisy_future = torch.randn(B, self.chunk_size, (self.image_size // 16) ** 2, 
                                       self.latent_dim, device=device)
            noise_level = torch.full((B, self.chunk_size), self.noise_scheduler.num_steps - 1, device=device)

        tokens, modal_types = self.transformer.prepare_multimodal_tokens(
            instruction_tokens=encoded['instruction'],
            vision_tokens=encoded['vision_history'],
            state_tokens=encoded['state'],
            action_tokens=action_queries,
            future_tokens=noisy_future,
        )

        # 通过Transformer
        cond = (noise_level.float().mean(dim=1, keepdim=True) / self.noise_scheduler.num_steps)  # [B, 1]
        output, aux = self.transformer(tokens, cond, modal_types)

        # ========== 提取输出 ==========

        # 1. 动作输出
        # 找到动作Token的位置
        action_mask = (modal_types == self.modal_encoder.ACTION)
        action_output = output[action_mask].view(B, self.chunk_size, -1)  # [B, chunk_size, D]

        # 通过Flow Matching Head预测动作
        pred_actions, action_loss = self.action_head(action_output, target_actions, is_training=True)

        # 2. 未来帧输出
        future_mask = (modal_types == self.modal_encoder.VISION)
        # 区分历史视觉和未来视觉 (未来视觉在序列末尾)
        future_output = output[:, -noisy_future.shape[1] * noisy_future.shape[2]:]  # 简化处理
        future_output = future_output.view(B, self.chunk_size, -1, self.transformer.dim)

        pred_future_flat = self.future_pred_head(future_output)  # [B, K, H*W, latent_dim]

        # 3. 计算损失
        losses = {}

        # 动作损失 (Flow Matching)
        losses['action'] = action_loss if action_loss is not None else torch.tensor(0.0, device=device)

        # 未来帧重建损失 (MSE)
        if future_latents is not None:
            pred_future_latents = pred_future_flat.view(B, self.chunk_size, self.latent_dim, -1).transpose(-2, -1)
            target_future_latents = future_latents.view(B, self.chunk_size, self.latent_dim, -1).transpose(-2, -1)

            # 稀疏帧损失 (只计算stride步长的帧)
            sparse_indices = list(range(0, self.chunk_size, self.prediction_stride))
            losses['future'] = F.mse_loss(
                pred_future_latents[:, sparse_indices],
                target_future_latents[:, sparse_indices],
            )
        else:
            losses['future'] = torch.tensor(0.0, device=device)

        # 动作-视觉一致性损失
        if future_latents is not None:
            pred_future_images = self.vae_decoder.decode_from_tokens(
                pred_future_flat.view(B * self.chunk_size, -1, self.latent_dim),
                self.image_size // 16,
                self.image_size // 16,
            ).view(B, self.chunk_size, 3, self.image_size, self.image_size)

            losses['consistency'] = self.consistency_loss(
                pred_actions,
                pred_future_latents,
                target_future_latents,
            )
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)

        # 总损失
        total_loss = (losses['action'] + 
                     0.5 * losses['future'] + 
                     0.1 * losses['consistency'])

        return {
            'pred_actions': pred_actions,
            'pred_future_latents': pred_future_flat if future_latents is not None else None,
            'losses': losses,
            'total_loss': total_loss,
            'routing_weights': aux.get('routing_weights'),
        }

    @torch.no_grad()
    def predict(
        self,
        current_image: torch.Tensor,
        instruction_tokens: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        history_images: Optional[torch.Tensor] = None,
        num_denoising_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        推理: 预测Action Chunk和下一帧图像

        Args:
            current_image: [B, 3, H, W]
            instruction_tokens: [B, L]
            state: [B, action_dim]
            history_images: [B, H, 3, H, W]
            num_denoising_steps: 去噪步数

        Returns:
            predictions: 包含动作和图像预测
        """
        B = current_image.shape[0]
        device = current_image.device

        # 编码输入
        encoded = self.encode_inputs(current_image, instruction_tokens, state, history_images)

        # 构建Token序列 (无目标信息)
        action_queries = nn.Parameter(torch.randn(1, self.chunk_size, self.action_dim, device=device))
        action_queries = action_queries.expand(B, -1, -1)

        # 初始化未来帧为噪声
        noisy_future = torch.randn(
            B, self.chunk_size, (self.image_size // 16) ** 2, self.latent_dim,
            device=device,
        )

        tokens, modal_types = self.transformer.prepare_multimodal_tokens(
            instruction_tokens=encoded['instruction'],
            vision_tokens=encoded['vision_history'],
            state_tokens=encoded['state'],
            action_tokens=action_queries,
            future_tokens=noisy_future,
        )

        # 逐步去噪 (类似DDIM)
        for step in range(num_denoising_steps - 1, -1, -1):
            t = torch.full((B, 1), step / num_denoising_steps, device=device)

            output, _ = self.transformer(tokens, t, modal_types, use_cache=True)

            # 更新未来帧Token
            future_output = output[:, -noisy_future.shape[1] * noisy_future.shape[2]:]
            pred_noise = self.future_pred_head(future_output.view(B, self.chunk_size, -1, self.transformer.dim))

            # DDIM去噪
            alpha = (step / num_denoising_steps) ** 2
            alpha_prev = ((step - 1) / num_denoising_steps) ** 2 if step > 0 else 1.0

            noisy_future = (noisy_future - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            noisy_future = alpha_prev.sqrt() * noisy_future + (1 - alpha_prev).sqrt() * torch.randn_like(noisy_future)

            # 更新Token序列
            tokens, modal_types = self.transformer.prepare_multimodal_tokens(
                instruction_tokens=encoded['instruction'],
                vision_tokens=encoded['vision_history'],
                state_tokens=encoded['state'],
                action_tokens=action_queries,
                future_tokens=noisy_future,
            )

        # 最终预测
        output, _ = self.transformer(tokens, torch.zeros(B, 1, device=device), modal_types)

        # 提取动作
        action_mask = (modal_types == self.modal_encoder.ACTION)
        action_output = output[action_mask].view(B, self.chunk_size, -1)
        pred_actions = self.action_head.predict_action_chunk(action_output, num_steps=10)

        # 提取未来帧并解码
        future_output = output[:, -noisy_future.shape[1] * noisy_future.shape[2]:]
        pred_future_flat = self.future_pred_head(future_output.view(B, self.chunk_size, -1, self.transformer.dim))
        pred_future_latents = pred_future_flat.view(B, self.chunk_size, self.latent_dim, -1).transpose(-2, -1)

        # 解码图像
        pred_images = []
        for k in range(self.chunk_size):
            img = self.vae_decoder.decode(pred_future_latents[:, k])
            pred_images.append(img)
        pred_images = torch.stack(pred_images, dim=1)

        return {
            'pred_actions': pred_actions,
            'pred_images': pred_images,
            'pred_latents': pred_future_latents,
        }

    def get_pretrain_loss(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        """预训练阶段的损失"""
        return F.mse_loss(pred_latent, target_latent)

    def get_finetune_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        pred_future: Optional[torch.Tensor] = None,
        target_future: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """微调阶段的复合损失"""
        losses = {}

        # 动作损失
        losses['action'] = F.mse_loss(pred_actions, target_actions)

        # 视觉损失
        if pred_future is not None and target_future is not None:
            losses['future'] = F.mse_loss(pred_future, target_future)

        # 总损失
        losses['total'] = losses['action'] + 0.5 * losses.get('future', 0)

        return losses
