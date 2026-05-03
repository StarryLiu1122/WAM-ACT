"""
Diffusion Forcing Training Framework
核心创新: 独立噪声调度 + 自回归扩散训练

创新点:
1. 每帧独立噪声水平，训练时随机采样
2. 支持因果注意力掩码，实现流式推理
3. 稀疏未来帧预测，减少冗余监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class NoiseScheduler:
    """
    噪声调度器 - 为每个帧独立采样噪声水平

    这是Diffusion Forcing的核心: 不同于传统扩散模型对所有帧使用相同噪声水平，
    这里每帧有独立的噪声水平，使得模型可以处理任意混合噪声/干净帧的序列
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'cosine',
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 预计算噪声调度参数
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == 'cosine':
            timesteps = torch.arange(num_steps + 1)
            alphas_cumprod = torch.cos(
                ((timesteps / num_steps) + 0.008) / 1.008 * np.pi / 2
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        向输入添加噪声

        Args:
            x: [B, ...] 干净数据
            noise: [B, ...] 噪声
            timesteps: [B] 或 [B, seq_len] 噪声水平索引

        Returns:
            noisy_x: [B, ...] 加噪后的数据
        """
        # 将调度参数移到正确的设备
        device = x.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)

        # 处理不同形状的timesteps
        if timesteps.dim() == 1:
            # [B] -> 为每个batch采样
            t = timesteps
            sqrt_a = sqrt_alpha[t].view(-1, *([1] * (x.dim() - 1)))
            sqrt_1ma = sqrt_one_minus_alpha[t].view(-1, *([1] * (x.dim() - 1)))
        else:
            # [B, seq_len] -> 为每个序列位置独立采样
            B, S = timesteps.shape
            t = timesteps.view(-1)
            sqrt_a = sqrt_alpha[t].view(B, S, *([1] * (x.dim() - 2)))
            sqrt_1ma = sqrt_one_minus_alpha[t].view(B, S, *([1] * (x.dim() - 2)))

        return sqrt_a * x + sqrt_1ma * noise

    def sample_timesteps(self, batch_size: int, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        随机采样噪声水平

        Args:
            batch_size: batch大小
            seq_len: 序列长度 (如果为None，则每个batch一个timestep)

        Returns:
            timesteps: [B] 或 [B, seq_len]
        """
        if seq_len is None:
            return torch.randint(0, self.num_steps, (batch_size,))
        else:
            # 为序列中每个位置独立采样噪声水平
            return torch.randint(0, self.num_steps, (batch_size, seq_len))


class DiffusionForcingTrainer(nn.Module):
    """
    Diffusion Forcing训练器

    核心思想:
    1. 训练时，序列中每个帧被赋予独立的随机噪声水平
    2. 历史帧可以是干净的(噪声=0)，未来帧是带噪的
    3. 模型学习从任意噪声水平的混合序列中预测干净帧
    4. 推理时，可以自回归地生成任意长度的序列
    """

    def __init__(
        self,
        backbone: nn.Module,  # Transformer backbone
        noise_scheduler: NoiseScheduler,
        latent_dim: int = 16,
        num_frames: int = 16,
        action_dim: int = 7,
        history_len: int = 4,
        prediction_stride: int = 4,  # 稀疏预测步长 (参考GigaWorld)
    ):
        super().__init__()
        self.backbone = backbone
        self.noise_scheduler = noise_scheduler
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.history_len = history_len
        self.prediction_stride = prediction_stride

        # 时间步嵌入 - 将噪声水平编码为条件信号
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        # 动作嵌入 - 将动作向量编码为条件
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        # 预测头 - 从Transformer输出预测去噪后的Latent
        self.pred_head = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        latent_seq: torch.Tensor,  # [B, T, H, W, C] Latent序列
        actions: Optional[torch.Tensor] = None,  # [B, T, action_dim]
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            latent_seq: [B, T, H, W, C] Latent帧序列
            actions: [B, T, action_dim] 动作序列 (微调阶段使用)
            is_training: 是否训练模式

        Returns:
            pred_noise: [B, T_pred, H, W, C] 预测的噪声
            target_noise: [B, T_pred, H, W, C] 目标噪声
        """
        B, T, H, W, C = latent_seq.shape
        device = latent_seq.device

        # 分离历史帧和未来帧
        history_frames = latent_seq[:, :self.history_len]  # [B, H, H, W, C]

        # 稀疏采样未来帧 (参考GigaWorld的稀疏预测)
        future_indices = list(range(self.history_len, T, self.prediction_stride))
        future_frames = latent_seq[:, future_indices]  # [B, T_pred, H, W, C]
        T_pred = len(future_indices)

        # 为每个帧独立采样噪声水平
        if is_training:
            # 历史帧: 小噪声或零噪声 (作为条件)
            history_noise_levels = torch.zeros(B, self.history_len, device=device)
            # 未来帧: 随机噪声水平
            future_noise_levels = self.noise_scheduler.sample_timesteps(B, T_pred).to(device)

            noise_levels = torch.cat([history_noise_levels, future_noise_levels], dim=1)  # [B, H+T_pred]
        else:
            # 推理时: 历史帧干净，未来帧从高噪声开始
            noise_levels = torch.zeros(B, self.history_len + T_pred, device=device)
            noise_levels[:, self.history_len:] = self.noise_scheduler.num_steps - 1

        # 为所有帧添加噪声
        full_seq = torch.cat([history_frames, future_frames], dim=1)  # [B, H+T_pred, H, W, C]
        noise = torch.randn_like(full_seq)
        noisy_seq = self.noise_scheduler.add_noise(full_seq, noise, noise_levels)

        # 将Latent序列转换为Token
        # [B, T, H, W, C] -> [B, T, H*W, C]
        tokens = noisy_seq.view(B, -1, H * W, C)

        # 添加时间步嵌入到每个Token
        time_embeds = self.time_embed(noise_levels.unsqueeze(-1).float() / self.noise_scheduler.num_steps)
        # [B, T, C] -> [B, T, 1, C]
        time_embeds = time_embeds.unsqueeze(2)
        tokens = tokens + time_embeds

        # 如果有动作，添加动作条件
        if actions is not None:
            # 为每个预测帧获取对应的动作
            future_actions = actions[:, future_indices]  # [B, T_pred, action_dim]
            # 历史帧的动作用零填充
            history_actions = torch.zeros(B, self.history_len, self.action_dim, device=device)
            full_actions = torch.cat([history_actions, future_actions], dim=1)
            action_embeds = self.action_embed(full_actions)  # [B, T, C]
            action_embeds = action_embeds.unsqueeze(2)  # [B, T, 1, C]
            tokens = tokens + action_embeds

        # 通过Transformer backbone
        # tokens: [B, T, seq_len, C] -> 需要reshape为 [B, T*seq_len, C]
        B, T_total, seq_len, C = tokens.shape
        flat_tokens = tokens.view(B, T_total * seq_len, C)

        # 创建因果注意力掩码 (时间维度因果)
        causal_mask = self._create_temporal_causal_mask(T_total, seq_len, device)

        output = self.backbone(flat_tokens, attn_mask=causal_mask)

        # 重塑回帧格式
        output = output.view(B, T_total, seq_len, C)

        # 只预测未来帧的噪声
        future_output = output[:, self.history_len:]  # [B, T_pred, seq_len, C]

        # 预测头
        pred_noise = self.pred_head(future_output)  # [B, T_pred, seq_len, C]

        # 重塑回Latent格式
        pred_noise = pred_noise.view(B, T_pred, H, W, C)

        # 目标噪声
        target_noise = noise[:, self.history_len:]

        return pred_noise, target_noise

    def _create_temporal_causal_mask(
        self,
        T_total: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        创建时间因果掩码

        确保帧t只能看到帧 <= t的信息
        """
        # 创建帧级别的因果掩码
        frame_mask = torch.triu(
            torch.ones(T_total, T_total, device=device),
            diagonal=1,
        ).bool()

        # 扩展到Token级别
        # [T_total, T_total] -> [T_total*seq_len, T_total*seq_len]
        mask = frame_mask.repeat_interleave(seq_len, dim=0).repeat_interleave(seq_len, dim=1)

        return mask

    @torch.no_grad()
    def generate(
        self,
        history_latents: torch.Tensor,  # [B, H, H, W, C]
        actions: Optional[torch.Tensor] = None,
        num_future_frames: int = 8,
        num_denoising_steps: int = 10,
    ) -> torch.Tensor:
        """
        自回归生成未来帧

        Args:
            history_latents: 历史Latent帧
            actions: 未来动作序列
            num_future_frames: 要生成的未来帧数量
            num_denoising_steps: 每帧的去噪步数

        Returns:
            generated: [B, num_future_frames, H, W, C] 生成的未来帧
        """
        B, _, H, W, C = history_latents.shape
        device = history_latents.device

        generated = []
        current_history = history_latents.clone()

        for i in range(num_future_frames):
            # 初始化未来帧为纯噪声
            future_frame = torch.randn(B, 1, H, W, C, device=device)

            # 逐步去噪
            for step in range(num_denoising_steps - 1, -1, -1):
                t = torch.full((B, 1), step, device=device)

                # 拼接历史和未来
                full_seq = torch.cat([current_history, future_frame], dim=1)

                # 预测噪声
                pred_noise, _ = self.forward(full_seq, actions, is_training=False)

                # 去噪一步
                alpha_t = self.noise_scheduler.alphas_cumprod[step]
                alpha_prev = self.noise_scheduler.alphas_cumprod[max(step - 1, 0)]

                future_frame = (future_frame - (1 - alpha_t).sqrt() * pred_noise[:, -1:]) / alpha_t.sqrt()
                future_frame = alpha_prev.sqrt() * future_frame + (1 - alpha_prev).sqrt() * torch.randn_like(future_frame)

            generated.append(future_frame[:, 0])

            # 更新历史 (滑动窗口)
            current_history = torch.cat([current_history[:, 1:], future_frame[:, -1:]], dim=1)

        return torch.stack(generated, dim=1)
