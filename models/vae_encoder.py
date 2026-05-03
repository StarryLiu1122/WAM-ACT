"""
VAE Encoder/Decoder
视觉Latent空间编码器

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class VAEEncoder(nn.Module):
    """
    VAE编码器: 将RGB图像编码到Latent空间

    支持多视角输入拼接 
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,  # Latent通道数 (SD VAE使用16)
        hidden_dims: List[int] = [128, 256, 512, 1024],
        image_size: int = 256,
        num_views: int = 1,  # 多视角数量
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_views = num_views
        self.image_size = image_size

        # 多视角拼接: 将多个视角的图像在通道维度拼接
        # 参考GigaWorld-Policy: Compose(o_left, o_front, o_right)
        total_in_channels = in_channels * num_views if num_views > 1 else in_channels

        # 编码器层
        modules = []
        curr_dim = total_in_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(curr_dim, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            curr_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        # 计算下采样后的空间尺寸
        self.downsample_factor = 2 ** len(hidden_dims)
        self.latent_size = image_size // self.downsample_factor

        # VAE的均值和对数方差投影
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像到Latent分布参数

        Args:
            x: [B, C*num_views, H, W] 多视角拼接图像

        Returns:
            mu: [B, latent_dim, H', W'] 均值
            logvar: [B, latent_dim, H', W'] 对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播

        Returns:
            z: [B, latent_dim, H', W'] 采样后的Latent
            mu: 均值
            logvar: 对数方差
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像编码为Token序列 (用于Transformer输入)

        Returns:
            tokens: [B, seq_len, dim] 其中seq_len = H'*W', dim = latent_dim
        """
        z, _, _ = self.forward(x)
        B, C, H, W = z.shape
        # 展平为Token序列
        tokens = z.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        return tokens


class VAEDecoder(nn.Module):
    """
    VAE解码器: 将Latent解码回RGB图像
    """

    def __init__(
        self,
        latent_dim: int = 16,
        out_channels: int = 3,
        hidden_dims: List[int] = [1024, 512, 256, 128],
        image_size: int = 256,
        num_views: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_views = num_views

        total_out_channels = out_channels * num_views if num_views > 1 else out_channels

        # 计算上采样后的空间尺寸
        self.upsample_factor = 2 ** len(hidden_dims)
        self.latent_size = image_size // self.upsample_factor

        # 初始投影
        self.init_conv = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)

        # 解码器层
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], 
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], total_out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # 输出范围[-1, 1]
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码Latent到图像

        Args:
            z: [B, latent_dim, H', W']

        Returns:
            x: [B, C*num_views, H, W]
        """
        h = self.init_conv(z)
        h = self.decoder(h)
        x = self.final_conv(h)
        return x

    def decode_from_tokens(self, tokens: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        """
        从Token序列解码图像

        Args:
            tokens: [B, seq_len, dim]
            latent_h: Latent高度
            latent_w: Latent宽度

        Returns:
            x: [B, C, H, W]
        """
        B, seq_len, dim = tokens.shape
        # 重塑为Latent格式
        z = tokens.transpose(1, 2).view(B, dim, latent_h, latent_w)
        return self.decode(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)


class MultiViewComposer:
    """
    多视角图像拼接器
    参考GigaWorld-Policy: Compose(o_left, o_front, o_right)
    """

    @staticmethod
    def compose_views(views: List[torch.Tensor], layout: str = 'horizontal') -> torch.Tensor:
        """
        将多个视角图像拼接为单个图像

        Args:
            views: List of [B, C, H, W] 多视角图像
            layout: 'horizontal' or 'grid'

        Returns:
            composed: [B, C*num_views, H, W] 拼接后的图像
        """
        if layout == 'horizontal':
            # 在通道维度拼接
            return torch.cat(views, dim=1)
        elif layout == 'grid':
            # 2x2网格拼接 (假设4个视角)
            assert len(views) == 4
            top = torch.cat([views[0], views[1]], dim=3)  # 水平拼接
            bottom = torch.cat([views[2], views[3]], dim=3)
            composed = torch.cat([top, bottom], dim=2)  # 垂直拼接
            return composed
        else:
            raise ValueError(f"Unknown layout: {layout}")

    @staticmethod
    def decompose_views(composed: torch.Tensor, num_views: int) -> List[torch.Tensor]:
        """将拼接图像分解为多个视角"""
        C = composed.shape[1] // num_views
        return [composed[:, i*C:(i+1)*C] for i in range(num_views)]
