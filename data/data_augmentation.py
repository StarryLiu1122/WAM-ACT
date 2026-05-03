"""
Data Augmentation
数据增强模块

参考:
- Diffusion Policy: 图像增强策略
- GigaWorld-Policy: 多视角一致性增强
- MOTUS: 时间维度增强
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from typing import Dict, Tuple, Optional


class DataAugmentor:
    """
    数据增强器

    包含:
    1. 图像增强 (颜色抖动、几何变换)
    2. 动作增强 (添加噪声)
    3. 时间增强 (帧丢弃、速度变化)
    """

    def __init__(
        self,
        image_size: int = 256,
        color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1),
        action_noise_scale: float = 0.01,
        temporal_drop_prob: float = 0.1,
    ):
        self.image_size = image_size
        self.action_noise_scale = action_noise_scale
        self.temporal_drop_prob = temporal_drop_prob

        # 图像增强
        self.color_jitter = T.ColorJitter(
            brightness=color_jitter[0],
            contrast=color_jitter[1],
            saturation=color_jitter[2],
            hue=color_jitter[3],
        )

        self.geometric_aug = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        增强单张图像

        Args:
            image: [3, H, W]

        Returns:
            augmented: [3, H, W]
        """
        # 颜色增强
        image = self.color_jitter(image)

        # 几何增强
        image = self.geometric_aug(image)

        return image

    def augment_sequence(
        self,
        images: torch.Tensor,  # [T, 3, H, W]
        actions: torch.Tensor,  # [T, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强图像序列和动作序列

        保持时间一致性: 对序列中的所有帧应用相同的随机变换
        """
        T_len = images.shape[0]

        # 生成统一的随机参数
        # 颜色增强参数
        brightness_factor = np.random.uniform(1 - 0.4, 1 + 0.4)
        contrast_factor = np.random.uniform(1 - 0.4, 1 + 0.4)
        saturation_factor = np.random.uniform(1 - 0.4, 1 + 0.4)
        hue_factor = np.random.uniform(-0.1, 0.1)

        # 几何增强参数
        i, j, h, w = T.RandomResizedCrop.get_params(
            images[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
        )
        flip = np.random.random() < 0.5

        # 应用增强
        augmented_images = []
        for t in range(T_len):
            img = images[t]

            # 颜色增强 (相同参数)
            img = T.functional.adjust_brightness(img, brightness_factor)
            img = T.functional.adjust_contrast(img, contrast_factor)
            img = T.functional.adjust_saturation(img, saturation_factor)
            img = T.functional.adjust_hue(img, hue_factor)

            # 几何增强 (相同参数)
            img = T.functional.resized_crop(img, i, j, h, w, (self.image_size, self.image_size))
            if flip:
                img = T.functional.hflip(img)

            augmented_images.append(img)

        augmented_images = torch.stack(augmented_images)

        # 动作增强: 添加小噪声
        noise = torch.randn_like(actions) * self.action_noise_scale
        augmented_actions = actions + noise

        # 时间增强: 随机丢弃帧
        if np.random.random() < self.temporal_drop_prob and T_len > 2:
            drop_idx = np.random.randint(1, T_len - 1)
            mask = torch.ones(T_len, dtype=torch.bool)
            mask[drop_idx] = False
            augmented_images = augmented_images[mask]
            augmented_actions = augmented_actions[mask]

        return augmented_images, augmented_actions

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        增强整个batch

        Args:
            batch: 包含以下键的字典
                - 'current_image': [B, 3, H, W]
                - 'history_images': [B, H, 3, H, W]
                - 'future_images': [B, K, 3, H, W]
                - 'actions': [B, K, action_dim]

        Returns:
            augmented_batch: 增强后的batch
        """
        B = batch['current_image'].shape[0]

        # 增强当前图像
        batch['current_image'] = torch.stack([
            self.augment_image(batch['current_image'][i]) 
            for i in range(B)
        ])

        # 增强历史图像序列
        for i in range(B):
            hist = batch['history_images'][i]  # [H, 3, H, W]
            # 对历史帧应用统一增强
            hist_aug = self.augment_sequence(hist, torch.zeros(hist.shape[0], 1))[0]
            batch['history_images'][i] = hist_aug

        # 增强未来图像和动作
        for i in range(B):
            future = batch['future_images'][i]  # [K, 3, H, W]
            actions = batch['actions'][i]  # [K, action_dim]
            future_aug, actions_aug = self.augment_sequence(future, actions)
            batch['future_images'][i] = future_aug
            batch['actions'][i] = actions_aug

        return batch


class MultiViewAugmentor:
    """
    多视角数据增强

    参考GigaWorld-Policy:
    - 保持多视角几何一致性
    - 视角间颜色匹配
    """

    def __init__(self, num_views: int = 3):
        self.num_views = num_views

    def augment_multiview(
        self,
        views: torch.Tensor,  # [num_views, 3, H, W]
    ) -> torch.Tensor:
        """
        增强多视角图像

        保持:
        1. 相同的几何变换 (确保视角对齐)
        2. 独立的颜色变换 (模拟光照差异)
        """
        # 生成统一的几何参数
        i, j, h, w = T.RandomResizedCrop.get_params(
            views[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
        )
        flip = np.random.random() < 0.5

        augmented = []
        for v in range(self.num_views):
            img = views[v]

            # 独立的颜色增强
            img = T.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)

            # 统一的几何增强
            img = T.functional.resized_crop(img, i, j, h, w, (img.shape[-2], img.shape[-1]))
            if flip:
                img = T.functional.hflip(img)

            augmented.append(img)

        return torch.stack(augmented)
