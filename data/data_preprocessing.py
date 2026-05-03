"""
Data Preprocessing
数据预处理模块

包含:
- 图像预处理 (归一化、resize、裁剪)
- 动作归一化 (参考Diffusion Policy)
- 指令Token化
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torchvision import transforms


class ImagePreprocessor:
    """
    图像预处理器

    参考:
    - CLIP预处理: 归一化到[-1, 1]
    - Stable Diffusion VAE: 归一化到[-1, 1]
    """

    def __init__(
        self,
        image_size: int = 256,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        augment: bool = False,
    ):
        self.image_size = image_size

        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 数据增强 (训练时使用)
        if augment:
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.augment = self.base_transform

    def __call__(self, image: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: [3, H, W] 或 [H, W, 3]
            is_training: 是否训练模式

        Returns:
            processed: [3, image_size, image_size]
        """
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        if is_training:
            return self.augment(image)
        else:
            return self.base_transform(image)


class ActionNormalizer:
    """
    动作归一化器

    参考Diffusion Policy:
    - 使用训练集的统计量进行归一化
    - 支持Min-Max和Z-Score两种方法
    """

    def __init__(
        self,
        method: str = 'minmax',
        action_dim: int = 7,
    ):
        self.method = method
        self.action_dim = action_dim

        # 统计量
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit(self, actions: np.ndarray):
        """
        从数据计算归一化参数

        Args:
            actions: [N, action_dim] 动作数据
        """
        if self.method == 'minmax':
            self.min = np.min(actions, axis=0)
            self.max = np.max(actions, axis=0)
            # 避免除零
            self.max = np.where(self.max == self.min, self.min + 1, self.max)
        elif self.method == 'zscore':
            self.mean = np.mean(actions, axis=0)
            self.std = np.std(actions, axis=0)
            self.std = np.where(self.std < 1e-6, 1.0, self.std)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """归一化动作"""
        if self.method == 'minmax':
            device = actions.device
            min_val = torch.from_numpy(self.min).to(device)
            max_val = torch.from_numpy(self.max).to(device)
            return 2.0 * (actions - min_val) / (max_val - min_val) - 1.0
        elif self.method == 'zscore':
            device = actions.device
            mean = torch.from_numpy(self.mean).to(device)
            std = torch.from_numpy(self.std).to(device)
            return (actions - mean) / std

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """反归一化动作"""
        if self.method == 'minmax':
            device = actions.device
            min_val = torch.from_numpy(self.min).to(device)
            max_val = torch.from_numpy(self.max).to(device)
            return (actions + 1.0) / 2.0 * (max_val - min_val) + min_val
        elif self.method == 'zscore':
            device = actions.device
            mean = torch.from_numpy(self.mean).to(device)
            std = torch.from_numpy(self.std).to(device)
            return actions * std + mean

    def save(self, path: str):
        """保存归一化参数"""
        stats = {
            'method': self.method,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'std': self.std,
        }
        np.savez(path, **stats)

    def load(self, path: str):
        """加载归一化参数"""
        stats = np.load(path)
        self.method = str(stats['method'])
        self.min = stats.get('min', None)
        self.max = stats.get('max', None)
        self.mean = stats.get('mean', None)
        self.std = stats.get('std', None)


class InstructionTokenizer:
    """
    指令Token化器

    简单的WordPiece/BPE分词器
    实际项目中可用HuggingFace Tokenizer
    """

    def __init__(self, vocab_size: int = 50000, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # 简化的词汇表 (实际应使用预训练tokenizer)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

    def encode(self, text: str) -> torch.Tensor:
        """
        将文本编码为Token ID

        Args:
            text: 输入文本

        Returns:
            tokens: [max_length] Token ID序列
        """
        words = text.lower().split()
        tokens = [self.word_to_idx.get('<BOS>', 2)]

        for word in words:
            if word not in self.word_to_idx:
                # 分配新ID
                new_id = len(self.word_to_idx)
                if new_id < self.vocab_size:
                    self.word_to_idx[word] = new_id
                    self.idx_to_word[new_id] = word
            tokens.append(self.word_to_idx.get(word, self.word_to_idx.get('<UNK>', 1)))

        tokens.append(self.word_to_idx.get('<EOS>', 3))

        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.word_to_idx.get('<PAD>', 0)] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """将Token ID解码为文本"""
        words = []
        for idx in tokens.tolist():
            if idx in [0, 2, 3]:  # PAD, BOS, EOS
                continue
            words.append(self.idx_to_word.get(idx, '<UNK>'))
        return ' '.join(words)

    def batch_encode(self, texts: list) -> torch.Tensor:
        """批量编码"""
        tokens = [self.encode(text) for text in texts]
        return torch.stack(tokens)
