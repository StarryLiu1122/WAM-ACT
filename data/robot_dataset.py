"""
Robot Dataset
机器人操作数据集加载器

支持格式:
- RLDS (TensorFlow Datasets)
- HDF5 (自定义格式)
- Zarr (大规模数据集)

"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import pickle


class RobotDataset(Dataset):
    """
    机器人操作数据集

    数据格式:
    - 图像序列: [T, 3, H, W]
    - 动作序列: [T, action_dim]
    - 指令: 文本字符串
    - 状态: [T, state_dim]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        history_len: int = 4,
        chunk_size: int = 16,
        image_size: int = 256,
        action_dim: int = 7,
        num_views: int = 1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.history_len = history_len
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.num_views = num_views

        # 加载数据索引
        self.load_index()

        # 缓存
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_index(self):
        """加载数据索引"""
        index_file = self.data_dir / f'{self.split}_index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.index = json.load(f)
        else:
            # 自动生成索引
            self.index = self._build_index()

        self.episodes = list(self.index.keys())

    def _build_index(self) -> Dict:
        """构建数据索引"""
        index = {}
        data_files = sorted(self.data_dir.glob('*.pkl'))

        for i, file in enumerate(data_files):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            episode_len = len(data['actions'])
            index[f'episode_{i}'] = {
                'file': str(file),
                'length': episode_len,
                'valid_starts': list(range(0, episode_len - self.chunk_size - self.history_len)),
            }

        # 保存索引
        with open(self.data_dir / f'{self.split}_index.json', 'w') as f:
            json.dump(index, f)

        return index

    def __len__(self) -> int:
        return sum(len(ep['valid_starts']) for ep in self.index.values())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本

        Returns:
            sample: 包含以下键的字典
                - 'current_image': [3, H, W]
                - 'history_images': [history_len, 3, H, W]
                - 'next_image': [3, H, W] (预训练用)
                - 'future_images': [chunk_size, 3, H, W] (微调用)
                - 'actions': [chunk_size, action_dim]
                - 'instruction': str
                - 'state': [state_dim]
        """
        # 找到对应的episode和起始位置
        episode_idx, start_idx = self._locate_sample(idx)
        episode_info = self.index[self.episodes[episode_idx]]

        # 加载数据
        data = self._load_episode(episode_info['file'])

        # 提取样本
        sample = self._extract_sample(data, start_idx)

        return sample

    def _locate_sample(self, idx: int) -> Tuple[int, int]:
        """定位样本所在的episode和起始位置"""
        cumulative = 0
        for i, episode in enumerate(self.episodes):
            episode_len = len(self.index[episode]['valid_starts'])
            if idx < cumulative + episode_len:
                return i, self.index[episode]['valid_starts'][idx - cumulative]
            cumulative += episode_len
        raise IndexError(f"Index {idx} out of range")

    def _load_episode(self, file_path: str) -> Dict:
        """加载一个episode的数据"""
        # 检查缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{Path(file_path).stem}.pt"
            if cache_file.exists():
                return torch.load(cache_file)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 转换为Tensor
        data['images'] = torch.from_numpy(data['images']).float() / 255.0  # [T, H, W, 3]
        data['images'] = data['images'].permute(0, 3, 1, 2)  # [T, 3, H, W]
        data['actions'] = torch.from_numpy(data['actions']).float()
        data['states'] = torch.from_numpy(data['states']).float()

        # 缓存
        if self.cache_dir:
            torch.save(data, cache_file)

        return data

    def _extract_sample(self, data: Dict, start_idx: int) -> Dict[str, torch.Tensor]:
        """从episode中提取一个训练样本"""
        # 历史帧
        history_start = max(0, start_idx - self.history_len)
        history_images = data['images'][history_start:start_idx]  # [H, 3, H, W]

        # 如果历史帧不足，用第一帧填充
        if len(history_images) < self.history_len:
            padding = [history_images[0:1]] * (self.history_len - len(history_images))
            history_images = torch.cat(padding + [history_images], dim=0)

        # 当前帧
        current_image = data['images'][start_idx]  # [3, H, W]

        # 下一帧 (预训练目标)
        next_image = data['images'][start_idx + 1]  # [3, H, W]

        # 未来帧 (微调目标)
        future_end = min(start_idx + 1 + self.chunk_size, len(data['images']))
        future_images = data['images'][start_idx + 1:future_end]  # [<=chunk_size, 3, H, W]

        # 如果未来帧不足，用最后一帧填充
        if len(future_images) < self.chunk_size:
            padding = [future_images[-1:]] * (self.chunk_size - len(future_images))
            future_images = torch.cat([future_images] + padding, dim=0)

        # 动作块
        action_end = min(start_idx + self.chunk_size, len(data['actions']))
        actions = data['actions'][start_idx:action_end]  # [<=chunk_size, action_dim]

        if len(actions) < self.chunk_size:
            padding = [actions[-1:]] * (self.chunk_size - len(actions))
            actions = torch.cat([actions] + padding, dim=0)

        # 状态
        state = data['states'][start_idx]  # [state_dim]

        # 指令 (如果有)
        instruction = data.get('instruction', '')

        return {
            'current_image': current_image,
            'history_images': history_images,
            'next_image': next_image,
            'future_images': future_images,
            'actions': actions,
            'state': state,
            'instruction': instruction,
        }


class RobotDataLoader:
    """
    机器人数据加载器工厂
    """

    @staticmethod
    def create_dataloader(
        data_dir: str,
        split: str = 'train',
        batch_size: int = 32,
        num_workers: int = 4,
        **dataset_kwargs,
    ) -> DataLoader:
        """创建DataLoader"""
        dataset = RobotDataset(data_dir, split, **dataset_kwargs)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
