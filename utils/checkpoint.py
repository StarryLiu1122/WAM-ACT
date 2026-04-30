"""
Checkpoint Manager
检查点管理

功能:
- 保存/加载检查点
- 自动清理旧检查点
- 最佳模型跟踪
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import shutil


class CheckpointManager:
    """
    检查点管理器

    自动管理检查点文件，保留最新的N个和最佳的1个
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        keep_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best

        self.checkpoints = []  # 保存的检查点列表
        self.best_checkpoint = None

    def save(self, checkpoint: Dict, name: str):
        """
        保存检查点

        Args:
            checkpoint: 检查点字典
            name: 检查点名称
        """
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)

        self.checkpoints.append(path)

        # 清理旧检查点
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint != self.best_checkpoint:
                old_checkpoint.unlink(missing_ok=True)

        # 如果是最佳模型，单独保存
        if name == 'best' and self.keep_best:
            best_path = self.checkpoint_dir / 'best.pt'
            shutil.copy(path, best_path)
            self.best_checkpoint = best_path

    def load(self, name: str) -> Dict:
        """加载检查点"""
        path = self.checkpoint_dir / f"{name}.pt"
        return torch.load(path)

    def get_latest(self) -> Optional[Path]:
        """获取最新的检查点"""
        if self.checkpoints:
            return self.checkpoints[-1]
        return None

    def get_best(self) -> Optional[Path]:
        """获取最佳检查点"""
        best_path = self.checkpoint_dir / 'best.pt'
        if best_path.exists():
            return best_path
        return None

    def list_checkpoints(self) -> List[Path]:
        """列出所有检查点"""
        return sorted(self.checkpoint_dir.glob('*.pt'))
