"""
Base Trainer
通用训练器基类

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Any
import time
import json
from collections import defaultdict

from ..utils.logging_utils import get_logger
from ..utils.checkpoint import CheckpointManager


class BaseTrainer:
    """
    训练器基类

    功能:
    1. 训练循环管理
    2. 混合精度训练 (AMP)
    3. 梯度裁剪
    4. 日志记录
    5. 检查点保存
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        output_dir: str = './outputs',
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        save_every: int = 1,
        log_every: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.save_every = save_every
        self.log_every = log_every

        # 优化器
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # 混合精度
        self.scaler = GradScaler() if use_amp else None

        # 日志
        self.logger = get_logger('trainer')

        # 检查点管理
        self.checkpoint_manager = CheckpointManager(self.output_dir / 'checkpoints')

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 指标记录
        self.metrics_history = defaultdict(list)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证
        """
        raise NotImplementedError

    def train(self):
        """主训练循环"""
        self.logger.info(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()
            self._log_metrics('train', train_metrics)

            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self._log_metrics('val', val_metrics)

                # 保存最佳模型
                val_loss = val_metrics.get('loss', val_metrics.get('total_loss', float('inf')))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')

            # 定期保存
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_metrics.get('loss', 0)
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        self.logger.info("Training completed!")
        self.save_checkpoint('final')

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]):
        """记录指标"""
        log_str = f"[{prefix}] Epoch {self.current_epoch+1}/{self.max_epochs}"
        for key, value in metrics.items():
            log_str += f" | {key}: {value:.6f}"
            self.metrics_history[f"{prefix}/{key}"].append(value)

        if self.global_step % self.log_every == 0:
            self.logger.info(log_str)

    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': dict(self.metrics_history),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        self.checkpoint_manager.save(checkpoint, name)
        self.logger.info(f"Checkpoint saved: {name}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Checkpoint loaded: {path}")

    def _backward_step(self, loss: torch.Tensor):
        """反向传播一步"""
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        self.global_step += 1


class DistributedTrainer(BaseTrainer):
    """
    分布式训练器

    支持多GPU训练 (DDP)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

    def train_epoch(self) -> Dict[str, float]:
        """
        分布式训练epoch
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            with autocast(enabled=self.use_amp):
                loss = self.compute_loss(batch)

            self._backward_step(loss)

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.log_every == 0:
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        return {'loss': total_loss / num_batches}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算损失
        """
        raise NotImplementedError
