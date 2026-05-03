"""
Pretrain Trainer
预训练阶段: 基于当前图像预测下一帧图像

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from collections import defaultdict

from .trainer import BaseTrainer
from ..models.wam_act import WAMACT
from ..data.robot_dataset import RobotDataset


class PretrainTrainer(BaseTrainer):
    """
    预训练器

    训练流程:
    1. 加载batch数据
    2. 编码当前帧和历史帧
    3. 对下一帧添加噪声
    4. Transformer预测去噪后的下一帧
    5. 计算MSE重建损失
    6. 反向传播更新
    """

    def __init__(
        self,
        model: WAMACT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        # 预训练特定配置
        self.stage = 'pretrain'

    def train_epoch(self) -> Dict[str, float]:
        """预训练一个epoch"""
        self.model.train()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据转移到设备
            current_image = batch['current_image'].to(self.device)
            next_image = batch['next_image'].to(self.device)
            history_images = batch.get('history_images', None)
            if history_images is not None:
                history_images = history_images.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            instruction = batch.get('instruction', None)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_latent, target_latent = self.model.forward_pretrain(
                    current_image=current_image,
                    next_image=next_image,
                    instruction_tokens=instruction,
                    state=state,
                    history_images=history_images,
                )

                # 计算损失
                loss = self.model.get_pretrain_loss(pred_latent, target_latent)

            # 反向传播
            self._backward_step(loss)

            # 记录指标
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['mse'] += F.mse_loss(pred_latent, target_latent).item()

            num_batches += 1

            # 日志
            if batch_idx % self.log_every == 0:
                self.logger.info(
                    f"[Pretrain] Epoch {self.current_epoch+1} | "
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        # 平均指标
        metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """预训练验证"""
        self.model.eval()

        val_metrics = defaultdict(float)
        num_batches = 0

        for batch in self.val_loader:
            current_image = batch['current_image'].to(self.device)
            next_image = batch['next_image'].to(self.device)
            history_images = batch.get('history_images', None)
            if history_images is not None:
                history_images = history_images.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            # 前向传播
            pred_latent, target_latent = self.model.forward_pretrain(
                current_image=current_image,
                next_image=next_image,
                state=state,
                history_images=history_images,
            )

            loss = self.model.get_pretrain_loss(pred_latent, target_latent)

            val_metrics['loss'] += loss.item()
            val_metrics['mse'] += F.mse_loss(pred_latent, target_latent).item()

            num_batches += 1

        metrics = {k: v / num_batches for k, v in val_metrics.items()}

        return metrics

    @torch.no_grad()
    def visualize_prediction(self, num_samples: int = 4):
        """
        可视化预测结果

        生成当前帧 -> 预测下一帧的对比图
        """
        import torchvision
        from pathlib import Path

        self.model.eval()

        # 获取验证batch
        batch = next(iter(self.val_loader))
        current_image = batch['current_image'][:num_samples].to(self.device)
        next_image = batch['next_image'][:num_samples].to(self.device)

        # 预测
        pred_latent, _ = self.model.forward_pretrain(current_image, next_image)

        # 解码预测帧
        pred_image = self.model.vae_decoder.decode(pred_latent)

        # 保存对比图
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        for i in range(num_samples):
            comparison = torch.cat([current_image[i], next_image[i], pred_image[i]], dim=2)
            torchvision.utils.save_image(
                comparison,
                viz_dir / f'pretrain_epoch{self.current_epoch+1}_sample{i}.png',
                normalize=True,
                value_range=(-1, 1),
            )


def run_pretrain(
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    image_size: int = 256,
    **model_kwargs,
):
    """
    运行预训练

    使用示例:
    ```python
    run_pretrain(
        data_dir='/path/to/data',
        output_dir='./outputs/pretrain',
        batch_size=32,
        num_epochs=100,
    )
    ```
    """
    from ..data.robot_dataset import RobotDataLoader
    from .lr_scheduler import get_cosine_schedule_with_warmup

    # 创建数据加载器
    train_loader = RobotDataLoader.create_dataloader(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        image_size=image_size,
    )

    val_loader = RobotDataLoader.create_dataloader(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
        image_size=image_size,
    )

    # 创建模型
    model = WAMACT(image_size=image_size, **model_kwargs)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度
    total_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 创建训练器
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_epochs=num_epochs,
    )

    # 训练
    trainer.train()

    return trainer
