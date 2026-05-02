"""
Finetune Trainer
微调阶段: 同步预测下一帧图像和Action Chunk

训练目标:
- 学习动作-视觉耦合表示
- 预测Action Chunk (未来K步动作)
- 预测稀疏未来帧 (保持世界模型能力)
- 动作-视觉一致性约束

参考:
- MOTUS (Bi et al., 2025): 流匹配VLA微调
- CogACT (Zhao et al., 2025): DiT动作微调
- GigaWorld-Policy (2026): 稀疏帧监督
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from collections import defaultdict

from .trainer import BaseTrainer
from ..models.wam_act import WAMACT


class FinetuneTrainer(BaseTrainer):
    """
    微调训练器

    训练流程:
    1. 加载预训练权重 (冻结部分层)
    2. 添加动作预测头
    3. 联合训练动作预测和未来帧预测
    4. 使用复合损失优化
    """

    def __init__(
        self,
        model: WAMACT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
        action_loss_weight: float = 1.0,
        future_loss_weight: float = 0.5,
        consistency_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        self.stage = 'finetune'
        self.action_loss_weight = action_loss_weight
        self.future_loss_weight = future_loss_weight
        self.consistency_loss_weight = consistency_loss_weight

        # 加载预训练权重
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        # 冻结策略
        if freeze_encoder:
            self._freeze_encoder()

        # 只优化动作相关参数
        self._setup_optimizer()

    def load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location=self.device)

        # 过滤掉动作头的权重 (微调时重新训练)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.model.state_dict()

        # 只加载匹配的权重
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)

        self.logger.info(f"Loaded pretrained weights from {path}")
        self.logger.info(f"Loaded {len(filtered_dict)}/{len(pretrained_dict)} layers")

    def _freeze_encoder(self):
        """冻结编码器和Transformer部分层"""
        # 冻结VAE编码器
        for param in self.model.vae_encoder.parameters():
            param.requires_grad = False

        # 冻结文本编码器
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        # 冻结状态编码器
        for param in self.model.state_encoder.parameters():
            param.requires_grad = False

        # 冻结Transformer的前半部分层
        freeze_layers = len(self.model.transformer.blocks) // 2
        for i, block in enumerate(self.model.transformer.blocks):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

        self.logger.info(f"Frozen encoder and first {freeze_layers} transformer blocks")

    def _setup_optimizer(self):
        """设置优化器 - 只优化可训练参数"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)

        self.logger.info(f"Optimizer set up with {len(trainable_params)} parameter groups")

    def train_epoch(self) -> Dict[str, float]:
        """微调一个epoch"""
        self.model.train()

        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据转移
            current_image = batch['current_image'].to(self.device)
            target_actions = batch['actions'].to(self.device)
            future_images = batch.get('future_images', None)
            if future_images is not None:
                future_images = future_images.to(self.device)

            history_images = batch.get('history_images', None)
            if history_images is not None:
                history_images = history_images.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model.forward_finetune(
                    current_image=current_image,
                    target_actions=target_actions,
                    target_future_images=future_images,
                    state=state,
                    history_images=history_images,
                )

                loss = outputs['total_loss']

            # 反向传播
            self._backward_step(loss)

            # 记录指标
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['action_loss'] += outputs['losses']['action'].item()

            if future_images is not None:
                epoch_metrics['future_loss'] += outputs['losses']['future'].item()
                epoch_metrics['consistency_loss'] += outputs['losses']['consistency'].item()

            num_batches += 1

            # 日志
            if batch_idx % self.log_every == 0:
                self.logger.info(
                    f"[Finetune] Epoch {self.current_epoch+1} | "
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Action: {outputs['losses']['action'].item():.6f}"
                )

        # 平均指标
        metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """微调验证"""
        self.model.eval()

        val_metrics = defaultdict(float)
        num_batches = 0

        for batch in self.val_loader:
            current_image = batch['current_image'].to(self.device)
            target_actions = batch['actions'].to(self.device)
            future_images = batch.get('future_images', None)
            if future_images is not None:
                future_images = future_images.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            # 前向传播
            outputs = self.model.forward_finetune(
                current_image=current_image,
                target_actions=target_actions,
                target_future_images=future_images,
                state=state,
            )

            val_metrics['total_loss'] += outputs['total_loss'].item()
            val_metrics['action_loss'] += outputs['losses']['action'].item()

            if future_images is not None:
                val_metrics['future_loss'] += outputs['losses']['future'].item()
                val_metrics['consistency_loss'] += outputs['losses']['consistency'].item()

            # 计算动作MSE
            pred_actions = outputs['pred_actions']
            action_mse = F.mse_loss(pred_actions, target_actions)
            val_metrics['action_mse'] += action_mse.item()

            num_batches += 1

        metrics = {k: v / num_batches for k, v in val_metrics.items()}

        return metrics

    @torch.no_grad()
    def evaluate_action_accuracy(self, num_batches: int = 10):
        """
        评估动作预测精度

        计算:
        - 动作MSE
        - 动作MAE
        - 逐维度误差
        """
        self.model.eval()

        all_pred_actions = []
        all_target_actions = []

        for i, batch in enumerate(self.val_loader):
            if i >= num_batches:
                break

            current_image = batch['current_image'].to(self.device)
            target_actions = batch['actions'].to(self.device)

            # 预测
            outputs = self.model.forward_finetune(
                current_image=current_image,
                target_actions=target_actions,
            )

            all_pred_actions.append(outputs['pred_actions'].cpu())
            all_target_actions.append(target_actions.cpu())

        pred = torch.cat(all_pred_actions, dim=0)
        target = torch.cat(all_target_actions, dim=0)

        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()

        # 逐维度误差
        dim_mse = F.mse_loss(pred, target, reduction='none').mean(dim=(0, 1))

        return {
            'mse': mse,
            'mae': mae,
            'dim_mse': dim_mse.tolist(),
        }


def run_finetune(
    data_dir: str,
    pretrained_path: str,
    output_dir: str,
    batch_size: int = 16,
    num_epochs: int = 50,
    lr: float = 1e-5,
    freeze_encoder: bool = True,
    **model_kwargs,
):
    """
    运行微调

    使用示例:
    ```python
    run_finetune(
        data_dir='/path/to/data',
        pretrained_path='./outputs/pretrain/checkpoints/best.pt',
        output_dir='./outputs/finetune',
        batch_size=16,
        num_epochs=50,
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
    )

    val_loader = RobotDataLoader.create_dataloader(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
    )

    # 创建模型
    model = WAMACT(**model_kwargs)

    # 创建训练器
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
        output_dir=output_dir,
        max_epochs=num_epochs,
    )

    # 训练
    trainer.train()

    return trainer
