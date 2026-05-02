"""
Training Module Tests
训练模块单元测试
"""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from wam_act.models import WAMACT
from wam_act.training import BaseTrainer, PretrainTrainer, FinetuneTrainer
from wam_act.training.lr_scheduler import get_cosine_schedule_with_warmup


class TestLRScheduler:
    """测试学习率调度器"""

    def test_cosine_schedule(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )

        # 检查warmup阶段
        for _ in range(10):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] >= 1e-4

        # 检查衰减阶段
        for _ in range(90):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] < 1e-4


class TestBaseTrainer:
    """测试基础训练器"""

    def test_trainer_init(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)

        # 创建虚拟数据
        images = torch.randn(10, 3, 64, 64)
        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=2)

        trainer = BaseTrainer(
            model=model,
            train_loader=loader,
            max_epochs=1,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None


class TestPretrainTrainer:
    """测试预训练器"""

    def test_pretrain_loss(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)

        current = torch.randn(2, 3, 64, 64)
        next_img = torch.randn(2, 3, 64, 64)

        pred_latent, target_latent = model.forward_pretrain(current, next_img)
        loss = model.get_pretrain_loss(pred_latent, target_latent)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestFinetuneTrainer:
    """测试微调训练器"""

    def test_finetune_loss(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)

        current = torch.randn(2, 3, 64, 64)
        actions = torch.randn(2, 16, 7)
        future = torch.randn(2, 16, 3, 64, 64)

        outputs = model.forward_finetune(current, actions, future)

        assert outputs['total_loss'].item() >= 0
        assert not torch.isnan(outputs['total_loss'])
        assert 'losses' in outputs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
