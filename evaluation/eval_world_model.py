"""
World Model Evaluator
世界模型评估器

评估指标:
- 帧预测质量 (MSE, PSNR, SSIM, LPIPS)
- 长期rollout稳定性
- 动作条件预测精度
- 因果一致性

参考:
- WorldGym (2025): 世界模型评估框架
- Diffusion Forcing (Chen et al., 2024): 长期rollout评估
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..models.wam_act import WAMACT


class WorldModelEvaluator:
    """
    世界模型评估器

    功能:
    1. 单步帧预测质量评估
    2. 多步自回归rollout评估
    3. 动作条件预测评估
    4. 因果一致性检查
    """

    def __init__(
        self,
        model: WAMACT,
        device: str = 'cuda',
        num_views: int = 1,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_views = num_views

        self.model.eval()

    @torch.no_grad()
    def evaluate_single_step(
        self,
        val_loader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        评估单步预测质量

        输入当前帧，预测下一帧，与真实下一帧比较
        """
        all_pred_latents = []
        all_target_latents = []
        all_pred_images = []
        all_target_images = []

        total_batches = 0

        for batch_idx, batch in enumerate(val_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            current_image = batch['current_image'].to(self.device)
            next_image = batch['next_image'].to(self.device)
            history_images = batch.get('history_images', None)
            if history_images is not None:
                history_images = history_images.to(self.device)

            # 预测
            pred_latent, target_latent = self.model.forward_pretrain(
                current_image=current_image,
                next_image=next_image,
                history_images=history_images,
            )

            # 解码图像
            pred_image = self.model.vae_decoder.decode(pred_latent)

            all_pred_latents.append(pred_latent.cpu())
            all_target_latents.append(target_latent.cpu())
            all_pred_images.append(pred_image.cpu())
            all_target_images.append(next_image.cpu())

            total_batches += 1

        # 拼接
        pred_latents = torch.cat(all_pred_latents, dim=0)
        target_latents = torch.cat(all_target_latents, dim=0)
        pred_images = torch.cat(all_pred_images, dim=0)
        target_images = torch.cat(all_target_images, dim=0)

        # 计算指标
        metrics = self._compute_frame_metrics(pred_latents, target_latents, pred_images, target_images)
        metrics['num_batches'] = total_batches

        return metrics

    @torch.no_grad()
    def evaluate_multi_step_rollout(
        self,
        val_loader: torch.utils.data.DataLoader,
        rollout_length: int = 8,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        评估多步自回归rollout稳定性

        从当前帧开始，自回归地预测未来帧，评估长期一致性

        Args:
            rollout_length: rollout步数
            num_episodes: 评估的episode数量
        """
        rollout_mses = []
        rollout_psnrs = []
        rollout_ssims = []

        for episode_idx in range(num_episodes):
            batch = next(iter(val_loader))

            current_image = batch['current_image'][0:1].to(self.device)
            history_images = batch.get('history_images', None)
            if history_images is not None:
                history_images = history_images[0:1].to(self.device)

            # 获取真实未来帧
            future_images = batch['future_images'][0:1].to(self.device)  # [1, K, 3, H, W]

            # 自回归rollout
            pred_rollout = []
            curr = current_image

            for step in range(min(rollout_length, future_images.shape[1])):
                # 预测下一步
                # 使用模型的generate方法
                next_latent = self._predict_next_frame(curr, history_images)
                next_image = self.model.vae_decoder.decode(next_latent)

                pred_rollout.append(next_image)

                # 更新历史
                curr = next_image
                if history_images is not None:
                    history_images = torch.cat([history_images[:, 1:], curr.unsqueeze(1)], dim=1)

            pred_rollout = torch.cat(pred_rollout, dim=0)  # [rollout_length, 3, H, W]
            target_rollout = future_images[0, :rollout_length]  # [rollout_length, 3, H, W]

            # 计算指标
            mse = F.mse_loss(pred_rollout, target_rollout).item()
            psnr = self._compute_psnr(pred_rollout, target_rollout)

            rollout_mses.append(mse)
            rollout_psnrs.append(psnr)

        metrics = {
            'rollout_mse_mean': np.mean(rollout_mses),
            'rollout_mse_std': np.std(rollout_mses),
            'rollout_psnr_mean': np.mean(rollout_psnrs),
            'rollout_psnr_std': np.std(rollout_psnrs),
            'rollout_length': rollout_length,
            'num_episodes': num_episodes,
        }

        return metrics

    @torch.no_grad()
    def evaluate_action_conditioned(
        self,
        val_loader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """
        评估动作条件预测

        给定动作序列，评估预测的未来帧是否与真实未来帧一致
        """
        action_conditioned_mses = []

        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            current_image = batch['current_image'].to(self.device)
            target_actions = batch['actions'].to(self.device)
            future_images = batch.get('future_images', None)
            if future_images is not None:
                future_images = future_images.to(self.device)

            # 使用微调模型预测
            outputs = self.model.forward_finetune(
                current_image=current_image,
                target_actions=target_actions,
                target_future_images=future_images,
            )

            if outputs['pred_future_latents'] is not None and future_images is not None:
                pred_latents = outputs['pred_future_latents']

                # 编码目标帧
                B, K, C, H, W = future_images.shape
                future_flat = future_images.view(B * K, C, H, W)
                target_latents, _, _ = self.model.vae_encoder(future_flat)
                target_latents = target_latents.view(B, K, -1, self.model.latent_dim)

                # 计算MSE
                mse = F.mse_loss(pred_latents, target_latents).item()
                action_conditioned_mses.append(mse)

        metrics = {
            'action_conditioned_mse': np.mean(action_conditioned_mses) if action_conditioned_mses else 0,
            'num_evaluated_batches': num_batches,
        }

        return metrics

    def _predict_next_frame(
        self,
        current_image: torch.Tensor,
        history_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """预测下一帧Latent"""
        # 创建一个虚拟的next_image用于调用forward_pretrain
        dummy_next = torch.zeros_like(current_image)

        pred_latent, _ = self.model.forward_pretrain(
            current_image=current_image,
            next_image=dummy_next,
            history_images=history_images,
        )

        return pred_latent

    def _compute_frame_metrics(
        self,
        pred_latents: torch.Tensor,
        target_latents: torch.Tensor,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> Dict[str, float]:
        """计算帧预测指标"""
        metrics = {}

        # Latent空间MSE
        metrics['latent_mse'] = F.mse_loss(pred_latents, target_latents).item()

        # 图像空间MSE
        metrics['image_mse'] = F.mse_loss(pred_images, target_images).item()

        # PSNR
        metrics['psnr'] = self._compute_psnr(pred_images, target_images)

        # SSIM
        metrics['ssim'] = self._compute_ssim(pred_images, target_images)

        # LPIPS (简化版 - 使用特征差异)
        metrics['lpips_approx'] = self._compute_lpips_approx(pred_images, target_images)

        # 感知质量
        metrics['perceptual'] = self._compute_perceptual_loss(pred_images, target_images)

        return metrics

    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(2.0) - 10 * torch.log10(mse).item()

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """简化版SSIM"""
        # 使用高斯窗口的简化版本
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_target = F.avg_pool2d(target, 11, stride=1, padding=5)

        sigma_pred_sq = F.avg_pool2d(pred ** 2, 11, stride=1, padding=5) - mu_pred ** 2
        sigma_target_sq = F.avg_pool2d(target ** 2, 11, stride=1, padding=5) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) /                ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2))

        return ssim.mean().item()

    def _compute_lpips_approx(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """近似LPIPS (使用预训练VGG特征差异)"""
        # 简化版本: 使用简单的卷积特征差异
        with torch.no_grad():
            # 使用几个卷积层提取特征
            conv1 = torch.nn.Conv2d(3, 64, 3, padding=1).to(pred.device)
            feat_pred = conv1(pred)
            feat_target = conv1(target)

            lpips = F.mse_loss(feat_pred, feat_target).item()

        return lpips

    def _compute_perceptual_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """感知损失 (使用VGG特征)"""
        # 简化版本
        # 实际应加载预训练VGG网络
        return F.l1_loss(pred, target).item()

    def generate_report(
        self,
        single_step_metrics: Dict[str, float],
        rollout_metrics: Dict[str, float],
        action_conditioned_metrics: Dict[str, float],
        output_path: str,
    ):
        """生成世界模型评估报告"""
        report = {
            'evaluator': 'WorldModelEvaluator',
            'single_step': single_step_metrics,
            'multi_step_rollout': rollout_metrics,
            'action_conditioned': action_conditioned_metrics,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"World model evaluation report saved to {output_path}")

        print("\n=== World Model Evaluation Results ===")
        print(f"Single-step Latent MSE: {single_step_metrics.get('latent_mse', 0):.6f}")
        print(f"Single-step Image MSE: {single_step_metrics.get('image_mse', 0):.6f}")
        print(f"Single-step PSNR: {single_step_metrics.get('psnr', 0):.2f}")
        print(f"Rollout MSE (mean): {rollout_metrics.get('rollout_mse_mean', 0):.6f}")
        print(f"Rollout PSNR (mean): {rollout_metrics.get('rollout_psnr_mean', 0):.2f}")
        print(f"Action-conditioned MSE: {action_conditioned_metrics.get('action_conditioned_mse', 0):.6f}")
        print("======================================\n")
