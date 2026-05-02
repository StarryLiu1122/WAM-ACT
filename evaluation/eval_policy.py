"""
Policy Evaluator
策略评估器

评估指标:
- 动作MSE/MAE
- 成功率 (任务完成率)
- 动作平滑度
- 时间效率

参考:
- MOTUS (Bi et al., 2025): 流匹配策略评估
- CogACT (Zhao et al., 2025): DiT策略评估
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..models.wam_act import WAMACT
from ..data.robot_dataset import RobotDataset


class PolicyEvaluator:
    """
    策略评估器

    功能:
    1. 在验证集上评估动作预测精度
    2. 计算各种动作质量指标
    3. 生成评估报告
    """

    def __init__(
        self,
        model: WAMACT,
        device: str = 'cuda',
        action_dim: int = 7,
        chunk_size: int = 16,
    ):
        self.model = model.to(device)
        self.device = device
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        评估策略性能

        Args:
            val_loader: 验证数据加载器
            num_batches: 评估的batch数量 (None表示全部)

        Returns:
            metrics: 评估指标字典
        """
        all_pred_actions = []
        all_target_actions = []
        all_pred_images = []
        all_target_images = []

        total_batches = 0

        for batch_idx, batch in enumerate(val_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            current_image = batch['current_image'].to(self.device)
            target_actions = batch['actions'].to(self.device)
            future_images = batch.get('future_images', None)
            if future_images is not None:
                future_images = future_images.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            # 预测
            outputs = self.model.forward_finetune(
                current_image=current_image,
                target_actions=target_actions,
                target_future_images=future_images,
                state=state,
            )

            pred_actions = outputs['pred_actions']

            all_pred_actions.append(pred_actions.cpu())
            all_target_actions.append(target_actions.cpu())

            if future_images is not None and outputs['pred_future_latents'] is not None:
                # 解码预测图像
                pred_latents = outputs['pred_future_latents']
                B, K, seq_len, C = pred_latents.shape
                pred_latents = pred_latents.view(B * K, seq_len, C)
                pred_imgs = self.model.vae_decoder.decode_from_tokens(
                    pred_latents,
                    self.model.image_size // 16,
                    self.model.image_size // 16,
                ).view(B, K, 3, self.model.image_size, self.model.image_size)

                all_pred_images.append(pred_imgs.cpu())
                all_target_images.append(future_images.cpu())

            total_batches += 1

        # 拼接所有结果
        pred_actions = torch.cat(all_pred_actions, dim=0)
        target_actions = torch.cat(all_target_actions, dim=0)

        # 计算指标
        metrics = self._compute_action_metrics(pred_actions, target_actions)

        # 视觉指标
        if all_pred_images:
            pred_images = torch.cat(all_pred_images, dim=0)
            target_images = torch.cat(all_target_images, dim=0)
            visual_metrics = self._compute_visual_metrics(pred_images, target_images)
            metrics.update(visual_metrics)

        metrics['num_evaluated_batches'] = total_batches

        return metrics

    def _compute_action_metrics(
        self,
        pred: torch.Tensor,  # [N, K, action_dim]
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """计算动作预测指标"""
        metrics = {}

        # MSE
        metrics['action_mse'] = F.mse_loss(pred, target).item()

        # MAE
        metrics['action_mae'] = F.l1_loss(pred, target).item()

        # RMSE
        metrics['action_rmse'] = np.sqrt(metrics['action_mse'])

        # 逐维度MSE
        dim_mse = F.mse_loss(pred, target, reduction='none').mean(dim=(0, 1))
        for i, mse in enumerate(dim_mse):
            metrics[f'action_dim_{i}_mse'] = mse.item()

        # 动作平滑度 (相邻帧动作差异)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        metrics['action_smoothness'] = F.mse_loss(pred_diff, target_diff).item()

        # 动作范围检查
        pred_min = pred.min().item()
        pred_max = pred.max().item()
        metrics['pred_action_min'] = pred_min
        metrics['pred_action_max'] = pred_max

        # 动作分布统计
        metrics['pred_action_mean'] = pred.mean().item()
        metrics['pred_action_std'] = pred.std().item()

        return metrics

    def _compute_visual_metrics(
        self,
        pred: torch.Tensor,  # [N, K, 3, H, W]
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """计算视觉预测指标"""
        metrics = {}

        # MSE
        metrics['visual_mse'] = F.mse_loss(pred, target).item()

        # PSNR
        mse = F.mse_loss(pred, target)
        metrics['visual_psnr'] = 20 * np.log10(2.0) - 10 * torch.log10(mse).item()

        # SSIM (简化版)
        metrics['visual_ssim'] = self._compute_ssim(pred, target)

        # LPIPS (需要预训练网络，简化处理)
        # metrics['visual_lpips'] = ...

        return metrics

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """简化版SSIM计算"""
        # 这里使用简化版本，实际应使用完整SSIM实现
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred.view(-1, 3, pred.shape[-2], pred.shape[-1]), window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target.view(-1, 3, target.shape[-2], target.shape[-1]), window_size, stride=1, padding=window_size//2)

        sigma_pred_sq = F.avg_pool2d(pred.view(-1, 3, pred.shape[-2], pred.shape[-1]) ** 2, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
        sigma_target_sq = F.avg_pool2d(target.view(-1, 3, target.shape[-2], target.shape[-1]) ** 2, window_size, stride=1, padding=window_size//2) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(
            pred.view(-1, 3, pred.shape[-2], pred.shape[-1]) * target.view(-1, 3, target.shape[-2], target.shape[-1]),
            window_size, stride=1, padding=window_size//2
        ) - mu_pred * mu_target

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) /                ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2))

        return ssim.mean().item()

    def generate_report(self, metrics: Dict[str, float], output_path: str):
        """生成评估报告"""
        report = {
            'evaluator': 'PolicyEvaluator',
            'metrics': metrics,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to {output_path}")

        # 打印关键指标
        print("\n=== Evaluation Results ===")
        print(f"Action MSE: {metrics.get('action_mse', 0):.6f}")
        print(f"Action MAE: {metrics.get('action_mae', 0):.6f}")
        print(f"Action RMSE: {metrics.get('action_rmse', 0):.6f}")
        print(f"Visual MSE: {metrics.get('visual_mse', 0):.6f}")
        print(f"Visual PSNR: {metrics.get('visual_psnr', 0):.2f}")
        print("===========================\n")


class SimulatedPolicyEvaluator(PolicyEvaluator):
    """
    模拟环境策略评估器

    在模拟环境中 rollout 策略，评估任务成功率
    """

    def __init__(self, model: WAMACT, env, device: str = 'cuda'):
        super().__init__(model, device)
        self.env = env

    @torch.no_grad()
    def rollout(
        self,
        instruction: str,
        max_steps: int = 100,
        render: bool = False,
    ) -> Dict[str, any]:
        """
        在环境中 rollout 策略

        Args:
            instruction: 语言指令
            max_steps: 最大步数
            render: 是否渲染

        Returns:
            trajectory: 包含观测、动作、奖励的轨迹
        """
        obs = self.env.reset(instruction=instruction)

        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'success': False,
        }

        for step in range(max_steps):
            # 准备输入
            current_image = torch.from_numpy(obs['image']).unsqueeze(0).to(self.device)
            state = torch.from_numpy(obs['state']).unsqueeze(0).to(self.device) if 'state' in obs else None

            # 预测Action Chunk
            outputs = self.model.predict(
                current_image=current_image,
                state=state,
                num_denoising_steps=10,
            )

            action_chunk = outputs['pred_actions'][0]  # [K, action_dim]

            # 执行动作 (只执行第一个动作，然后重新规划)
            action = action_chunk[0].cpu().numpy()

            obs, reward, done, info = self.env.step(action)

            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)

            if render:
                self.env.render()

            if done:
                trajectory['success'] = info.get('success', False)
                break

        return trajectory
