"""
Metrics
通用评估指标计算

包含:
- 动作相关指标
- 视觉相关指标
- 综合指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred_actions: [N, K, action_dim]
        target_actions: [N, K, action_dim]
        pred_images: [N, K, 3, H, W]
        target_images: [N, K, 3, H, W]

    Returns:
        metrics: 指标字典
    """
    metrics = {}

    # 动作指标
    metrics.update(compute_action_metrics(pred_actions, target_actions))

    # 视觉指标
    metrics.update(compute_visual_metrics(pred_images, target_images))

    # 综合指标
    metrics.update(compute_combined_metrics(pred_actions, target_actions, pred_images, target_images))

    return metrics


def compute_action_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """计算动作预测指标"""
    metrics = {}

    # 基础误差
    metrics['action_mse'] = F.mse_loss(pred, target).item()
    metrics['action_mae'] = F.l1_loss(pred, target).item()
    metrics['action_rmse'] = np.sqrt(metrics['action_mse'])

    # 逐维度误差
    dim_mse = F.mse_loss(pred, target, reduction='none').mean(dim=(0, 1))
    for i, mse in enumerate(dim_mse):
        metrics[f'action_dim_{i}_mse'] = mse.item()

    # 平滑度
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    metrics['action_smoothness_mse'] = F.mse_loss(pred_diff, target_diff).item()
    metrics['action_smoothness_mae'] = F.l1_loss(pred_diff, target_diff).item()

    # 分布统计
    metrics['pred_mean'] = pred.mean().item()
    metrics['pred_std'] = pred.std().item()
    metrics['target_mean'] = target.mean().item()
    metrics['target_std'] = target.std().item()

    # 范围检查
    metrics['pred_min'] = pred.min().item()
    metrics['pred_max'] = pred.max().item()

    return metrics


def compute_visual_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """计算视觉预测指标"""
    metrics = {}

    # 像素级误差
    metrics['visual_mse'] = F.mse_loss(pred, target).item()
    metrics['visual_mae'] = F.l1_loss(pred, target).item()
    metrics['visual_rmse'] = np.sqrt(metrics['visual_mse'])

    # PSNR
    mse = F.mse_loss(pred, target)
    if mse > 0:
        metrics['visual_psnr'] = 20 * np.log10(2.0) - 10 * torch.log10(mse).item()
    else:
        metrics['visual_psnr'] = float('inf')

    # 逐通道误差
    for c in range(pred.shape[2]):
        channel_mse = F.mse_loss(pred[:, :, c], target[:, :, c]).item()
        metrics[f'visual_channel_{c}_mse'] = channel_mse

    return metrics


def compute_combined_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
) -> Dict[str, float]:
    """计算动作-视觉联合指标"""
    metrics = {}

    # 动作-视觉一致性 (简化版)
    # 计算动作变化与视觉变化的相关性
    action_changes = target_actions[:, 1:] - target_actions[:, :-1]
    visual_changes = target_images[:, 1:] - target_images[:, :-1]

    # 展平
    action_changes_flat = action_changes.reshape(-1, action_changes.shape[-1])
    visual_changes_flat = visual_changes.reshape(-1, visual_changes.shape[-2] * visual_changes.shape[-1] * visual_changes.shape[-2])

    # 计算协方差 (简化)
    if action_changes_flat.shape[0] > 1:
        cov = torch.matmul(action_changes_flat.T, visual_changes_flat) / action_changes_flat.shape[0]
        metrics['action_visual_covariance'] = cov.norm().item()

    # 综合得分
    metrics['combined_score'] = (
        -metrics.get('action_mse', 0) * 10 +
        metrics.get('visual_psnr', 0) +
        -metrics.get('visual_mse', 0) * 100
    )

    return metrics


def compute_success_rate(
    trajectories: List[Dict],
    threshold: float = 0.05,
) -> float:
    """
    计算任务成功率

    Args:
        trajectories: 轨迹列表，每个包含 'success' 键
        threshold: 成功阈值

    Returns:
        success_rate: 成功率
    """
    successes = [traj.get('success', False) for traj in trajectories]
    return sum(successes) / len(successes) if successes else 0.0


def compute_efficiency(
    trajectories: List[Dict],
    max_steps: int = 100,
) -> Dict[str, float]:
    """
    计算效率指标

    Args:
        trajectories: 轨迹列表
        max_steps: 最大步数

    Returns:
        efficiency: 效率指标
    """
    lengths = [len(traj.get('actions', [])) for traj in trajectories]

    metrics = {
        'mean_steps': np.mean(lengths),
        'std_steps': np.std(lengths),
        'max_steps': max(lengths),
        'min_steps': min(lengths),
        'completion_rate': sum(1 for l in lengths if l < max_steps) / len(lengths),
    }

    return metrics
