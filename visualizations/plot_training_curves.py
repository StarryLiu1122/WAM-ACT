"""
Training Curves Visualization
训练曲线可视化

绘制训练过程中的损失曲线和指标变化
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str = 'training_curves.png',
):
    """
    绘制训练曲线

    Args:
        metrics_history: 指标历史记录
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    ax = axes[0, 0]
    if 'train/loss' in metrics_history:
        ax.plot(metrics_history['train/loss'], label='Train Loss', color='#2196F3')
    if 'val/loss' in metrics_history:
        ax.plot(metrics_history['val/loss'], label='Val Loss', color='#F44336')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 动作MSE
    ax = axes[0, 1]
    if 'train/action_mse' in metrics_history:
        ax.plot(metrics_history['train/action_mse'], label='Train', color='#2196F3')
    if 'val/action_mse' in metrics_history:
        ax.plot(metrics_history['val/action_mse'], label='Val', color='#F44336')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Action Prediction MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 视觉PSNR
    ax = axes[1, 0]
    if 'train/visual_psnr' in metrics_history:
        ax.plot(metrics_history['train/visual_psnr'], label='Train', color='#2196F3')
    if 'val/visual_psnr' in metrics_history:
        ax.plot(metrics_history['val/visual_psnr'], label='Val', color='#F44336')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Visual Prediction PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 学习率
    ax = axes[1, 1]
    if 'train/lr' in metrics_history:
        ax.plot(metrics_history['train/lr'], color='#4CAF50')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_comparison(
    methods: Dict[str, Dict[str, List[float]]],
    metric_name: str = 'action_mse',
    save_path: str = 'comparison.png',
):
    """
    比较不同方法的性能

    Args:
        methods: {method_name: {metric: [values]}}
        metric_name: 要比较的指标
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for i, (method_name, metrics) in enumerate(methods.items()):
        if metric_name in metrics:
            ax.plot(metrics[metric_name], label=method_name, 
                   color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Method Comparison: {metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


if __name__ == '__main__':
    # 示例数据
    metrics = {
        'train/loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val/loss': [1.1, 0.9, 0.7, 0.6, 0.5],
        'train/action_mse': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val/action_mse': [0.55, 0.45, 0.35, 0.3, 0.25],
    }

    plot_training_curves(metrics)
