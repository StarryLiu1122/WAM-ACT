"""
Attention Visualization
注意力可视化

可视化Transformer中的注意力权重和路由分布
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_attention_weights(
    attention_weights: torch.Tensor,
    token_labels: Optional[list] = None,
    save_path: str = 'attention_weights.png',
):
    """
    绘制注意力权重热力图

    Args:
        attention_weights: [num_heads, seq_len, seq_len]
        token_labels: Token标签列表
        save_path: 保存路径
    """
    num_heads, seq_len, _ = attention_weights.shape

    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attention_weights[i].cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i+1}')

        if token_labels:
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(token_labels, fontsize=6)

        plt.colorbar(im, ax=ax)

    # 隐藏多余的子图
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention weights saved to {save_path}")


def plot_routing_distribution(
    routing_weights: torch.Tensor,
    modal_types: torch.Tensor,
    save_path: str = 'routing_distribution.png',
):
    """
    绘制Token路由分布

    Args:
        routing_weights: [batch, seq_len, num_experts]
        modal_types: [batch, seq_len] 模态类型
        save_path: 保存路径
    """
    # 取第一个batch
    weights = routing_weights[0].cpu().numpy()
    types = modal_types[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 路由权重分布
    ax1.bar(range(weights.shape[1]), weights.mean(axis=0), color='#2196F3')
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Average Routing Weight')
    ax1.set_title('Expert Routing Distribution')
    ax1.grid(True, alpha=0.3)

    # 模态类型分布
    type_names = {0: 'Instruction', 1: 'Vision', 2: 'State', 3: 'Action'}
    type_counts = np.bincount(types, minlength=4)

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    ax2.bar(range(4), type_counts, color=colors)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([type_names.get(i, f'Type {i}') for i in range(4)])
    ax2.set_ylabel('Token Count')
    ax2.set_title('Modal Type Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Routing distribution saved to {save_path}")


def plot_token_flow(
    token_activations: torch.Tensor,
    save_path: str = 'token_flow.png',
):
    """
    绘制Token在Transformer各层的激活变化

    Args:
        token_activations: [num_layers, seq_len, dim]
        save_path: 保存路径
    """
    num_layers, seq_len, dim = token_activations.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 计算每层的激活范数
    norms = torch.norm(token_activations, dim=-1).cpu().numpy()

    im = ax.imshow(norms, cmap='plasma', aspect='auto')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Token Activation Norms Across Layers')
    plt.colorbar(im, ax=ax, label='L2 Norm')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Token flow saved to {save_path}")


if __name__ == '__main__':
    # 示例数据
    attn = torch.softmax(torch.randn(8, 100, 100), dim=-1)
    plot_attention_weights(attn)

    routing = torch.softmax(torch.randn(2, 100, 4), dim=-1)
    modal = torch.randint(0, 4, (2, 100))
    plot_routing_distribution(routing, modal)
