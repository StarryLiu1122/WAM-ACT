"""
Architecture Visualization
架构可视化脚本

生成模型架构的详细可视化图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def plot_detailed_architecture(save_path: str = 'architecture_detailed.png'):
    """绘制详细的架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(24, 32))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 32)
    ax.axis('off')

    colors = {
        'input': '#E3F2FD',
        'encoder': '#FFF8E1',
        'transformer': '#FCE4EC',
        'head': '#E8F5E9',
        'output': '#F3E5F5',
        'loss': '#FFF3E0',
        'arrow': '#424242',
    }

    def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.3",
                             facecolor=color, edgecolor='#333', linewidth=2 if bold else 1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                color='#333', weight=weight, wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, color='#555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # 标题
    ax.text(12, 31, 'WAM-ACT Detailed Architecture', ha='center', va='center',
            fontsize=18, weight='bold', color='#1a1a2e')
    ax.text(12, 30.3, 'World-Action Model with Adaptive Causal Transformer',
            ha='center', va='center', fontsize=12, color='#555')

    # 输入层
    draw_box(ax, 1, 28, 5, 1.5, 'Current Image\n$o_t$ (3x256x256)', colors['input'], bold=True)
    draw_box(ax, 7, 28, 5, 1.5, 'Language Instruction\n$l$ (Text Tokens)', colors['input'], bold=True)
    draw_box(ax, 13, 28, 5, 1.5, 'Proprioception State\n$s_t$ (7-dim)', colors['input'], bold=True)
    draw_box(ax, 19, 28, 4, 1.5, 'History Frames\n$\{o_{t-4:t}\}$', colors['input'], bold=True)

    # 编码器层
    draw_box(ax, 1.5, 25.5, 4, 1.5, 'VAE Encoder\n(3->16 channels)', colors['encoder'])
    draw_box(ax, 7.5, 25.5, 4, 1.5, 'Text Encoder\n(Vocab->768-dim)', colors['encoder'])
    draw_box(ax, 13.5, 25.5, 4, 1.5, 'State Encoder\n(MLP: 7->768)', colors['encoder'])

    # Token序列
    ax.text(1, 24.2, 'Token Sequence:', fontsize=11, weight='bold')

    tokens = [
        ('Instr\n$T_l$', '#C8E6C9'),
        ('Hist Vis\n$T_{o,hist}$', '#C8E6C9'),
        ('Curr Vis\n$T_{o,curr}$', '#C8E6C9'),
        ('State\n$T_s$', '#C8E6C9'),
        ('Action\n$T_a$', '#FFCCBC'),
        ('Future\n$T_f$', '#D1C4E9'),
    ]

    for i, (text, color) in enumerate(tokens):
        draw_box(ax, 0.5 + i*3.8, 22.5, 3.5, 1.2, text, color, fontsize=8)

    # Transformer Blocks
    ax.text(1, 21.5, 'Adaptive Causal Transformer Blocks:', fontsize=11, weight='bold')

    for i in range(6):
        y = 19.5 - i*1.2
        block_text = f'Block {i+1}: Multi-Head Causal Attn + AdaLN-Zero + Cross-Modal Attn'
        draw_box(ax, 1, y, 21, 1, block_text, colors['transformer'], fontsize=9)

        # 标注创新点
        if i == 0:
            ax.text(22.5, y+0.5, 'RoPE\nPos Enc', fontsize=7, color='#E91E63')
        elif i == 1:
            ax.text(22.5, y+0.5, 'Action\nRouting', fontsize=7, color='#E91E63')
        elif i == 2:
            ax.text(22.5, y+0.5, 'KV-Cache\nReady', fontsize=7, color='#E91E63')

    # 输出头
    draw_box(ax, 1, 12, 5, 1.5, 'Vision Pred Head\n(768->16)', colors['head'])
    draw_box(ax, 7, 12, 5, 1.5, 'Action Head\n(Flow Matching)', colors['head'], bold=True)
    draw_box(ax, 13, 12, 5, 1.5, 'Future Pred Head\n(768->16)', colors['head'])

    # 输出
    draw_box(ax, 1, 9.5, 5, 1.5, 'Predicted Next Frame\n$\hat{z}_{t+1}$ (16x16x16)', colors['output'], bold=True)
    draw_box(ax, 7, 9.5, 5, 1.5, 'Action Chunk\n$\hat{A}_{t:t+16}$ (16x7)', colors['output'], bold=True)
    draw_box(ax, 13, 9.5, 5, 1.5, 'Future Frames\n$\{\hat{z}_{t+k}\}$', colors['output'], bold=True)

    # 解码器
    draw_box(ax, 1, 7, 5, 1.5, 'VAE Decoder\n(16->3 channels)', colors['encoder'])

    # 最终输出
    draw_box(ax, 1, 4.5, 5, 1.5, 'Predicted Image\n$\hat{o}_{t+1}$ (3x256x256)', colors['output'], bold=True)

    # 损失
    draw_box(ax, 7, 4.5, 7, 1.5, 
             'Finetune Loss: $L_{action} + L_{future} + L_{consistency}$',
             colors['loss'], bold=True)

    # 创新点框
    innovations = [
        '1. Diffusion Forcing: Independent noise per frame',
        '2. Action-Aware Routing: Dynamic attention head selection',
        '3. Sparse Prediction: Keyframe-only future prediction',
        '4. Consistency Loss: Bidirectional action-visual constraint',
        '5. Streaming Inference: KV-Cache for real-time rollout',
    ]

    ax.text(1, 3, 'Key Innovations:', fontsize=11, weight='bold', color='#E91E63')
    for i, text in enumerate(innovations):
        ax.text(1, 2.3 - i*0.4, f'  {text}', fontsize=9, color='#333')

    # 绘制箭头
    # 输入->编码器
    draw_arrow(ax, 3.5, 28, 3.5, 27)
    draw_arrow(ax, 9.5, 28, 9.5, 27)
    draw_arrow(ax, 15.5, 28, 15.5, 27)

    # 编码器->Token
    draw_arrow(ax, 3.5, 25.5, 2.3, 23.7)
    draw_arrow(ax, 9.5, 25.5, 8.1, 23.7)
    draw_arrow(ax, 15.5, 25.5, 13.9, 23.7)

    # Token->Transformer
    for i in range(6):
        x = 2.3 + i*3.8
        draw_arrow(ax, x, 22.5, x, 21.5)

    # Transformer内部
    for i in range(5):
        y1 = 19.5 - i*1.2
        y2 = 19.5 - (i+1)*1.2
        draw_arrow(ax, 11.5, y1, 11.5, y2+1)

    # Transformer->输出头
    draw_arrow(ax, 3.5, 13.5, 3.5, 13.5)
    draw_arrow(ax, 9.5, 13.5, 9.5, 13.5)
    draw_arrow(ax, 15.5, 13.5, 15.5, 13.5)

    # 输出头->输出
    draw_arrow(ax, 3.5, 12, 3.5, 11)
    draw_arrow(ax, 9.5, 12, 9.5, 11)
    draw_arrow(ax, 15.5, 12, 15.5, 11)

    # 视觉输出->解码器
    draw_arrow(ax, 3.5, 9.5, 3.5, 8.5)

    # 解码器->最终图像
    draw_arrow(ax, 3.5, 7, 3.5, 6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Architecture diagram saved to {save_path}")


if __name__ == '__main__':
    plot_detailed_architecture()
