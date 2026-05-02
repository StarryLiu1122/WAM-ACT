"""
Pretrain Demo
预训练演示

展示如何使用WAM-ACT进行预训练
"""

import torch
from wam_act.models import WAMACT
from wam_act.data import RobotDataset, RobotDataLoader
from wam_act.training import run_pretrain


def main():
    print("=" * 60)
    print("WAM-ACT Pretrain Demo")
    print("=" * 60)

    # 配置
    config = {
        'image_size': 256,
        'latent_dim': 16,
        'transformer_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'action_dim': 7,
        'chunk_size': 16,
        'history_len': 4,
    }

    print("\n1. Creating model...")
    model = WAMACT(**config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\n2. Pretrain forward pass demo...")
    # 创建虚拟输入
    batch_size = 2
    current_image = torch.randn(batch_size, 3, 256, 256)
    next_image = torch.randn(batch_size, 3, 256, 256)
    history_images = torch.randn(batch_size, 4, 3, 256, 256)

    # 前向传播
    pred_latent, target_latent = model.forward_pretrain(
        current_image=current_image,
        next_image=next_image,
        history_images=history_images,
    )

    print(f"   Input shape: {current_image.shape}")
    print(f"   Predicted latent shape: {pred_latent.shape}")
    print(f"   Target latent shape: {target_latent.shape}")

    # 计算损失
    loss = model.get_pretrain_loss(pred_latent, target_latent)
    print(f"   Pretrain loss: {loss.item():.6f}")

    print("\n3. VAE encode/decode demo...")
    # 编码
    latent, mu, logvar = model.vae_encoder(current_image)
    print(f"   Encoded latent shape: {latent.shape}")

    # 解码
    reconstructed = model.vae_decoder.decode(latent)
    print(f"   Reconstructed shape: {reconstructed.shape}")

    print("\n4. Pretrain training demo...")
    print("   (This would normally run for many epochs)")
    print("   Use run_pretrain() for full training")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
