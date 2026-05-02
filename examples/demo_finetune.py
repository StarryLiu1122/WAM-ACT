"""
Finetune Demo
微调演示

展示如何使用WAM-ACT进行微调
"""

import torch
from wam_act.models import WAMACT


def main():
    print("=" * 60)
    print("WAM-ACT Finetune Demo")
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

    print("\n2. Finetune forward pass demo...")
    # 创建虚拟输入
    batch_size = 2
    current_image = torch.randn(batch_size, 3, 256, 256)
    target_actions = torch.randn(batch_size, 16, 7)
    future_images = torch.randn(batch_size, 16, 3, 256, 256)
    history_images = torch.randn(batch_size, 4, 3, 256, 256)

    # 前向传播
    outputs = model.forward_finetune(
        current_image=current_image,
        target_actions=target_actions,
        target_future_images=future_images,
        history_images=history_images,
    )

    print(f"   Input shape: {current_image.shape}")
    print(f"   Predicted actions shape: {outputs['pred_actions'].shape}")
    print(f"   Total loss: {outputs['total_loss'].item():.6f}")
    print(f"   Action loss: {outputs['losses']['action'].item():.6f}")
    print(f"   Future loss: {outputs['losses']['future'].item():.6f}")
    print(f"   Consistency loss: {outputs['losses']['consistency'].item():.6f}")

    print("\n3. Action prediction demo...")
    # 推理模式
    with torch.no_grad():
        predictions = model.predict(
            current_image=current_image[:1],
            num_denoising_steps=10,
        )

    print(f"   Predicted actions shape: {predictions['pred_actions'].shape}")
    print(f"   Predicted images shape: {predictions['pred_images'].shape}")

    print("\n4. Flow Matching action head demo...")
    from wam_act.models import FlowMatchingActionHead

    head = FlowMatchingActionHead(token_dim=768, action_dim=7, chunk_size=16)
    action_tokens = torch.randn(batch_size, 16, 768)

    pred_actions, loss = head(action_tokens, target_actions, is_training=True)
    print(f"   Flow matching loss: {loss.item():.6f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
