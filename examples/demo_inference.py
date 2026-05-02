"""
Inference Demo
推理演示

展示如何使用训练好的WAM-ACT进行推理
"""

import torch
from wam_act.models import WAMACT


def main():
    print("=" * 60)
    print("WAM-ACT Inference Demo")
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

    print("\n1. Loading model...")
    model = WAMACT(**config)
    model.eval()

    print("\n2. Simulating robot observation...")
    # 模拟机器人当前观测
    current_image = torch.randn(1, 3, 256, 256)
    state = torch.randn(1, 7)
    instruction = "pick up the red cube"

    print(f"   Current image shape: {current_image.shape}")
    print(f"   State shape: {state.shape}")
    print(f"   Instruction: {instruction}")

    print("\n3. Running inference...")
    with torch.no_grad():
        predictions = model.predict(
            current_image=current_image,
            state=state,
            num_denoising_steps=10,
        )

    pred_actions = predictions['pred_actions']
    pred_images = predictions['pred_images']

    print(f"   Predicted action chunk shape: {pred_actions.shape}")
    print(f"   [1, chunk_size={pred_actions.shape[1]}, action_dim={pred_actions.shape[2]}]")
    print(f"   Predicted future images shape: {pred_images.shape}")
    print(f"   [1, chunk_size={pred_images.shape[1]}, 3, H, W]")

    print("\n4. Action analysis...")
    # 分析预测的动作
    print(f"   First action: {pred_actions[0, 0].numpy()}")
    print(f"   Action mean: {pred_actions.mean().item():.4f}")
    print(f"   Action std: {pred_actions.std().item():.4f}")
    print(f"   Action min: {pred_actions.min().item():.4f}")
    print(f"   Action max: {pred_actions.max().item():.4f}")

    print("\n5. Streaming inference demo...")
    print("   (Simulating real-time robot control)")

    # 模拟流式推理
    history = []
    for step in range(5):
        # 获取新观测
        obs = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            pred = model.predict(
                current_image=obs,
                num_denoising_steps=5,  # 快速推理
            )

        action = pred['pred_actions'][0, 0]  # 执行第一个动作
        history.append(action)

        print(f"   Step {step+1}: action executed")

    print(f"\n   Executed {len(history)} actions in streaming mode")

    print("\n" + "=" * 60)
    print("Inference demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
