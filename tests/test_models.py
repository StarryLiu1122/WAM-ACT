"""
Model Unit Tests
模型单元测试
"""

import torch
import pytest
from wam_act.models import (
    WAMACT,
    AdaptiveCausalTransformer,
    FlowMatchingActionHead,
    ActionAwareTokenRouter,
    VAEEncoder,
    VAEDecoder,
)


class TestVAEEncoder:
    """测试VAE编码器"""

    def test_encode_shape(self):
        encoder = VAEEncoder(latent_dim=16, image_size=256)
        x = torch.randn(2, 3, 256, 256)
        z, mu, logvar = encoder(x)

        assert z.shape == (2, 16, 16, 16)  # 16倍下采样
        assert mu.shape == z.shape
        assert logvar.shape == z.shape

    def test_encode_to_tokens(self):
        encoder = VAEEncoder(latent_dim=16, image_size=256)
        x = torch.randn(2, 3, 256, 256)
        tokens = encoder.encode_to_tokens(x)

        assert tokens.shape == (2, 256, 16)  # 16*16=256 tokens


class TestVAEDecoder:
    """测试VAE解码器"""

    def test_decode_shape(self):
        decoder = VAEDecoder(latent_dim=16, image_size=256)
        z = torch.randn(2, 16, 16, 16)
        x = decoder(z)

        assert x.shape == (2, 3, 256, 256)

    def test_decode_from_tokens(self):
        decoder = VAEDecoder(latent_dim=16, image_size=256)
        tokens = torch.randn(2, 256, 16)
        x = decoder.decode_from_tokens(tokens, 16, 16)

        assert x.shape == (2, 3, 256, 256)


class TestActionAwareTokenRouter:
    """测试Action-Aware Token Router"""

    def test_forward_shape(self):
        router = ActionAwareTokenRouter(dim=768, num_heads=12)
        x = torch.randn(2, 100, 768)
        modal_types = torch.zeros(2, 100, dtype=torch.long)
        modal_types[:, 50:] = 3  # 后半部分为动作token

        output, routing_weights = router(x, modal_types)

        assert output.shape == x.shape
        assert routing_weights.shape == (2, 100, 4)  # 4个专家

    def test_routing_weights_sum(self):
        router = ActionAwareTokenRouter(dim=768, num_heads=12)
        x = torch.randn(2, 100, 768)
        modal_types = torch.zeros(2, 100, dtype=torch.long)

        output, routing_weights = router(x, modal_types)

        # Top-k权重应归一化到1
        assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(2, 100), atol=1e-5)


class TestFlowMatchingActionHead:
    """测试Flow Matching Action Head"""

    def test_forward_training(self):
        head = FlowMatchingActionHead(token_dim=768, action_dim=7, chunk_size=16)
        action_tokens = torch.randn(2, 16, 768)
        target_actions = torch.randn(2, 16, 7)

        pred_actions, loss = head(action_tokens, target_actions, is_training=True)

        assert pred_actions.shape == (2, 16, 7)
        assert loss is not None
        assert loss.item() >= 0

    def test_forward_inference(self):
        head = FlowMatchingActionHead(token_dim=768, action_dim=7, chunk_size=16)
        action_tokens = torch.randn(2, 16, 768)

        pred_actions, loss = head(action_tokens, is_training=False)

        assert pred_actions.shape == (2, 16, 7)
        assert loss is None

    def test_predict_action_chunk(self):
        head = FlowMatchingActionHead(token_dim=768, action_dim=7, chunk_size=16)
        action_tokens = torch.randn(2, 16, 768)

        actions = head.predict_action_chunk(action_tokens, num_steps=10)

        assert actions.shape == (2, 16, 7)


class TestAdaptiveCausalTransformer:
    """测试Adaptive Causal Transformer"""

    def test_forward_shape(self):
        transformer = AdaptiveCausalTransformer(dim=768, num_layers=4, num_heads=12)
        tokens = torch.randn(2, 100, 768)
        cond = torch.randn(2, 1)

        output, aux = transformer(tokens, cond)

        assert output.shape == tokens.shape

    def test_prepare_multimodal_tokens(self):
        transformer = AdaptiveCausalTransformer(dim=768, num_layers=4, num_heads=12)

        instruction = torch.randint(0, 1000, (2, 10))
        vision = torch.randn(2, 4, 256, 16)  # 4帧, 每帧256 tokens, 16 channels
        state = torch.randn(2, 7)
        action = torch.randn(2, 16, 7)

        tokens, modal_types = transformer.prepare_multimodal_tokens(
            instruction_tokens=instruction,
            vision_tokens=vision,
            state_tokens=state,
            action_tokens=action,
        )

        expected_len = 10 + 4*256 + 1 + 16
        assert tokens.shape == (2, expected_len, 768)
        assert modal_types.shape == (2, expected_len)


class TestWAMACT:
    """测试WAM-ACT完整模型"""

    def test_pretrain_forward(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)
        current = torch.randn(2, 3, 64, 64)
        next_img = torch.randn(2, 3, 64, 64)

        pred_latent, target_latent = model.forward_pretrain(current, next_img)

        assert pred_latent.shape == target_latent.shape

    def test_finetune_forward(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)
        current = torch.randn(2, 3, 64, 64)
        actions = torch.randn(2, 16, 7)
        future = torch.randn(2, 16, 3, 64, 64)

        outputs = model.forward_finetune(current, actions, future)

        assert 'pred_actions' in outputs
        assert 'total_loss' in outputs
        assert outputs['pred_actions'].shape == (2, 16, 7)

    def test_predict(self):
        model = WAMACT(image_size=64, latent_dim=16, transformer_dim=256, num_layers=2)
        current = torch.randn(1, 3, 64, 64)

        predictions = model.predict(current, num_denoising_steps=5)

        assert 'pred_actions' in predictions
        assert 'pred_images' in predictions
        assert predictions['pred_actions'].shape == (1, 16, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
