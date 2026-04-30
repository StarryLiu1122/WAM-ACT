"""
Data Module Tests
数据模块单元测试
"""

import torch
import pytest
import numpy as np
from pathlib import Path
import tempfile
import pickle

from wam_act.data import (
    RobotDataset,
    ImagePreprocessor,
    ActionNormalizer,
    InstructionTokenizer,
    DataAugmentor,
)


class TestImagePreprocessor:
    """测试图像预处理器"""

    def test_preprocess_shape(self):
        preprocessor = ImagePreprocessor(image_size=256)
        image = torch.randn(3, 512, 512)

        processed = preprocessor(image, is_training=False)

        assert processed.shape == (3, 256, 256)

    def test_preprocess_range(self):
        preprocessor = ImagePreprocessor(image_size=256)
        image = torch.randn(3, 512, 512)

        processed = preprocessor(image, is_training=False)

        # 归一化后范围应在[-1, 1]附近
        assert processed.min() >= -2.0
        assert processed.max() <= 2.0


class TestActionNormalizer:
    """测试动作归一化器"""

    def test_minmax_normalization(self):
        normalizer = ActionNormalizer(method='minmax', action_dim=7)
        actions = np.random.randn(100, 7) * 10

        normalizer.fit(actions)
        normalized = normalizer.normalize(torch.from_numpy(actions[:10]))

        # 归一化后范围应在[-1, 1]
        assert normalized.min() >= -1.5
        assert normalized.max() <= 1.5

    def test_zscore_normalization(self):
        normalizer = ActionNormalizer(method='zscore', action_dim=7)
        actions = np.random.randn(100, 7)

        normalizer.fit(actions)
        normalized = normalizer.normalize(torch.from_numpy(actions[:10]))

        # 均值应接近0，标准差接近1
        assert abs(normalized.mean().item()) < 0.5
        assert abs(normalized.std().item() - 1.0) < 0.5

    def test_denormalize(self):
        normalizer = ActionNormalizer(method='minmax', action_dim=7)
        actions = np.random.randn(100, 7) * 10

        normalizer.fit(actions)

        original = torch.from_numpy(actions[:10])
        normalized = normalizer.normalize(original)
        denormalized = normalizer.denormalize(normalized)

        # 反归一化应恢复原始值
        assert torch.allclose(original, denormalized, atol=1e-5)


class TestInstructionTokenizer:
    """测试指令Token化器"""

    def test_encode_decode(self):
        tokenizer = InstructionTokenizer(vocab_size=50000, max_length=77)
        text = "pick up the red cube"

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert tokens.shape == (77,)
        assert isinstance(decoded, str)

    def test_batch_encode(self):
        tokenizer = InstructionTokenizer(vocab_size=50000, max_length=77)
        texts = ["pick up the red cube", "place the blue sphere"]

        tokens = tokenizer.batch_encode(texts)

        assert tokens.shape == (2, 77)


class TestDataAugmentor:
    """测试数据增强器"""

    def test_augment_image(self):
        augmentor = DataAugmentor(image_size=256)
        image = torch.randn(3, 256, 256)

        augmented = augmentor.augment_image(image)

        assert augmented.shape == image.shape

    def test_augment_sequence(self):
        augmentor = DataAugmentor(image_size=256)
        images = torch.randn(10, 3, 256, 256)
        actions = torch.randn(10, 7)

        aug_images, aug_actions = augmentor.augment_sequence(images, actions)

        assert aug_images.shape == images.shape
        assert aug_actions.shape == actions.shape


class TestRobotDataset:
    """测试机器人数据集"""

    def test_dataset_loading(self):
        # 创建临时数据
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                'images': np.random.randint(0, 255, (100, 256, 256, 3), dtype=np.uint8),
                'actions': np.random.randn(100, 7),
                'states': np.random.randn(100, 7),
                'instruction': 'pick up the object',
            }

            data_file = Path(tmpdir) / 'episode_0.pkl'
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)

            # 创建数据集
            dataset = RobotDataset(
                data_dir=tmpdir,
                split='train',
                history_len=4,
                chunk_size=16,
            )

            assert len(dataset) > 0

            # 获取一个样本
            sample = dataset[0]

            assert 'current_image' in sample
            assert 'history_images' in sample
            assert 'next_image' in sample
            assert 'future_images' in sample
            assert 'actions' in sample
            assert 'state' in sample

            assert sample['current_image'].shape == (3, 256, 256)
            assert sample['history_images'].shape == (4, 3, 256, 256)
            assert sample['actions'].shape == (16, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
