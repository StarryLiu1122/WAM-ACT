"""
Config
配置管理模块

支持:
- YAML配置文件加载
- 命令行参数覆盖
- 配置验证
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """模型配置"""
    # VAE
    image_size: int = 256
    latent_dim: int = 16
    vae_hidden_dims: list = field(default_factory=lambda: [128, 256, 512, 1024])

    # Transformer
    transformer_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    max_seq_len: int = 4096

    # 任务
    action_dim: int = 7
    chunk_size: int = 16
    history_len: int = 4
    prediction_stride: int = 4

    # 训练
    num_diffusion_steps: int = 1000
    num_flow_steps: int = 50

    # 多视角
    num_views: int = 1


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = './data'
    batch_size: int = 32
    num_workers: int = 4
    history_len: int = 4
    chunk_size: int = 16
    image_size: int = 256
    action_dim: int = 7
    num_views: int = 1

    # 增强
    use_augmentation: bool = True
    color_jitter: tuple = (0.4, 0.4, 0.4, 0.1)
    action_noise_scale: float = 0.01


@dataclass
class TrainConfig:
    """训练配置"""
    # 预训练
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 32

    # 微调
    finetune_epochs: int = 50
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 16

    # 优化器
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # 调度器
    warmup_ratio: float = 0.1

    # 系统
    device: str = 'cuda'
    use_amp: bool = True
    seed: int = 42

    # 日志
    output_dir: str = './outputs'
    log_every: int = 100
    save_every: int = 1

    # 检查点
    pretrained_path: Optional[str] = None
    freeze_encoder: bool = True


@dataclass
class EvalConfig:
    """评估配置"""
    num_eval_batches: int = 10
    rollout_length: int = 8
    num_eval_episodes: int = 10
    render: bool = False


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """从YAML加载配置"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            train=TrainConfig(**config_dict.get('train', {})),
            eval=EvalConfig(**config_dict.get('eval', {})),
        )

    def to_yaml(self, path: str):
        """保存为YAML"""
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'train': asdict(self.train),
            'eval': asdict(self.eval),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, (ModelConfig, DataConfig, TrainConfig, EvalConfig)):
                    for k, v in value.items():
                        if hasattr(attr, k):
                            setattr(attr, k, v)
                else:
                    setattr(self, key, value)
