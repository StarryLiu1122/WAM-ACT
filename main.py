"""
WAM-ACT Main Entry Point
主入口文件

使用方式:
    python main.py --mode pretrain --config configs/pretrain_config.yaml
    python main.py --mode finetune --config configs/finetune_config.yaml
    python main.py --mode eval --checkpoint outputs/finetune/checkpoints/best.pt
"""

import argparse
import torch
from pathlib import Path

from utils.config import Config
from utils.logging_utils import setup_logging, get_logger
from models.wam_act import WAMACT
from data.robot_dataset import RobotDataLoader
from training.pretrain import PretrainTrainer, run_pretrain
from training.finetune import FinetuneTrainer, run_finetune
from evaluation.eval_policy import PolicyEvaluator
from evaluation.eval_world_model import WorldModelEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='WAM-ACT: World-Action Model')

    parser.add_argument('--mode', type=str, required=True,
                       choices=['pretrain', 'finetune', 'eval', 'predict'],
                       help='运行模式')

    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                       help='配置文件路径')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径 (eval/predict模式)')

    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录 (覆盖配置)')

    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (覆盖配置)')

    parser.add_argument('--device', type=str, default=None,
                       help='设备 (覆盖配置)')

    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = Config.from_yaml(args.config)

    # 命令行参数覆盖配置
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.device:
        config.train.device = args.device

    # 设置日志
    log_file = Path(config.train.output_dir) / 'train.log'
    setup_logging(str(log_file))
    logger = get_logger('main')

    logger.info(f"Starting WAM-ACT in {args.mode} mode")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {config.train.output_dir}")

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'pretrain':
        # 预训练
        logger.info("Starting pretraining...")

        trainer = run_pretrain(
            data_dir=config.data.data_dir,
            output_dir=config.train.output_dir,
            batch_size=config.data.batch_size,
            num_epochs=config.train.pretrain_epochs,
            lr=config.train.pretrain_lr,
            image_size=config.model.image_size,
            **config.model.__dict__,
        )

        logger.info("Pretraining completed!")

    elif args.mode == 'finetune':
        # 微调
        logger.info("Starting finetuning...")

        if not config.train.pretrained_path:
            raise ValueError("pretrained_path must be specified for finetuning")

        trainer = run_finetune(
            data_dir=config.data.data_dir,
            pretrained_path=config.train.pretrained_path,
            output_dir=config.train.output_dir,
            batch_size=config.data.batch_size,
            num_epochs=config.train.finetune_epochs,
            lr=config.train.finetune_lr,
            freeze_encoder=config.train.freeze_encoder,
            **config.model.__dict__,
        )

        logger.info("Finetuning completed!")

    elif args.mode == 'eval':
        # 评估
        logger.info("Starting evaluation...")

        if not args.checkpoint:
            raise ValueError("checkpoint must be specified for evaluation")

        # 加载模型
        model = WAMACT(**config.model.__dict__)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # 创建数据加载器
        val_loader = RobotDataLoader.create_dataloader(
            data_dir=config.data.data_dir,
            split='val',
            batch_size=config.data.batch_size,
            **config.data.__dict__,
        )

        # 策略评估
        policy_eval = PolicyEvaluator(model, device=device)
        policy_metrics = policy_eval.evaluate(val_loader, num_batches=config.eval.num_eval_batches)

        # 世界模型评估
        world_eval = WorldModelEvaluator(model, device=device)
        single_step_metrics = world_eval.evaluate_single_step(val_loader, num_batches=config.eval.num_eval_batches)
        rollout_metrics = world_eval.evaluate_multi_step_rollout(
            val_loader,
            rollout_length=config.eval.rollout_length,
            num_episodes=config.eval.num_eval_episodes,
        )
        action_conditioned_metrics = world_eval.evaluate_action_conditioned(
            val_loader,
            num_batches=config.eval.num_eval_batches,
        )

        # 生成报告
        output_path = Path(config.train.output_dir) / 'evaluation_report.json'
        world_eval.generate_report(
            single_step_metrics,
            rollout_metrics,
            action_conditioned_metrics,
            str(output_path),
        )

        logger.info("Evaluation completed!")

    elif args.mode == 'predict':
        # 预测 (推理)
        logger.info("Starting prediction...")

        if not args.checkpoint:
            raise ValueError("checkpoint must be specified for prediction")

        # 加载模型
        model = WAMACT(**config.model.__dict__)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # 示例推理
        # 这里可以替换为实际的输入数据
        dummy_image = torch.randn(1, 3, config.model.image_size, config.model.image_size).to(device)
        dummy_state = torch.randn(1, config.model.action_dim).to(device)

        with torch.no_grad():
            predictions = model.predict(
                current_image=dummy_image,
                state=dummy_state,
                num_denoising_steps=10,
            )

        logger.info(f"Predicted actions shape: {predictions['pred_actions'].shape}")
        logger.info(f"Predicted images shape: {predictions['pred_images'].shape}")
        logger.info("Prediction completed!")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
