"""
Logging Utils
日志工具

提供统一的日志记录接口
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
):
    """
    设置日志

    Args:
        log_file: 日志文件路径
        level: 日志级别
        format_str: 日志格式
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name)
