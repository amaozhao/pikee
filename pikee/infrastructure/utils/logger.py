"""日志工具模块.

提供统一的日志配置和工具函数。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from pikee.infrastructure.config.settings import Settings


def setup_logger(name: str, settings: Settings, log_file: Optional[Path] = None) -> logging.Logger:
    """配置并返回日志记录器.

    Args:
        name: 日志记录器名称
        settings: 应用配置
        log_file: 日志文件路径（可选）

    Returns:
        配置好的日志记录器

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> logger = setup_logger(__name__, settings)
        >>> logger.info("应用启动")
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    # 设置日志级别
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果提供了日志文件路径）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 阻止日志传播到父记录器
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器（简化版）.

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("处理开始")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类.

    为类提供日志功能。

    Examples:
        >>> class MyService(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("开始处理")
    """

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器.

        Returns:
            日志记录器
        """
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
