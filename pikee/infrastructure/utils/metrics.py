"""性能指标收集模块.

提供性能监控和指标收集功能。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, Optional

from pikee.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricRecord:
    """指标记录.

    Attributes:
        name: 指标名称
        value: 指标值
        unit: 单位
        timestamp: 时间戳
        tags: 标签
    """

    name: str
    value: float
    unit: str = "count"
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典.

        Returns:
            字典表示
        """
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class MetricsCollector:
    """指标收集器.

    收集和管理应用性能指标。

    Examples:
        >>> collector = MetricsCollector()
        >>> collector.increment("requests_total")
        >>> collector.record("response_time", 0.5, unit="seconds")
        >>> stats = collector.get_stats()
    """

    def __init__(self) -> None:
        """初始化指标收集器."""
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list[float]] = {}
        self._records: list[MetricRecord] = []

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, Any]] = None) -> None:
        """递增计数器.

        Args:
            name: 指标名称
            value: 递增值
            tags: 标签
        """
        self._counters[name] = self._counters.get(name, 0) + value

        record = MetricRecord(name=name, value=self._counters[name], unit="count", tags=tags or {})
        self._records.append(record)

        logger.debug(f"Counter incremented: {name}={self._counters[name]}")

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """设置仪表盘值.

        Args:
            name: 指标名称
            value: 值
            tags: 标签
        """
        self._gauges[name] = value

        record = MetricRecord(name=name, value=value, unit="gauge", tags=tags or {})
        self._records.append(record)

        logger.debug(f"Gauge set: {name}={value}")

    def record(self, name: str, value: float, unit: str = "count", tags: Optional[Dict[str, Any]] = None) -> None:
        """记录直方图数据.

        Args:
            name: 指标名称
            value: 值
            unit: 单位
            tags: 标签
        """
        if name not in self._histograms:
            self._histograms[name] = []

        self._histograms[name].append(value)

        record = MetricRecord(name=name, value=value, unit=unit, tags=tags or {})
        self._records.append(record)

        logger.debug(f"Histogram recorded: {name}={value} {unit}")

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """计时器上下文管理器.

        Args:
            name: 指标名称
            tags: 标签

        Yields:
            None

        Examples:
            >>> collector = MetricsCollector()
            >>> with collector.timer("process_document"):
            ...     # 执行耗时操作
            ...     pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            self.record(name, elapsed_time, unit="seconds", tags=tags)
            logger.debug(f"Timer: {name} took {elapsed_time:.3f}s")

    def get_counter(self, name: str) -> int:
        """获取计数器值.

        Args:
            name: 指标名称

        Returns:
            计数器值
        """
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """获取仪表盘值.

        Args:
            name: 指标名称

        Returns:
            仪表盘值
        """
        return self._gauges.get(name, 0.0)

    def get_histogram(self, name: str) -> list[float]:
        """获取直方图数据.

        Args:
            name: 指标名称

        Returns:
            直方图数据
        """
        return self._histograms.get(name, [])

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """获取直方图统计信息.

        Args:
            name: 指标名称

        Returns:
            统计信息（min, max, avg, count）
        """
        data = self.get_histogram(name)

        if not data:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}

        return {"min": min(data), "max": max(data), "avg": sum(data) / len(data), "count": len(data)}

    def get_stats(self) -> Dict[str, Any]:
        """获取所有统计信息.

        Returns:
            统计信息字典
        """
        histogram_stats = {name: self.get_histogram_stats(name) for name in self._histograms.keys()}

        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": histogram_stats,
            "total_records": len(self._records),
        }

    def reset(self) -> None:
        """重置所有指标."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._records.clear()
        logger.info("Metrics reset")


# 全局指标收集器实例
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器.

    Returns:
        指标收集器实例

    Examples:
        >>> collector = get_metrics_collector()
        >>> collector.increment("documents_processed")
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector() -> None:
    """重置全局指标收集器."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()
