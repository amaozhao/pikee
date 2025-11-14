"""性能指标测试."""

import time

import pytest

from pikee.infrastructure.utils.metrics import (
    MetricRecord,
    MetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestMetricRecord:
    """测试 MetricRecord 数据类."""

    def test_metric_record_creation(self) -> None:
        """测试指标记录创建."""
        record = MetricRecord(
            name="test_metric",
            value=100.5,
            unit="count",
            tags={"env": "test"},
        )

        assert record.name == "test_metric"
        assert record.value == 100.5
        assert record.unit == "count"
        assert record.tags == {"env": "test"}
        assert record.timestamp is not None

    def test_metric_record_to_dict(self) -> None:
        """测试转换为字典."""
        record = MetricRecord(
            name="test_metric",
            value=50.0,
            unit="seconds",
        )

        data = record.to_dict()

        assert data["name"] == "test_metric"
        assert data["value"] == 50.0
        assert data["unit"] == "seconds"
        assert "timestamp" in data
        assert "tags" in data


class TestMetricsCollector:
    """测试 MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """创建指标收集器."""
        return MetricsCollector()

    def test_init(self, collector: MetricsCollector) -> None:
        """测试初始化."""
        assert collector.get_counter("any") == 0
        assert collector.get_gauge("any") == 0.0
        assert collector.get_histogram("any") == []

    def test_increment_counter(self, collector: MetricsCollector) -> None:
        """测试递增计数器."""
        collector.increment("requests")
        assert collector.get_counter("requests") == 1

        collector.increment("requests", value=5)
        assert collector.get_counter("requests") == 6

    def test_increment_counter_with_tags(self, collector: MetricsCollector) -> None:
        """测试带标签的计数器."""
        collector.increment("api_calls", tags={"endpoint": "/api/v1"})
        assert collector.get_counter("api_calls") == 1

    def test_set_gauge(self, collector: MetricsCollector) -> None:
        """测试设置仪表盘."""
        collector.set_gauge("memory_usage", 512.5)
        assert collector.get_gauge("memory_usage") == 512.5

        collector.set_gauge("memory_usage", 1024.0)
        assert collector.get_gauge("memory_usage") == 1024.0

    def test_record_histogram(self, collector: MetricsCollector) -> None:
        """测试记录直方图."""
        collector.record("response_time", 0.5, unit="seconds")
        collector.record("response_time", 1.2, unit="seconds")
        collector.record("response_time", 0.8, unit="seconds")

        data = collector.get_histogram("response_time")
        assert len(data) == 3
        assert 0.5 in data
        assert 1.2 in data
        assert 0.8 in data

    def test_get_histogram_stats(self, collector: MetricsCollector) -> None:
        """测试直方图统计."""
        collector.record("latency", 10.0)
        collector.record("latency", 20.0)
        collector.record("latency", 30.0)

        stats = collector.get_histogram_stats("latency")

        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["avg"] == 20.0
        assert stats["count"] == 3

    def test_get_histogram_stats_empty(self, collector: MetricsCollector) -> None:
        """测试空直方图统计."""
        stats = collector.get_histogram_stats("nonexistent")

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["avg"] == 0.0
        assert stats["count"] == 0

    def test_timer_context_manager(self, collector: MetricsCollector) -> None:
        """测试计时器上下文管理器."""
        with collector.timer("process_duration"):
            time.sleep(0.1)  # 模拟耗时操作

        data = collector.get_histogram("process_duration")
        assert len(data) == 1
        assert data[0] >= 0.1  # 至少 0.1 秒

    def test_timer_with_tags(self, collector: MetricsCollector) -> None:
        """测试带标签的计时器."""
        with collector.timer("operation", tags={"type": "batch"}):
            time.sleep(0.05)

        data = collector.get_histogram("operation")
        assert len(data) == 1

    def test_get_stats(self, collector: MetricsCollector) -> None:
        """测试获取所有统计信息."""
        collector.increment("counter1")
        collector.set_gauge("gauge1", 100.0)
        collector.record("histogram1", 50.0)

        stats = collector.get_stats()

        assert "counters" in stats
        assert "gauges" in stats
        assert "histograms" in stats
        assert "total_records" in stats

        assert stats["counters"]["counter1"] == 1
        assert stats["gauges"]["gauge1"] == 100.0
        assert "histogram1" in stats["histograms"]

    def test_reset(self, collector: MetricsCollector) -> None:
        """测试重置指标."""
        collector.increment("counter")
        collector.set_gauge("gauge", 50.0)
        collector.record("histogram", 10.0)

        collector.reset()

        assert collector.get_counter("counter") == 0
        assert collector.get_gauge("gauge") == 0.0
        assert collector.get_histogram("histogram") == []

        stats = collector.get_stats()
        assert stats["total_records"] == 0


class TestGlobalMetricsCollector:
    """测试全局指标收集器."""

    def test_get_metrics_collector(self) -> None:
        """测试获取全局收集器."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        # 应该返回同一个实例
        assert collector1 is collector2

    def test_reset_metrics_collector(self) -> None:
        """测试重置全局收集器."""
        collector = get_metrics_collector()
        collector.increment("test_counter")

        assert collector.get_counter("test_counter") == 1

        reset_metrics_collector()

        assert collector.get_counter("test_counter") == 0


class TestMetricsCollectorEdgeCases:
    """测试边界情况."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """创建指标收集器."""
        return MetricsCollector()

    def test_multiple_counters(self, collector: MetricsCollector) -> None:
        """测试多个计数器."""
        collector.increment("counter1")
        collector.increment("counter2", value=5)
        collector.increment("counter3", value=10)

        assert collector.get_counter("counter1") == 1
        assert collector.get_counter("counter2") == 5
        assert collector.get_counter("counter3") == 10

    def test_large_histogram(self, collector: MetricsCollector) -> None:
        """测试大量数据的直方图."""
        for i in range(1000):
            collector.record("large_histogram", float(i))

        data = collector.get_histogram("large_histogram")
        assert len(data) == 1000

        stats = collector.get_histogram_stats("large_histogram")
        assert stats["min"] == 0.0
        assert stats["max"] == 999.0
        assert stats["count"] == 1000

    def test_negative_values(self, collector: MetricsCollector) -> None:
        """测试负值."""
        collector.set_gauge("temperature", -10.5)
        collector.record("profit", -50.0)

        assert collector.get_gauge("temperature") == -10.5

        stats = collector.get_histogram_stats("profit")
        assert stats["min"] == -50.0

