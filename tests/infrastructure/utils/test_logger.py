"""日志工具测试."""

import logging
from pathlib import Path

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.utils.logger import LoggerMixin, get_logger, setup_logger


class TestSetupLogger:
    """测试 setup_logger 函数."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Mock 配置."""
        return Settings(log_level="INFO", openai_api_key="test-key", neo4j_password="test-password")

    def test_setup_logger_basic(self, mock_settings: Settings) -> None:
        """测试基本日志配置."""
        logger = setup_logger("test_logger", mock_settings)

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_with_debug_level(self) -> None:
        """测试 DEBUG 级别配置."""
        settings = Settings(log_level="DEBUG", openai_api_key="test-key", neo4j_password="test-password")
        logger = setup_logger("debug_logger", settings)

        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self, mock_settings: Settings, tmp_path: Path) -> None:
        """测试文件日志配置."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("file_logger", mock_settings, log_file=log_file)

        # 验证文件处理器已添加
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0

        # 验证日志文件已创建
        assert log_file.parent.exists()

    def test_setup_logger_no_duplicate_handlers(self, mock_settings: Settings) -> None:
        """测试避免重复配置."""
        logger1 = setup_logger("duplicate_test", mock_settings)
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_logger("duplicate_test", mock_settings)
        handler_count_2 = len(logger2.handlers)

        # 应该返回同一个 logger，且处理器数量不增加
        assert logger1 is logger2
        assert handler_count_1 == handler_count_2

    def test_setup_logger_propagate_false(self, mock_settings: Settings) -> None:
        """测试日志不传播到父记录器."""
        logger = setup_logger("no_propagate", mock_settings)
        assert logger.propagate is False


class TestGetLogger:
    """测试 get_logger 函数."""

    def test_get_logger(self) -> None:
        """测试获取日志记录器."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_returns_same_instance(self) -> None:
        """测试返回同一个实例."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        assert logger1 is logger2


class TestLoggerMixin:
    """测试 LoggerMixin 混入类."""

    def test_logger_property(self) -> None:
        """测试日志属性."""

        class TestService(LoggerMixin):
            """测试服务."""

            pass

        service = TestService()
        logger = service.logger

        assert isinstance(logger, logging.Logger)
        expected_name = f"{TestService.__module__}.{TestService.__name__}"
        assert logger.name == expected_name

    def test_logger_mixin_in_class(self) -> None:
        """测试在类中使用混入."""

        class MyProcessor(LoggerMixin):
            """示例处理器."""

            def process(self) -> str:
                """处理方法."""
                self.logger.info("Processing...")
                return "done"

        processor = MyProcessor()
        result = processor.process()

        assert result == "done"
        assert hasattr(processor, "logger")


class TestLoggerIntegration:
    """日志集成测试."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Mock 配置."""
        return Settings(log_level="INFO", openai_api_key="test-key", neo4j_password="test-password")

    def test_log_messages(self, mock_settings: Settings, capsys: pytest.CaptureFixture) -> None:
        """测试日志消息记录."""
        logger = setup_logger("integration_test", mock_settings)
        logger.info("Test info message")
        logger.warning("Test warning message")

        # 验证日志消息（从 stdout 捕获）
        captured = capsys.readouterr()
        assert "Test info message" in captured.out
        assert "Test warning message" in captured.out

    def test_log_formatting(self, mock_settings: Settings) -> None:
        """测试日志格式."""
        logger = setup_logger("format_test", mock_settings)

        # 验证格式器已设置
        for handler in logger.handlers:
            assert handler.formatter is not None
            assert "%(asctime)s" in handler.formatter._fmt
            assert "%(levelname)s" in handler.formatter._fmt
