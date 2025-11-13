"""配置管理模块测试.

测试配置加载、验证和管理功能。
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pikee.infrastructure.config.settings import (
    ConfigManager,
    LocalSettings,
    Settings,
    get_config_manager,
    get_settings,
)


class TestLocalSettings:
    """测试本地配置类."""

    def test_local_settings_default_values(self) -> None:
        """测试默认值."""
        with patch.dict(os.environ, {}, clear=True):
            # 使用 _env_file=None 禁用 .env 文件加载
            settings = LocalSettings()

            assert settings.apollo_meta_server_address == "http://localhost:8080"
            assert settings.apollo_app_id == "pike-rag"
            assert settings.apollo_cluster == "default"
            assert settings.apollo_env == "DEV"
            assert settings.apollo_namespaces == ["application"]
            assert settings.env == "dev"
            assert settings.local_dev_mode is False

    def test_local_settings_from_env(self) -> None:
        """测试从环境变量加载."""
        env_vars = {
            "APOLLO_META_SERVER_ADDRESS": "http://test.apollo.com:8080",
            "APOLLO_APP_ID": "test-app",
            "APOLLO_CLUSTER": "test",
            "APOLLO_ENV": "TEST",
            "APOLLO_NAMESPACES": "application,common",
            "LOCAL_DEV_MODE": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # 禁用 .env 文件加载，只使用环境变量
            settings = LocalSettings()

            assert settings.apollo_meta_server_address == "http://test.apollo.com:8080"
            assert settings.apollo_app_id == "test-app"
            assert settings.apollo_cluster == "test"
            assert settings.apollo_env == "TEST"
            assert settings.apollo_namespaces == ["application", "common"]
            assert settings.local_dev_mode is True

    def test_parse_namespaces_from_string(self) -> None:
        """测试命名空间字符串解析."""
        env_vars = {"APOLLO_NAMESPACES": "application, common , other"}

        with patch.dict(os.environ, env_vars, clear=True):
            # 禁用 .env 文件加载，只使用环境变量
            settings = LocalSettings()

            assert settings.apollo_namespaces == ["application", "common", "other"]


class TestSettings:
    """测试应用配置类."""

    def test_settings_required_fields(self) -> None:
        """测试必填字段."""
        with pytest.raises(ValidationError) as exc_info:
            Settings()  # type: ignore

        errors = exc_info.value.errors()
        required_fields = {error["loc"][0] for error in errors}
        assert "openai_api_key" in required_fields
        assert "neo4j_password" in required_fields

    def test_settings_with_minimal_config(self) -> None:
        """测试最小配置."""
        settings = Settings(openai_api_key="sk-test", neo4j_password="password")

        assert settings.openai_api_key == "sk-test"
        assert settings.neo4j_password == "password"
        assert settings.app_name == "PIKEE"
        assert settings.openai_model == "gpt-4-turbo"

    def test_settings_from_local_env(self) -> None:
        """测试从本地环境变量加载."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key",
            "NEO4J_PASSWORD": "test-password",
            "QDRANT_URL": "http://qdrant:6333",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings.from_local_env()

            assert settings.openai_api_key == "sk-test-key"
            assert settings.neo4j_password == "test-password"
            assert settings.qdrant_url == "http://qdrant:6333"

    def test_settings_validation(self) -> None:
        """测试配置验证."""
        # 温度参数超出范围
        with pytest.raises(ValidationError):
            Settings(
                openai_api_key="sk-test",
                neo4j_password="password",
                openai_temperature=3.0,  # 超出 [0.0, 2.0] 范围
            )

        # 最大 token 数为负数
        with pytest.raises(ValidationError):
            Settings(openai_api_key="sk-test", neo4j_password="password", openai_max_tokens=-1)


class TestConfigManager:
    """测试配置管理器."""

    def test_config_manager_local_dev_mode(self) -> None:
        """测试本地开发模式."""
        env_vars = {"LOCAL_DEV_MODE": "true", "OPENAI_API_KEY": "sk-test", "NEO4J_PASSWORD": "password"}

        with patch.dict(os.environ, env_vars, clear=True):
            # Mock LocalSettings 以避免加载 .env 文件
            with patch("pikee.infrastructure.config.settings.LocalSettings") as mock_local_settings:
                mock_instance = MagicMock()
                mock_instance.local_dev_mode = True
                mock_instance.env = "dev"
                mock_local_settings.return_value = mock_instance

                manager = ConfigManager()
                settings = manager.initialize()

                assert settings.openai_api_key == "sk-test"
                assert settings.neo4j_password == "password"
                assert manager._initialized is True
                assert manager.apollo_client is None

    def test_config_manager_singleton(self) -> None:
        """测试单例模式."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2

    @patch("pikee.infrastructure.config.settings.ApolloClient")
    @patch("pikee.infrastructure.config.settings.ApolloSettingsConfig")
    def test_config_manager_apollo_mode(
        self, mock_apollo_settings_config: MagicMock, mock_apollo_client: MagicMock
    ) -> None:
        """测试 Apollo 模式."""
        # Mock Apollo 客户端
        mock_client = MagicMock()
        mock_client.get_value.side_effect = lambda key, namespace: {
            "openai_api_key": "sk-apollo-key",
            "neo4j_password": "apollo-password",
        }.get(key)
        mock_apollo_client.return_value = mock_client

        env_vars = {
            "LOCAL_DEV_MODE": "false",
            "APOLLO_META_SERVER_ADDRESS": "http://apollo:8080",
            "APOLLO_APP_ID": "test-app",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Mock LocalSettings 以避免加载 .env 文件
            with patch("pikee.infrastructure.config.settings.LocalSettings") as mock_local_settings:
                mock_local_instance = MagicMock()
                mock_local_instance.local_dev_mode = False
                mock_local_instance.env = "dev"
                mock_local_instance.apollo_meta_server_address = "http://apollo:8080"
                mock_local_instance.apollo_app_id = "test-app"
                mock_local_instance.apollo_cluster = "default"
                mock_local_instance.apollo_env = "DEV"
                mock_local_instance.apollo_namespaces = ["application"]
                mock_local_instance.apollo_app_secret = None
                mock_local_instance.apollo_using_app_secret = False
                mock_local_settings.return_value = mock_local_instance

                manager = ConfigManager()
                settings = manager.initialize()

                assert mock_apollo_client.called
                assert settings.openai_api_key == "sk-apollo-key"
                assert settings.neo4j_password == "apollo-password"

    def test_config_manager_reload(self) -> None:
        """测试配置重新加载."""
        env_vars = {"LOCAL_DEV_MODE": "true", "OPENAI_API_KEY": "sk-test", "NEO4J_PASSWORD": "password"}

        with patch.dict(os.environ, env_vars, clear=True):
            # Mock LocalSettings
            with patch("pikee.infrastructure.config.settings.LocalSettings") as mock_local_settings:
                mock_instance = MagicMock()
                mock_instance.local_dev_mode = True
                mock_instance.env = "dev"
                mock_local_settings.return_value = mock_instance

                manager = ConfigManager()
                settings1 = manager.initialize()

                # 修改环境变量
                os.environ["OPENAI_API_KEY"] = "sk-new-key"

                # 重新加载
                settings2 = manager.reload()

                assert settings1.openai_api_key == "sk-test"
                assert settings2.openai_api_key == "sk-new-key"


class TestGlobalFunctions:
    """测试全局函数."""

    def test_get_settings(self) -> None:
        """测试 get_settings 函数."""
        env_vars = {"LOCAL_DEV_MODE": "true", "OPENAI_API_KEY": "sk-test", "NEO4J_PASSWORD": "password"}

        with patch.dict(os.environ, env_vars, clear=True):
            # 清除缓存
            get_config_manager.cache_clear()

            # 创建一个真实的 LocalSettings 但禁用 .env 文件
            with patch.object(LocalSettings, "model_config") as mock_config:
                # 修改 model_config 禁用 .env 文件
                mock_config.get.return_value = None

                # 直接创建 ConfigManager 并强制本地开发模式
                manager = get_config_manager()
                manager.local_settings.local_dev_mode = True
                manager._initialized = False  # 确保重新初始化

                settings = get_settings()

                assert isinstance(settings, Settings)
                assert settings.openai_api_key == "sk-test"


# ====================================
# 集成测试
# ====================================


class TestConfigIntegration:
    """集成测试."""

    def test_full_config_flow_local_mode(self) -> None:
        """测试完整配置流程（本地模式）."""
        env_vars = {
            "LOCAL_DEV_MODE": "true",
            "ENV": "dev",
            "OPENAI_API_KEY": "sk-integration-test",
            "NEO4J_PASSWORD": "test-password",
            "QDRANT_URL": "http://qdrant:6333",
            "REDIS_URL": "redis://redis:6379",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # 清除缓存
            get_config_manager.cache_clear()

            with patch.object(LocalSettings, "model_config") as mock_config:
                mock_config.get.return_value = None

                # 直接创建 ConfigManager 并强制本地开发模式
                manager = get_config_manager()
                manager.local_settings.local_dev_mode = True
                manager._initialized = False

                # 获取配置
                settings = get_settings()

                # 验证配置
                assert settings.openai_api_key == "sk-integration-test"
                assert settings.neo4j_password == "test-password"
                assert settings.qdrant_url == "http://qdrant:6333"
                assert settings.redis_url == "redis://redis:6379"

                # 验证默认值
                assert settings.app_name == "PIKEE"
                assert settings.openai_model == "gpt-4-turbo"
                assert settings.chunk_size == 1000

    def test_config_usage_in_application(self) -> None:
        """测试在应用中使用配置."""
        env_vars = {
            "LOCAL_DEV_MODE": "true",
            "OPENAI_API_KEY": "sk-app-test",
            "NEO4J_PASSWORD": "password",
            "MAX_RETRY_ATTEMPTS": "5",
            "CHUNK_SIZE": "2000",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            get_config_manager.cache_clear()

            with patch.object(LocalSettings, "model_config") as mock_config:
                mock_config.get.return_value = None

                # 直接创建 ConfigManager 并强制本地开发模式
                manager = get_config_manager()
                manager.local_settings.local_dev_mode = True
                manager._initialized = False

                settings = get_settings()

                # 模拟应用使用配置
                assert settings.max_retry_attempts == 5
                assert settings.chunk_size == 2000

                # 类型检查
                assert isinstance(settings.max_retry_attempts, int)
                assert isinstance(settings.chunk_size, int)
