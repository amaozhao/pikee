"""应用配置管理模块.

本模块实现基于 Apollo 配置中心的两层配置加载机制：
1. 从本地 .env 文件加载 Apollo 连接配置
2. 从 Apollo 配置中心加载所有业务配置

使用示例:
    ```python
    from pikee.infrastructure.config.settings import get_settings

    settings = get_settings()
    print(settings.openai_api_key)
    ```
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pikee.infrastructure.pyapollo.client import ApolloClient
from pikee.infrastructure.pyapollo.settings import ApolloSettingsConfig

logger = logging.getLogger(__name__)


class LocalSettings(BaseSettings):
    """本地环境变量配置（仅包含 Apollo 连接信息）.

    此类用于从 .env 文件加载 Apollo 配置中心的连接信息。
    加载顺序：
    1. 环境变量（优先级最高）
    2. .env 文件

    Attributes:
        apollo_meta_server_address: Apollo 配置中心服务器地址
        apollo_app_id: Apollo 应用 ID
        apollo_cluster: Apollo 集群名称
        apollo_env: Apollo 环境（DEV/TEST/PROD）
        apollo_namespaces: Apollo 命名空间列表
        apollo_app_secret: Apollo 应用密钥（可选）
        apollo_using_app_secret: 是否使用密钥认证
        env: 当前运行环境
        local_dev_mode: 本地开发模式（不连接 Apollo）
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # 忽略 .env 文件中无法解析的字段
        env_ignore_empty=True,
    )

    # Apollo 配置
    apollo_meta_server_address: str = Field(default="http://localhost:8080", description="Apollo 配置中心地址")
    apollo_app_id: str = Field(default="pike-rag", description="Apollo 应用 ID")
    apollo_cluster: str = Field(default="default", description="Apollo 集群名称")
    apollo_env: str = Field(default="DEV", description="Apollo 环境（DEV/TEST/PROD）")
    apollo_namespaces_str: str = Field(
        default="application", alias="apollo_namespaces", description="Apollo 命名空间列表（逗号分隔或 JSON 数组）"
    )
    apollo_app_secret: Optional[str] = Field(default=None, description="Apollo 应用密钥")
    apollo_using_app_secret: bool = Field(default=False, description="是否使用密钥认证")

    # 环境标识
    env: str = Field(default="dev", description="当前运行环境（dev/test/prod）")

    # 本地开发模式
    local_dev_mode: bool = Field(default=False, description="本地开发模式（设为 true 时不连接 Apollo，从环境变量读取）")

    @property
    def apollo_namespaces(self) -> List[str]:
        """获取解析后的命名空间列表.

        Returns:
            List[str]: 命名空间列表
        """
        v = self.apollo_namespaces_str
        if not v:
            return ["application"]

        # 尝试解析 JSON 格式
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                import json

                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # 逗号分隔格式
        result = [ns.strip() for ns in v.split(",") if ns.strip()]
        return result if result else ["application"]


class Settings(BaseSettings):
    """应用配置（从 Apollo 加载）.

    此类定义了应用运行所需的所有配置项。
    配置来源：
    1. Apollo 配置中心（生产环境）
    2. 环境变量（本地开发模式）
    3. 默认值（fallback）

    Attributes:
        app_name: 应用名称
        app_version: 应用版本
        debug: 调试模式
        log_level: 日志级别

        openai_api_key: OpenAI API Key
        openai_api_base: OpenAI API Base URL
        openai_model: OpenAI 模型名称
        openai_temperature: LLM 温度参数
        openai_max_tokens: LLM 最大 token 数

        qdrant_url: Qdrant 服务地址
        qdrant_collection_chunks: Chunk 向量集合名称
        qdrant_collection_atoms: Atom 向量集合名称
        qdrant_vector_size: 向量维度

        neo4j_uri: Neo4j 连接 URI
        neo4j_user: Neo4j 用户名
        neo4j_password: Neo4j 密码
        neo4j_database: Neo4j 数据库名称

        redis_url: Redis 连接 URL

        max_retry_attempts: 最大重试次数
        retry_backoff_base: 重试退避基数
        retry_max_wait: 最大等待时间（秒）

        chunk_size: 文档切分大小
        chunk_overlap: 文档切分重叠大小

        enable_sentry: 是否启用 Sentry 错误追踪
        sentry_dsn: Sentry DSN
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # 基础配置
    app_name: str = Field(default="PIKEE", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    # LLM 配置
    openai_api_key: str = Field(..., description="OpenAI API Key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API Base URL")
    openai_model: str = Field(default="gpt-4-turbo", description="OpenAI 模型名称")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM 温度参数")
    openai_max_tokens: int = Field(default=4096, gt=0, description="LLM 最大 token 数")

    # Qdrant 配置
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant 服务地址")
    qdrant_collection_chunks: str = Field(default="pike_chunks", description="Chunk 向量集合名称")
    qdrant_collection_atoms: str = Field(default="pike_atoms", description="Atom 向量集合名称")
    qdrant_vector_size: int = Field(default=1536, gt=0, description="向量维度")

    # Neo4j 配置
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j 连接 URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field(..., description="Neo4j 密码")
    neo4j_database: str = Field(default="neo4j", description="Neo4j 数据库名称")

    # Redis 配置
    redis_url: str = Field(default="redis://localhost:6379", description="Redis 连接 URL")

    # 重试配置
    max_retry_attempts: int = Field(default=3, ge=0, description="最大重试次数")
    retry_backoff_base: float = Field(default=2.0, ge=1.0, description="重试退避基数")
    retry_max_wait: int = Field(default=60, gt=0, description="最大等待时间（秒）")

    # 文档处理配置
    chunk_size: int = Field(default=1000, gt=0, description="文档切分大小")
    chunk_overlap: int = Field(default=200, ge=0, description="文档切分重叠大小")

    # 监控配置
    enable_sentry: bool = Field(default=False, description="是否启用 Sentry 错误追踪")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")

    @classmethod
    def from_apollo(cls, apollo_client: ApolloClient, namespace: str = "application") -> "Settings":
        """从 Apollo 配置中心加载配置.

        Args:
            apollo_client: Apollo 客户端实例
            namespace: Apollo 命名空间

        Returns:
            Settings: 配置实例

        Raises:
            ValueError: 当必需的配置项缺失时
        """
        logger.info(f"从 Apollo 加载配置，命名空间: {namespace}")

        # 定义需要从 Apollo 读取的配置项
        config_keys = [
            # 基础配置
            "app_name",
            "app_version",
            "debug",
            "log_level",
            # LLM 配置
            "openai_api_key",
            "openai_api_base",
            "openai_model",
            "openai_temperature",
            "openai_max_tokens",
            # Qdrant 配置
            "qdrant_url",
            "qdrant_collection_chunks",
            "qdrant_collection_atoms",
            "qdrant_vector_size",
            # Neo4j 配置
            "neo4j_uri",
            "neo4j_user",
            "neo4j_password",
            "neo4j_database",
            # Redis 配置
            "redis_url",
            # 重试配置
            "max_retry_attempts",
            "retry_backoff_base",
            "retry_max_wait",
            # 文档处理配置
            "chunk_size",
            "chunk_overlap",
            # 监控配置
            "enable_sentry",
            "sentry_dsn",
        ]

        config_dict: Dict[str, Any] = {}

        # 从 Apollo 读取配置
        for key in config_keys:
            value = apollo_client.get_value(key, namespace=namespace)
            if value is not None:
                config_dict[key] = value
                logger.debug(f"从 Apollo 加载配置: {key} = {value}")

        # 创建配置实例
        try:
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"创建配置实例失败: {e}")
            raise

    @classmethod
    def from_local_env(cls) -> "Settings":
        """从本地环境变量加载配置（本地开发模式）.

        Returns:
            Settings: 配置实例
        """
        logger.info("从本地环境变量加载配置（本地开发模式）")
        return cls()  # type: ignore


class ConfigManager:
    """配置管理器.

    负责初始化和管理应用配置，包括：
    1. 加载本地 Apollo 连接配置
    2. 连接 Apollo 配置中心
    3. 加载业务配置
    4. 提供配置刷新功能
    """

    def __init__(self) -> None:
        """初始化配置管理器."""
        self.local_settings: LocalSettings = LocalSettings()
        self.apollo_client: Optional[ApolloClient] = None
        self.settings: Optional[Settings] = None
        self._initialized: bool = False

    def initialize(self) -> Settings:
        """初始化配置.

        Returns:
            Settings: 应用配置实例

        Raises:
            RuntimeError: 当配置初始化失败时
        """
        if self._initialized:
            logger.warning("配置管理器已经初始化，跳过重复初始化")
            return self.settings  # type: ignore

        logger.info(
            f"初始化配置管理器，环境: {self.local_settings.env}, 本地开发模式: {self.local_settings.local_dev_mode}"
        )

        try:
            if self.local_settings.local_dev_mode:
                # 本地开发模式：从环境变量加载
                logger.info("使用本地开发模式，从环境变量加载配置")
                self.settings = Settings.from_local_env()
            else:
                # 生产模式：从 Apollo 加载
                logger.info("使用 Apollo 配置中心")
                self._init_apollo_client()
                self.settings = Settings.from_apollo(
                    apollo_client=self.apollo_client,  # type: ignore
                    namespace=self.local_settings.apollo_namespaces[0],
                )

            self._initialized = True
            logger.info("配置管理器初始化成功")
            return self.settings

        except Exception as e:
            logger.error(f"配置管理器初始化失败: {e}")
            raise RuntimeError(f"配置初始化失败: {e}") from e

    def _init_apollo_client(self) -> None:
        """初始化 Apollo 客户端.

        Raises:
            RuntimeError: 当 Apollo 客户端初始化失败时
        """
        try:
            # 创建 Apollo 配置
            apollo_settings = ApolloSettingsConfig(
                meta_server_address=self.local_settings.apollo_meta_server_address,
                app_id=self.local_settings.apollo_app_id,
                cluster=self.local_settings.apollo_cluster,
                env=self.local_settings.apollo_env,
                namespaces=self.local_settings.apollo_namespaces,
                app_secret=self.local_settings.apollo_app_secret,
                using_app_secret=self.local_settings.apollo_using_app_secret,
            )

            # 创建 Apollo 客户端
            self.apollo_client = ApolloClient(settings=apollo_settings)

            logger.info(
                f"Apollo 客户端初始化成功，服务器: {self.local_settings.apollo_meta_server_address}, "
                f"应用: {self.local_settings.apollo_app_id}"
            )

        except Exception as e:
            logger.error(f"Apollo 客户端初始化失败: {e}")
            raise RuntimeError(f"Apollo 客户端初始化失败: {e}") from e

    def reload(self) -> Settings:
        """重新加载配置.

        用于动态刷新配置（例如 Apollo 配置变更后）。

        Returns:
            Settings: 更新后的配置实例
        """
        logger.info("重新加载配置")

        if self.local_settings.local_dev_mode:
            self.settings = Settings.from_local_env()
        elif self.apollo_client:
            # 从 Apollo 重新拉取配置
            self.apollo_client.fetch_configuration()
            self.settings = Settings.from_apollo(
                apollo_client=self.apollo_client, namespace=self.local_settings.apollo_namespaces[0]
            )
        else:
            raise RuntimeError("Apollo 客户端未初始化，无法重新加载配置")

        logger.info("配置重新加载成功")
        return self.settings

    def get_settings(self) -> Settings:
        """获取配置实例.

        如果配置尚未初始化，则自动初始化。

        Returns:
            Settings: 应用配置实例
        """
        if not self._initialized:
            return self.initialize()
        return self.settings  # type: ignore


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


@lru_cache
def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例（单例）.

    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_settings() -> Settings:
    """获取应用配置（便捷函数）.

    这是推荐的获取配置的方式。

    Returns:
        Settings: 应用配置实例

    Example:
        ```python
        from pikee.infrastructure.config.settings import get_settings

        settings = get_settings()
        print(settings.openai_api_key)
        ```
    """
    manager = get_config_manager()
    return manager.get_settings()


def reload_settings() -> Settings:
    """重新加载配置（便捷函数）.

    用于动态刷新配置，例如在管理接口中调用。

    Returns:
        Settings: 更新后的配置实例

    Example:
        ```python
        from pikee.infrastructure.config.settings import reload_settings


        @app.post("/admin/reload-config")
        async def admin_reload():
            settings = reload_settings()
            return {"message": "配置已重新加载"}
        ```
    """
    manager = get_config_manager()
    return manager.reload()
