"""gRPC 依赖注入.

提供 gRPC 服务所需的依赖项。
"""

import logging
from typing import Any

from pikee.infrastructure.config.settings import Settings, get_settings
from pikee.infrastructure.database.neo4j import Neo4jClient, get_neo4j_client
from pikee.infrastructure.database.qdrant import QdrantClient, get_qdrant_client
from pikee.infrastructure.llm.client import LLMClient
from pikee.infrastructure.llm.embedder import EmbedderFactory

logger = logging.getLogger(__name__)


class GrpcDependencies:
    """gRPC 依赖容器.

    管理 gRPC 服务所需的所有依赖项。

    Examples:
        >>> deps = GrpcDependencies()
        >>> settings = deps.get_settings()
        >>> qdrant = deps.get_qdrant_client()
    """

    def __init__(self) -> None:
        """初始化依赖容器."""
        self._settings: Settings | None = None
        self._qdrant_client: QdrantClient | None = None
        self._neo4j_client: Neo4jClient | None = None
        self._llm_client: LLMClient | None = None
        self._embedder: Any | None = None

        logger.info("GrpcDependencies 初始化完成")

    def get_settings(self) -> Settings:
        """获取配置.

        Returns:
            应用配置
        """
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def get_qdrant_client(self) -> QdrantClient:
        """获取 Qdrant 客户端.

        Returns:
            Qdrant 客户端
        """
        if self._qdrant_client is None:
            settings = self.get_settings()
            self._qdrant_client = get_qdrant_client(settings)
        return self._qdrant_client

    def get_neo4j_client(self) -> Neo4jClient:
        """获取 Neo4j 客户端.

        Returns:
            Neo4j 客户端
        """
        if self._neo4j_client is None:
            settings = self.get_settings()
            self._neo4j_client = get_neo4j_client(settings)
        return self._neo4j_client

    def get_llm_client(self) -> LLMClient:
        """获取 LLM 客户端.

        Returns:
            LLM 客户端
        """
        if self._llm_client is None:
            settings = self.get_settings()
            self._llm_client = LLMClient(settings)
        return self._llm_client

    def get_embedder(self) -> Any:
        """获取 Embedding 服务.

        Returns:
            Embedder 实例
        """
        if self._embedder is None:
            settings = self.get_settings()
            self._embedder = EmbedderFactory.create(
                settings.embedder_provider,
                settings,
            )
        return self._embedder

    def close(self) -> None:
        """关闭所有连接."""
        if self._qdrant_client:
            self._qdrant_client.close()
        if self._neo4j_client:
            self._neo4j_client.close()

        logger.info("所有连接已关闭")


# 全局依赖实例
_dependencies: GrpcDependencies | None = None


def get_dependencies() -> GrpcDependencies:
    """获取全局依赖容器.

    Returns:
        依赖容器实例

    Examples:
        >>> deps = get_dependencies()
        >>> settings = deps.get_settings()
    """
    global _dependencies
    if _dependencies is None:
        _dependencies = GrpcDependencies()
    return _dependencies

