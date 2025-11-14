"""Embedding 服务.

统一的 Embedding 接口，支持多种 Embedding 提供商。
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from pikee.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Embedding 基类."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本.

        Args:
            text: 待嵌入的文本

        Returns:
            List[float]: 向量表示
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本.

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embedding 服务.

    使用 OpenAI 的 text-embedding-ada-002 模型。

    特点:
        - 维度: 1536
        - 质量高
        - 需要 API Key

    Examples:
        >>> embedder = OpenAIEmbedder(api_key="sk-...")
        >>> vector = embedder.embed_text("Hello world")
        >>> print(len(vector))  # 1536
    """

    def __init__(
        self, api_key: str, model: str = "text-embedding-ada-002", api_base: Optional[str] = None, batch_size: int = 100
    ) -> None:
        """初始化 OpenAI Embedder.

        Args:
            api_key: OpenAI API Key
            model: 模型名称
            api_base: API Base URL（可选）
            batch_size: 批处理大小
        """

        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.batch_size = batch_size

        # 初始化 LangChain OpenAI Embeddings
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=api_key, model=model, openai_api_base=api_base, chunk_size=batch_size
        )

        logger.info(f"OpenAIEmbedder 初始化完成: model={model}, batch_size={batch_size}")

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本.

        Args:
            text: 待嵌入的文本

        Returns:
            List[float]: 1536维向量
        """
        try:
            vector = self._embeddings.embed_query(text)
            return vector
        except Exception as e:
            logger.error(f"OpenAI embedding 失败: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本.

        自动分批处理以避免 API 限制。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []

        try:
            vectors = self._embeddings.embed_documents(texts)
            return vectors
        except Exception as e:
            logger.error(f"批量 OpenAI embedding 失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        """向量维度."""
        return 1536


class FastEmbedEmbedder(BaseEmbedder):
    """FastEmbed 本地 Embedding 服务.

    使用本地模型，无需 API Key。

    特点:
        - 维度: 384 (BAAI/bge-small-en-v1.5)
        - 快速、免费
        - 本地运行

    Examples:
        >>> embedder = FastEmbedEmbedder()
        >>> vector = embedder.embed_text("Hello world")
        >>> print(len(vector))  # 384
    """

    def __init__(
        self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 256, cache_dir: Optional[str] = None
    ) -> None:
        """初始化 FastEmbed Embedder.

        Args:
            model: 模型名称
            batch_size: 批处理大小
            cache_dir: 模型缓存目录
        """
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError:
            raise ImportError("需要安装 fastembed: pip install fastembed")

        self.model = model
        self.batch_size = batch_size
        self._dim = 384  # BAAI/bge-small-en-v1.5 的维度

        # 初始化 FastEmbed
        self._embeddings = TextEmbedding(model_name=model, cache_dir=cache_dir)

        logger.info(f"FastEmbedEmbedder 初始化完成: model={model}, batch_size={batch_size}")

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本.

        Args:
            text: 待嵌入的文本

        Returns:
            List[float]: 384维向量
        """
        try:
            # FastEmbed 返回生成器，需要转换
            vectors = list(self._embeddings.embed([text]))
            return vectors[0].tolist()
        except Exception as e:
            logger.error(f"FastEmbed embedding 失败: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本.

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []

        try:
            # FastEmbed 返回生成器
            vectors = list(self._embeddings.embed(texts))
            return [vec.tolist() for vec in vectors]
        except Exception as e:
            logger.error(f"批量 FastEmbed embedding 失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        """向量维度."""
        return self._dim


class EmbedderFactory:
    """Embedder 工厂类.

    根据配置创建对应的 Embedder 实例。

    Examples:
        >>> settings = get_settings()
        >>> embedder = EmbedderFactory.create_embedder(settings, provider="openai")
        >>> vector = embedder.embed_text("Hello")
    """

    @staticmethod
    def create_embedder(settings: Settings, provider: str = "openai", batch_size: Optional[int] = None) -> BaseEmbedder:
        """创建 Embedder 实例.

        Args:
            settings: 应用配置
            provider: 提供商 ("openai" 或 "fastembed")
            batch_size: 批处理大小（可选，默认根据内存限制自动计算）

        Returns:
            BaseEmbedder: Embedder 实例

        Raises:
            ValueError: 当 provider 不支持时
        """
        # 根据512M内存限制计算批处理大小
        if batch_size is None:
            if provider == "openai":
                # OpenAI: 每个请求约5KB，保守估计
                batch_size = 100
            else:
                # FastEmbed: 本地运行，可以更大
                batch_size = 256

        provider = provider.lower()

        if provider == "openai":
            return OpenAIEmbedder(
                api_key=settings.openai_api_key, api_base=settings.openai_api_base, batch_size=batch_size
            )
        elif provider == "fastembed":
            return FastEmbedEmbedder(batch_size=batch_size)
        else:
            raise ValueError(f"不支持的 Embedding 提供商: {provider}。支持: 'openai', 'fastembed'")


def get_embedder(settings: Settings, provider: str = "openai", batch_size: Optional[int] = None) -> BaseEmbedder:
    """获取 Embedder 实例（便捷函数）.

    Args:
        settings: 应用配置
        provider: 提供商 ("openai" 或 "fastembed")
        batch_size: 批处理大小

    Returns:
        BaseEmbedder: Embedder 实例

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> embedder = get_embedder(settings, provider="fastembed")
        >>> vector = embedder.embed_text("测试文本")
    """
    return EmbedderFactory.create_embedder(settings, provider, batch_size)
