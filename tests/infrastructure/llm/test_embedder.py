"""Embedding 服务测试."""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.llm.embedder import (
    BaseEmbedder,
    EmbedderFactory,
    FastEmbedEmbedder,
    OpenAIEmbedder,
)


class TestOpenAIEmbedder:
    """测试 OpenAI Embedder."""

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_initialization(self, mock_openai_embeddings: MagicMock) -> None:
        """测试初始化."""
        api_key = "test-key"
        embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-ada-002")

        assert embedder.api_key == api_key
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.batch_size == 100
        mock_openai_embeddings.assert_called_once()

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_embed_text(self, mock_openai_embeddings: MagicMock) -> None:
        """测试单文本嵌入."""
        # Mock 返回值
        mock_instance = mock_openai_embeddings.return_value
        mock_vector = [0.1] * 1536
        mock_instance.embed_query.return_value = mock_vector

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed_text("test text")

        assert result == mock_vector
        assert len(result) == 1536
        mock_instance.embed_query.assert_called_once_with("test text")

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_embed_texts(self, mock_openai_embeddings: MagicMock) -> None:
        """测试批量嵌入."""
        mock_instance = mock_openai_embeddings.return_value
        mock_vectors = [[0.1] * 1536, [0.2] * 1536]
        mock_instance.embed_documents.return_value = mock_vectors

        embedder = OpenAIEmbedder(api_key="test-key")
        texts = ["text1", "text2"]
        result = embedder.embed_texts(texts)

        assert result == mock_vectors
        assert len(result) == 2
        mock_instance.embed_documents.assert_called_once_with(texts)

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_dimension(self, mock_openai_embeddings: MagicMock) -> None:
        """测试向量维度."""
        embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder.dimension == 1536


class TestFastEmbedEmbedder:
    """测试 FastEmbed Embedder."""

    @patch("fastembed.TextEmbedding")
    def test_initialization(self, mock_text_embedding: MagicMock) -> None:
        """测试初始化."""
        embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")

        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.batch_size == 256
        mock_text_embedding.assert_called_once()

    @patch("fastembed.TextEmbedding")
    def test_embed_text(self, mock_text_embedding: MagicMock) -> None:
        """测试单文本嵌入."""
        # Mock 返回值（FastEmbed 返回 numpy array）
        import numpy as np

        mock_instance = mock_text_embedding.return_value
        mock_vector = np.array([0.1] * 384)
        mock_instance.embed.return_value = iter([mock_vector])

        embedder = FastEmbedEmbedder()
        result = embedder.embed_text("test text")

        assert len(result) == 384
        mock_instance.embed.assert_called_once_with(["test text"])

    @patch("fastembed.TextEmbedding")
    def test_embed_texts(self, mock_text_embedding: MagicMock) -> None:
        """测试批量嵌入."""
        import numpy as np

        mock_instance = mock_text_embedding.return_value
        mock_vectors = [np.array([0.1] * 384), np.array([0.2] * 384)]
        mock_instance.embed.return_value = iter(mock_vectors)

        embedder = FastEmbedEmbedder()
        texts = ["text1", "text2"]
        result = embedder.embed_texts(texts)

        assert len(result) == 2
        assert len(result[0]) == 384
        mock_instance.embed.assert_called_once_with(texts)

    @patch("fastembed.TextEmbedding")
    def test_dimension(self, mock_text_embedding: MagicMock) -> None:
        """测试向量维度."""
        embedder = FastEmbedEmbedder()
        assert embedder.dimension == 384


class TestEmbedderFactory:
    """测试 Embedder 工厂."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建 mock settings."""
        return Settings(
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            neo4j_password="test-pass",
        )

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_create_openai_embedder(self, mock_openai_embeddings: MagicMock, mock_settings: Settings) -> None:
        """测试创建 OpenAI Embedder."""
        embedder = EmbedderFactory.create_embedder(mock_settings, provider="openai")

        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.api_key == "test-key"

    @patch("fastembed.TextEmbedding")
    def test_create_fastembed_embedder(self, mock_text_embedding: MagicMock, mock_settings: Settings) -> None:
        """测试创建 FastEmbed Embedder."""
        embedder = EmbedderFactory.create_embedder(mock_settings, provider="fastembed")

        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder.batch_size == 256

    def test_invalid_provider(self, mock_settings: Settings) -> None:
        """测试无效的提供商."""
        with pytest.raises(ValueError, match="不支持的 Embedding 提供商"):
            EmbedderFactory.create_embedder(mock_settings, provider="invalid")

    @patch("pikee.infrastructure.llm.embedder.OpenAIEmbeddings")
    def test_custom_batch_size(self, mock_openai_embeddings: MagicMock, mock_settings: Settings) -> None:
        """测试自定义批处理大小."""
        embedder = EmbedderFactory.create_embedder(mock_settings, provider="openai", batch_size=50)

        assert embedder.batch_size == 50

