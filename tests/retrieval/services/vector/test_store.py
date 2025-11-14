"""向量检索服务测试."""

from typing import List
from unittest.mock import MagicMock

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.qdrant import QdrantClient
from pikee.infrastructure.llm.embedder import BaseEmbedder
from pikee.retrieval.services.vector.store import VectorRetriever


class MockEmbedder(BaseEmbedder):
    """Mock Embedder for testing."""

    def embed_text(self, text: str) -> List[float]:
        """Mock embed_text."""
        return [0.1] * 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mock embed_texts."""
        return [[0.1] * 384] * len(texts)

    @property
    def dimension(self) -> int:
        """Mock dimension."""
        return 384


class TestVectorRetriever:
    """测试向量检索服务."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建 mock settings."""
        return Settings(
            openai_api_key="test-key",
            neo4j_password="test-pass",
            qdrant_collection_chunks="test_chunks",
            qdrant_collection_atoms="test_atoms",
        )

    @pytest.fixture
    def mock_qdrant_client(self) -> MagicMock:
        """创建 mock Qdrant client."""
        return MagicMock(spec=QdrantClient)

    @pytest.fixture
    def mock_embedder(self) -> MockEmbedder:
        """创建 mock embedder."""
        return MockEmbedder()

    def test_initialization(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试初始化."""
        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        assert retriever.chunk_collection == "test_chunks"
        assert retriever.atom_collection == "test_atoms"

    def test_search_chunks(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试检索 Chunks."""
        # Mock 搜索结果
        mock_qdrant_client.search_vectors.return_value = [
            (
                "chunk1",
                0.95,
                {
                    "chunk_id": "chunk1",
                    "document_id": "doc1",
                    "content": "Python is a programming language",
                    "summary": "About Python",
                    "title": "Python Intro",
                    "source": "test.txt",
                    "index": 0,
                },
            ),
            (
                "chunk2",
                0.85,
                {
                    "chunk_id": "chunk2",
                    "document_id": "doc1",
                    "content": "Python has GIL",
                    "summary": "About GIL",
                    "title": "Python Intro",
                    "source": "test.txt",
                    "index": 1,
                },
            ),
        ]

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        results = retriever.search_chunks("What is Python?", top_k=10)

        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk1"
        assert results[0]["score"] == 0.95
        assert results[1]["chunk_id"] == "chunk2"

    def test_search_atoms(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试检索 Atoms."""
        # Mock 搜索结果
        mock_qdrant_client.search_vectors.return_value = [
            (
                "atom1",
                0.92,
                {
                    "atom_id": "atom1",
                    "question": "What is Python?",
                    "answer": "Python is a programming language",
                    "source_chunk_id": "chunk1",
                    "document_id": "doc1",
                    "title": "Python Intro",
                },
            )
        ]

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        results = retriever.search_atoms("What is Python?", top_k=5)

        assert len(results) == 1
        assert results[0]["atom_id"] == "atom1"
        assert results[0]["question"] == "What is Python?"
        assert results[0]["score"] == 0.92

    def test_search_with_filter(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试带过滤条件的检索."""
        mock_qdrant_client.search_vectors.return_value = []

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        filter_by = {"document_id": "doc1"}
        retriever.search_chunks("test query", top_k=10, filter_by=filter_by)

        # 验证调用参数
        mock_qdrant_client.search_vectors.assert_called_once()
        call_kwargs = mock_qdrant_client.search_vectors.call_args[1]
        assert call_kwargs["filter_conditions"] == filter_by

    def test_search_with_score_threshold(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试带分数阈值的检索."""
        mock_qdrant_client.search_vectors.return_value = []

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        retriever.search_chunks("test query", top_k=10, score_threshold=0.8)

        # 验证调用参数
        call_kwargs = mock_qdrant_client.search_vectors.call_args[1]
        assert call_kwargs["score_threshold"] == 0.8

    def test_hybrid_search(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试混合检索."""

        # Mock 不同的返回值
        def search_side_effect(*args, **kwargs):
            collection_name = kwargs.get("collection_name")
            if collection_name == "test_chunks":
                return [("chunk1", 0.95, {"chunk_id": "chunk1", "content": "test"})]
            else:
                return [("atom1", 0.92, {"atom_id": "atom1", "question": "test?"})]

        mock_qdrant_client.search_vectors.side_effect = search_side_effect

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        chunks, atoms = retriever.hybrid_search("test query", top_k_chunks=5, top_k_atoms=3)

        assert len(chunks) == 1
        assert len(atoms) == 1
        assert chunks[0]["chunk_id"] == "chunk1"
        assert atoms[0]["atom_id"] == "atom1"

    def test_search_error_handling(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试错误处理."""
        # Mock 抛出异常
        mock_qdrant_client.search_vectors.side_effect = Exception("Search failed")

        retriever = VectorRetriever(settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder)

        # 应该返回空列表而不是抛出异常
        results = retriever.search_chunks("test query")
        assert results == []
