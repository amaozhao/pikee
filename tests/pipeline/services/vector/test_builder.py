"""向量数据库构建服务测试."""

from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.qdrant import QdrantClient
from pikee.infrastructure.llm.embedder import BaseEmbedder
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.services.vector.builder import VectorDatabaseBuilder


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


class TestVectorDatabaseBuilder:
    """测试向量数据库构建器."""

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
        client = MagicMock(spec=QdrantClient)
        client.collection_exists.return_value = False
        client.create_collection.return_value = True
        client.insert_vectors.return_value = True
        return client

    @pytest.fixture
    def mock_embedder(self) -> MockEmbedder:
        """创建 mock embedder."""
        return MockEmbedder()

    @pytest.fixture
    def sample_chunks(self) -> List[Chunk]:
        """创建测试用 Chunks."""
        return [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="This is chunk 1",
                summary="Summary 1",
                index=0,
                char_count=15,
                metadata={"title": "Test Doc", "source": "test.txt"},
                created_at=datetime.now(),
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="This is chunk 2",
                summary="Summary 2",
                index=1,
                char_count=15,
                metadata={"title": "Test Doc", "source": "test.txt"},
                created_at=datetime.now(),
            ),
        ]

    @pytest.fixture
    def sample_atoms(self) -> List[Atom]:
        """创建测试用 Atoms."""
        return [
            Atom(
                id="atom1",
                chunk_id="chunk1",
                question="What is Python?",
                answer="Python is a programming language",
                metadata={"document_id": "doc1", "title": "Test Doc"},
                created_at=datetime.now(),
            ),
            Atom(
                id="atom2",
                chunk_id="chunk1",
                question="What is GIL?",
                answer="GIL is Global Interpreter Lock",
                metadata={"document_id": "doc1", "title": "Test Doc"},
                created_at=datetime.now(),
            ),
        ]

    def test_initialization(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试初始化."""
        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        assert builder.batch_size == 50
        assert builder.chunk_collection == "test_chunks"
        assert builder.atom_collection == "test_atoms"

    def test_build_chunk_store(
        self,
        mock_settings: Settings,
        mock_qdrant_client: MagicMock,
        mock_embedder: MockEmbedder,
        sample_chunks: List[Chunk],
    ) -> None:
        """测试构建 Chunk 向量存储."""
        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        success = builder.build_chunk_store(sample_chunks, show_progress=False)

        assert success
        # 验证创建 Collection
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_chunks", vector_size=384, distance="Cosine"
        )
        # 验证插入向量
        assert mock_qdrant_client.insert_vectors.called

    def test_build_chunk_store_empty(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试空 Chunks 列表."""
        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        success = builder.build_chunk_store([], show_progress=False)

        assert not success
        assert not mock_qdrant_client.insert_vectors.called

    def test_build_atom_store(
        self,
        mock_settings: Settings,
        mock_qdrant_client: MagicMock,
        mock_embedder: MockEmbedder,
        sample_atoms: List[Atom],
    ) -> None:
        """测试构建 Atom 向量存储."""
        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        success = builder.build_atom_store(sample_atoms, show_progress=False)

        assert success
        # 验证创建 Collection
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_atoms", vector_size=384, distance="Cosine"
        )
        # 验证插入向量
        assert mock_qdrant_client.insert_vectors.called

    def test_build_all(
        self,
        mock_settings: Settings,
        mock_qdrant_client: MagicMock,
        mock_embedder: MockEmbedder,
        sample_chunks: List[Chunk],
        sample_atoms: List[Atom],
    ) -> None:
        """测试一次性构建."""
        # Mock get_collection_info
        mock_qdrant_client.get_collection_info.return_value = {"vectors_count": 2, "points_count": 2}

        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        success = builder.build_all(sample_chunks, sample_atoms, show_progress=False)

        assert success
        # 验证两个 Collection 都被创建
        assert mock_qdrant_client.create_collection.call_count == 2

    def test_recreate_collection(
        self,
        mock_settings: Settings,
        mock_qdrant_client: MagicMock,
        mock_embedder: MockEmbedder,
        sample_chunks: List[Chunk],
    ) -> None:
        """测试重新创建 Collection."""
        # 模拟 Collection 状态变化：
        # 第1次调用 collection_exists: 返回 True（存在）
        # 删除后，第2次调用 collection_exists: 返回 False（已删除）
        mock_qdrant_client.collection_exists.side_effect = [True, False]
        mock_qdrant_client.delete_collection.return_value = True

        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder
        )

        success = builder.build_chunk_store(sample_chunks, recreate_collection=True, show_progress=False)

        assert success
        # 验证删除和创建
        mock_qdrant_client.delete_collection.assert_called_once_with("test_chunks")
        mock_qdrant_client.create_collection.assert_called()

    def test_custom_batch_size(
        self, mock_settings: Settings, mock_qdrant_client: MagicMock, mock_embedder: MockEmbedder
    ) -> None:
        """测试自定义批处理大小."""
        builder = VectorDatabaseBuilder(
            settings=mock_settings, qdrant_client=mock_qdrant_client, embedder=mock_embedder, batch_size=10
        )

        assert builder.batch_size == 10
