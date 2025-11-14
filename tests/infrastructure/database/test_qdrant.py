"""Qdrant 客户端测试."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.qdrant import QdrantClient


class TestQdrantClient:
    """测试 Qdrant 客户端."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建 mock settings."""
        return Settings(
            openai_api_key="test-key",
            neo4j_password="test-pass",
            qdrant_url="http://localhost:6333",
            qdrant_collection_chunks="test_chunks",
            qdrant_collection_atoms="test_atoms",
        )

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_initialization(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试初始化."""
        client = QdrantClient(mock_settings)

        assert client.url == "http://localhost:6333"
        mock_qdrant_client.assert_called_once_with(url="http://localhost:6333")

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_create_collection(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试创建 Collection."""
        mock_instance = mock_qdrant_client.return_value
        mock_instance.get_collections.return_value.collections = []

        client = QdrantClient(mock_settings)
        success = client.create_collection("test_collection", vector_size=384)

        assert success
        assert mock_instance.create_collection.called

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_collection_exists(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试检查 Collection 是否存在."""
        mock_instance = mock_qdrant_client.return_value

        # Mock collection 列表
        mock_collection = MagicMock()
        mock_collection.name = "existing_collection"
        mock_instance.get_collections.return_value.collections = [mock_collection]

        client = QdrantClient(mock_settings)
        
        assert client.collection_exists("existing_collection")
        assert not client.collection_exists("non_existing_collection")

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_insert_vectors(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试插入向量."""
        mock_instance = mock_qdrant_client.return_value

        client = QdrantClient(mock_settings)
        
        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"text": "hello"}, {"text": "world"}]
        ids = ["id1", "id2"]

        success = client.insert_vectors(
            collection_name="test_collection",
            vectors=vectors,
            payloads=payloads,
            ids=ids,
        )

        assert success
        assert mock_instance.upsert.called

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_insert_vectors_mismatch(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试向量和 payload 数量不匹配."""
        client = QdrantClient(mock_settings)
        
        vectors = [[0.1] * 384]
        payloads = [{"text": "hello"}, {"text": "world"}]

        with pytest.raises(ValueError, match="向量数量.*与 payload 数量.*不匹配"):
            client.insert_vectors(
                collection_name="test_collection",
                vectors=vectors,
                payloads=payloads,
            )

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_search_vectors(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试检索向量."""
        mock_instance = mock_qdrant_client.return_value

        # Mock 搜索结果
        mock_result = MagicMock()
        mock_result.id = "id1"
        mock_result.score = 0.95
        mock_result.payload = {"text": "hello"}
        mock_instance.search.return_value = [mock_result]

        client = QdrantClient(mock_settings)
        
        query_vector = [0.1] * 384
        results = client.search_vectors(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=10,
        )

        assert len(results) == 1
        assert results[0][0] == "id1"  # ID
        assert results[0][1] == 0.95   # Score
        assert results[0][2] == {"text": "hello"}  # Payload

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_delete_collection(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试删除 Collection."""
        mock_instance = mock_qdrant_client.return_value

        client = QdrantClient(mock_settings)
        success = client.delete_collection("test_collection")

        assert success
        mock_instance.delete_collection.assert_called_once_with(collection_name="test_collection")

    @patch("pikee.infrastructure.database.qdrant.QdrantClientBase")
    def test_count_vectors(self, mock_qdrant_client: MagicMock, mock_settings: Settings) -> None:
        """测试统计向量数量."""
        mock_instance = mock_qdrant_client.return_value

        # Mock collection 信息
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_instance.get_collection.return_value = mock_info

        client = QdrantClient(mock_settings)
        count = client.count_vectors("test_collection")

        assert count == 100

