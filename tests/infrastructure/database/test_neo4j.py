"""Neo4j 客户端测试."""

from unittest.mock import MagicMock, patch

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.neo4j import Neo4jClient, get_neo4j_client


class TestNeo4jClient:
    """Neo4j 客户端测试."""

    @pytest.fixture
    def mock_driver(self) -> MagicMock:
        """Mock Neo4j 驱动."""
        with patch("pikee.infrastructure.database.neo4j.GraphDatabase.driver") as mock:
            yield mock

    @pytest.fixture
    def mock_session(self, mock_driver: MagicMock) -> MagicMock:
        """Mock Neo4j Session."""
        session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = session
        return session

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Mock 配置."""
        return Settings(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            openai_api_key="test-key",
        )

    @pytest.fixture
    def client(self, mock_settings: Settings, mock_driver: MagicMock) -> Neo4jClient:
        """创建客户端实例."""
        return Neo4jClient(mock_settings)

    def test_init(self, client: Neo4jClient, mock_settings: Settings) -> None:
        """测试初始化."""
        assert client.settings == mock_settings
        assert client.uri == mock_settings.neo4j_uri
        assert client.user == mock_settings.neo4j_user
        assert client.password == mock_settings.neo4j_password
        assert client.database == mock_settings.neo4j_database

    def test_close(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        """测试关闭连接."""
        client.close()
        client._driver.close.assert_called_once()

    def test_execute_query(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试执行查询."""
        # 模拟查询结果
        mock_record = MagicMock()
        mock_record.data.return_value = {"id": 1, "name": "test"}
        mock_session.run.return_value = [mock_record]

        query = "MATCH (n) RETURN n"
        result = client._execute_query(query)

        assert len(result) == 1
        assert result[0] == {"id": 1, "name": "test"}
        mock_session.run.assert_called_once_with(query, {})

    def test_execute_query_with_parameters(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试带参数的查询."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"id": 1}
        mock_session.run.return_value = [mock_record]

        query = "MATCH (n {id: $id}) RETURN n"
        params = {"id": 1}
        result = client._execute_query(query, params)

        assert len(result) == 1
        mock_session.run.assert_called_once_with(query, params)

    def test_create_document(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试创建文档节点."""
        mock_session.run.return_value = []

        success = client.create_document(
            document_id="doc1", title="测试文档", file_path="/path/to/doc", file_type="pdf", metadata={"author": "test"}
        )

        assert success is True
        mock_session.run.assert_called_once()

    def test_get_document(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取文档节点."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"d": {"document_id": "doc1", "title": "测试文档"}}
        mock_session.run.return_value = [mock_record]

        result = client.get_document("doc1")

        assert result is not None
        assert result["document_id"] == "doc1"
        assert result["title"] == "测试文档"

    def test_get_document_not_found(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取不存在的文档."""
        mock_session.run.return_value = []
        result = client.get_document("nonexistent")
        assert result is None

    def test_delete_document(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试删除文档."""
        mock_session.run.return_value = []
        success = client.delete_document("doc1")
        assert success is True

    def test_create_chunk(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试创建 Chunk 节点."""
        mock_session.run.return_value = []

        success = client.create_chunk(
            chunk_id="chunk1",
            document_id="doc1",
            content="测试内容",
            summary="摘要",
            index=0,
            metadata={"key": "value"},
        )

        assert success is True
        mock_session.run.assert_called_once()

    def test_get_chunk(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取 Chunk 节点."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"c": {"chunk_id": "chunk1", "content": "测试内容"}}
        mock_session.run.return_value = [mock_record]

        result = client.get_chunk("chunk1")

        assert result is not None
        assert result["chunk_id"] == "chunk1"

    def test_create_atom(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试创建 Atom 节点."""
        mock_session.run.return_value = []

        success = client.create_atom(
            atom_id="atom1", chunk_id="chunk1", question="测试问题?", answer="测试答案", metadata={"key": "value"}
        )

        assert success is True
        mock_session.run.assert_called_once()

    def test_get_atom(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取 Atom 节点."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"a": {"atom_id": "atom1", "question": "测试问题?"}}
        mock_session.run.return_value = [mock_record]

        result = client.get_atom("atom1")

        assert result is not None
        assert result["atom_id"] == "atom1"

    def test_create_chunks_batch(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试批量创建 Chunks."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"created": 3}
        mock_session.run.return_value = [mock_record]

        chunks = [
            {
                "chunk_id": f"chunk{i}",
                "document_id": "doc1",
                "content": f"内容{i}",
                "summary": f"摘要{i}",
                "index": i,
                "metadata": {},
            }
            for i in range(3)
        ]

        count = client.create_chunks_batch(chunks, batch_size=10)
        assert count == 3

    def test_create_atoms_batch(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试批量创建 Atoms."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"created": 5}
        mock_session.run.return_value = [mock_record]

        atoms = [
            {"atom_id": f"atom{i}", "chunk_id": "chunk1", "question": f"问题{i}?", "answer": f"答案{i}", "metadata": {}}
            for i in range(5)
        ]

        count = client.create_atoms_batch(atoms, batch_size=10)
        assert count == 5

    def test_get_chunks_by_document(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取文档的所有 Chunks."""
        mock_records = [MagicMock(data=lambda: {"c": {"chunk_id": f"chunk{i}"}}) for i in range(3)]

        # 需要让每个 record.data() 返回不同的值
        for i, record in enumerate(mock_records):
            record.data.return_value = {"c": {"chunk_id": f"chunk{i}"}}

        mock_session.run.return_value = mock_records

        chunks = client.get_chunks_by_document("doc1")
        assert len(chunks) == 3

    def test_get_atoms_by_chunk(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取 Chunk 的所有 Atoms."""
        mock_records = [MagicMock(data=lambda: {"a": {"atom_id": f"atom{i}"}}) for i in range(2)]

        for i, record in enumerate(mock_records):
            record.data.return_value = {"a": {"atom_id": f"atom{i}"}}

        mock_session.run.return_value = mock_records

        atoms = client.get_atoms_by_chunk("chunk1")
        assert len(atoms) == 2

    def test_get_graph_statistics(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取图谱统计信息."""
        mock_record = MagicMock()
        mock_record.data.return_value = {"documents": 10, "chunks": 100, "atoms": 500}
        mock_session.run.return_value = [mock_record]

        stats = client.get_graph_statistics()

        assert stats["documents"] == 10
        assert stats["chunks"] == 100
        assert stats["atoms"] == 500

    def test_get_neo4j_client(self, mock_settings: Settings, mock_driver: MagicMock) -> None:
        """测试便捷函数."""
        client = get_neo4j_client(mock_settings)
        assert isinstance(client, Neo4jClient)
        client.close()


class TestNeo4jClientErrorHandling:
    """Neo4j 客户端错误处理测试."""

    @pytest.fixture
    def mock_driver(self) -> MagicMock:
        """Mock Neo4j 驱动."""
        with patch("pikee.infrastructure.database.neo4j.GraphDatabase.driver") as mock:
            yield mock

    @pytest.fixture
    def mock_session(self, mock_driver: MagicMock) -> MagicMock:
        """Mock Neo4j Session."""
        session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = session
        return session

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Mock 配置."""
        return Settings(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            openai_api_key="test-key",
        )

    @pytest.fixture
    def client(self, mock_settings: Settings, mock_driver: MagicMock) -> Neo4jClient:
        """创建客户端实例."""
        return Neo4jClient(mock_settings)

    def test_create_document_failure(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试创建文档失败."""
        mock_session.run.side_effect = Exception("Database error")

        success = client.create_document(document_id="doc1", title="测试")
        assert success is False

    def test_get_document_error(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试获取文档时出错."""
        mock_session.run.side_effect = Exception("Database error")

        result = client.get_document("doc1")
        assert result is None

    def test_create_chunks_batch_partial_failure(self, client: Neo4jClient, mock_session: MagicMock) -> None:
        """测试批量创建 Chunks 部分失败."""
        # 第一批成功，第二批失败
        mock_record = MagicMock()
        mock_record.data.return_value = {"created": 50}
        mock_session.run.side_effect = [
            [mock_record],  # 第一批成功
            Exception("Database error"),  # 第二批失败
        ]

        chunks = [
            {
                "chunk_id": f"chunk{i}",
                "document_id": "doc1",
                "content": f"内容{i}",
                "summary": "",
                "index": i,
                "metadata": {},
            }
            for i in range(150)
        ]

        count = client.create_chunks_batch(chunks, batch_size=50)
        # 只有第一批成功
        assert count == 50
