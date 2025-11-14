"""图谱构建器测试."""

from unittest.mock import MagicMock, patch

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.neo4j import Neo4jClient
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.models.document import Document
from pikee.pipeline.services.graph.builder import GraphBuilder, build_knowledge_graph


class TestGraphBuilder:
    """图谱构建器测试."""

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
    def mock_neo4j_client(self) -> MagicMock:
        """Mock Neo4j 客户端."""
        return MagicMock(spec=Neo4jClient)

    @pytest.fixture
    def builder(self, mock_settings: Settings, mock_neo4j_client: MagicMock) -> GraphBuilder:
        """创建图谱构建器实例."""
        return GraphBuilder(mock_settings, mock_neo4j_client)

    @pytest.fixture
    def sample_document(self) -> Document:
        """示例文档."""
        return Document(
            id="doc1",
            title="测试文档",
            content="这是测试文档内容",
            file_path="/path/to/doc.txt",
            file_type="text",
            metadata={"author": "test"},
        )

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """示例 Chunks."""
        return [
            Chunk(id="chunk1", document_id="doc1", content="第一段内容", summary="第一段摘要", index=0),
            Chunk(id="chunk2", document_id="doc1", content="第二段内容", summary="第二段摘要", index=1),
        ]

    @pytest.fixture
    def sample_atoms(self) -> list[Atom]:
        """示例 Atoms."""
        return [
            Atom(id="atom1", chunk_id="chunk1", question="问题1?", answer="答案1"),
            Atom(id="atom2", chunk_id="chunk1", question="问题2?", answer="答案2"),
            Atom(id="atom3", chunk_id="chunk2", question="问题3?", answer="答案3"),
        ]

    def test_init(self, builder: GraphBuilder, mock_settings: Settings, mock_neo4j_client: MagicMock) -> None:
        """测试初始化."""
        assert builder.settings == mock_settings
        assert builder.neo4j_client == mock_neo4j_client

    def test_build_graph_success(
        self,
        builder: GraphBuilder,
        mock_neo4j_client: MagicMock,
        sample_document: Document,
        sample_chunks: list[Chunk],
        sample_atoms: list[Atom],
    ) -> None:
        """测试成功构建图谱."""
        # 配置 mock 返回值
        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = len(sample_chunks)
        mock_neo4j_client.create_atoms_batch.return_value = len(sample_atoms)

        # 执行构建
        success = builder.build_graph(sample_document, sample_chunks, sample_atoms)

        # 验证结果
        assert success is True

        # 验证调用
        mock_neo4j_client.create_document.assert_called_once_with(
            document_id="doc1",
            title="测试文档",
            file_path="/path/to/doc.txt",
            file_type="text",
            metadata={"author": "test"},
        )

        mock_neo4j_client.create_chunks_batch.assert_called_once()
        chunks_data = mock_neo4j_client.create_chunks_batch.call_args[0][0]
        assert len(chunks_data) == 2
        assert chunks_data[0]["chunk_id"] == "chunk1"
        assert chunks_data[1]["chunk_id"] == "chunk2"

        mock_neo4j_client.create_atoms_batch.assert_called_once()
        atoms_data = mock_neo4j_client.create_atoms_batch.call_args[0][0]
        assert len(atoms_data) == 3
        assert atoms_data[0]["atom_id"] == "atom1"

    def test_build_graph_document_creation_failed(
        self,
        builder: GraphBuilder,
        mock_neo4j_client: MagicMock,
        sample_document: Document,
        sample_chunks: list[Chunk],
        sample_atoms: list[Atom],
    ) -> None:
        """测试文档创建失败."""
        mock_neo4j_client.create_document.return_value = False

        success = builder.build_graph(sample_document, sample_chunks, sample_atoms)

        assert success is False
        # 确保没有继续创建 chunks 和 atoms
        mock_neo4j_client.create_chunks_batch.assert_not_called()
        mock_neo4j_client.create_atoms_batch.assert_not_called()

    def test_build_graph_chunks_partial_success(
        self,
        builder: GraphBuilder,
        mock_neo4j_client: MagicMock,
        sample_document: Document,
        sample_chunks: list[Chunk],
        sample_atoms: list[Atom],
    ) -> None:
        """测试 Chunks 部分创建成功."""
        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = 1  # 只创建成功 1 个
        mock_neo4j_client.create_atoms_batch.return_value = len(sample_atoms)

        success = builder.build_graph(sample_document, sample_chunks, sample_atoms)

        # 虽然部分失败，但整体仍返回 True（记录警告日志）
        assert success is True

    def test_build_graph_exception(
        self,
        builder: GraphBuilder,
        mock_neo4j_client: MagicMock,
        sample_document: Document,
        sample_chunks: list[Chunk],
        sample_atoms: list[Atom],
    ) -> None:
        """测试构建过程中异常."""
        mock_neo4j_client.create_document.side_effect = Exception("Database error")

        success = builder.build_graph(sample_document, sample_chunks, sample_atoms)

        assert success is False

    def test_build_document_graph_success(
        self, builder: GraphBuilder, mock_neo4j_client: MagicMock, sample_document: Document, sample_chunks: list[Chunk]
    ) -> None:
        """测试仅构建文档和 Chunks."""
        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = len(sample_chunks)

        success = builder.build_document_graph(sample_document, sample_chunks)

        assert success is True

        mock_neo4j_client.create_document.assert_called_once()
        mock_neo4j_client.create_chunks_batch.assert_called_once()

    def test_build_document_graph_no_chunks_created(
        self, builder: GraphBuilder, mock_neo4j_client: MagicMock, sample_document: Document, sample_chunks: list[Chunk]
    ) -> None:
        """测试没有 Chunk 创建成功."""
        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = 0

        success = builder.build_document_graph(sample_document, sample_chunks)

        # 如果没有创建成功任何 Chunk，返回 False
        assert success is False

    def test_build_document_graph_exception(
        self, builder: GraphBuilder, mock_neo4j_client: MagicMock, sample_document: Document, sample_chunks: list[Chunk]
    ) -> None:
        """测试构建文档图谱时异常."""
        mock_neo4j_client.create_document.side_effect = Exception("Database error")

        success = builder.build_document_graph(sample_document, sample_chunks)

        assert success is False


class TestBuildKnowledgeGraphFunction:
    """测试便捷函数."""

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
    def sample_document(self) -> Document:
        """示例文档."""
        return Document(id="doc1", title="测试文档", content="测试内容")

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """示例 Chunks."""
        return [Chunk(id="chunk1", document_id="doc1", content="内容1", summary="摘要1", index=0)]

    @pytest.fixture
    def sample_atoms(self) -> list[Atom]:
        """示例 Atoms."""
        return [Atom(id="atom1", chunk_id="chunk1", question="问题1?", answer="答案1")]

    def test_build_knowledge_graph_integration(
        self, mock_settings: Settings, sample_document: Document, sample_chunks: list[Chunk], sample_atoms: list[Atom]
    ) -> None:
        """测试便捷函数（集成测试风格，但使用 mock）."""
        with patch("pikee.infrastructure.database.neo4j.get_neo4j_client") as mock_get_client:
            # 创建 mock 客户端
            mock_client = MagicMock(spec=Neo4jClient)
            mock_get_client.return_value = mock_client

            # 配置返回值
            mock_client.create_document.return_value = True
            mock_client.create_chunks_batch.return_value = 1
            mock_client.create_atoms_batch.return_value = 1

            # 调用便捷函数
            success = build_knowledge_graph(mock_settings, sample_document, sample_chunks, sample_atoms)

            # 验证结果
            assert success is True

            # 验证客户端被正确创建和关闭
            mock_get_client.assert_called_once_with(mock_settings)
            mock_client.close.assert_called_once()


class TestGraphBuilderEdgeCases:
    """图谱构建器边界情况测试."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Mock 配置."""
        return Settings(
            neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password", openai_api_key="test-key"
        )

    @pytest.fixture
    def mock_neo4j_client(self) -> MagicMock:
        """Mock Neo4j 客户端."""
        return MagicMock(spec=Neo4jClient)

    @pytest.fixture
    def builder(self, mock_settings: Settings, mock_neo4j_client: MagicMock) -> GraphBuilder:
        """创建图谱构建器实例."""
        return GraphBuilder(mock_settings, mock_neo4j_client)

    def test_build_graph_with_empty_chunks(self, builder: GraphBuilder, mock_neo4j_client: MagicMock) -> None:
        """测试空 Chunks 列表."""
        document = Document(id="doc1", title="测试", content="内容")

        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = 0
        mock_neo4j_client.create_atoms_batch.return_value = 0

        success = builder.build_graph(document, [], [])

        assert success is True
        mock_neo4j_client.create_chunks_batch.assert_called_once_with([])

    def test_build_graph_with_empty_atoms(self, builder: GraphBuilder, mock_neo4j_client: MagicMock) -> None:
        """测试空 Atoms 列表."""
        document = Document(id="doc1", title="测试", content="内容")
        chunks = [Chunk(id="chunk1", document_id="doc1", content="内容1", summary="", index=0)]

        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = 1
        mock_neo4j_client.create_atoms_batch.return_value = 0

        success = builder.build_graph(document, chunks, [])

        assert success is True

    def test_build_graph_with_minimal_document(self, builder: GraphBuilder, mock_neo4j_client: MagicMock) -> None:
        """测试最小化文档（必填字段）."""
        document = Document(id="minimal", title="", content="")

        mock_neo4j_client.create_document.return_value = True
        mock_neo4j_client.create_chunks_batch.return_value = 0
        mock_neo4j_client.create_atoms_batch.return_value = 0

        success = builder.build_graph(document, [], [])

        assert success is True
        mock_neo4j_client.create_document.assert_called_once()
