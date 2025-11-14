"""测试 Atom 提取服务."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.services.tagger import AtomExtractor, SimpleAtomExtractor


class TestSimpleAtomExtractor:
    """测试 SimpleAtomExtractor 类."""

    def test_simple_extractor_initialization(self) -> None:
        """测试 SimpleAtomExtractor 初始化."""
        extractor = SimpleAtomExtractor(max_atoms_per_chunk=5)

        assert extractor.max_atoms_per_chunk == 5

    def test_extract_atoms_basic(self) -> None:
        """测试基本 Atom 提取."""
        extractor = SimpleAtomExtractor(max_atoms_per_chunk=3)

        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="这是第一句。这是第二句。这是第三句。这是第四句。",
        )

        atoms = extractor.extract_atoms(chunk)

        assert len(atoms) <= 3
        assert all(atom.chunk_id == chunk.id for atom in atoms)
        assert all(atom.question for atom in atoms)

    def test_extract_atoms_empty_chunk(self) -> None:
        """测试空 Chunk."""
        extractor = SimpleAtomExtractor()

        chunk = Chunk(id="empty-chunk", content="")

        atoms = extractor.extract_atoms(chunk)

        assert len(atoms) == 0

    def test_extract_atoms_batch(self) -> None:
        """测试批量提取."""
        extractor = SimpleAtomExtractor(max_atoms_per_chunk=2)

        chunks = [Chunk(id=f"chunk-{i}", content=f"句子1。句子2。句子3。" * i) for i in range(1, 4)]

        all_atoms = extractor.extract_atoms_batch(chunks, show_progress=False)

        assert len(all_atoms) > 0
        # 验证每个 chunk 都有对应的 atoms
        chunk_ids = {atom.chunk_id for atom in all_atoms}
        assert len(chunk_ids) == len(chunks)


class TestAtomExtractor:
    """测试 AtomExtractor 类."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建模拟配置."""
        import os

        os.environ["LOCAL_DEV_MODE"] = "true"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["NEO4J_PASSWORD"] = "test-password"

        settings = Settings.from_local_env()
        return settings

    @pytest.fixture
    def mock_llm_client(self) -> Any:
        """创建模拟 LLM 客户端."""
        client = MagicMock()
        # 模拟返回 JSON 格式的问题列表
        client.generate_content.return_value = '["What is Python?", "When was Python created?", "Who created Python?"]'
        return client

    def test_extractor_initialization_without_client(self, mock_settings: Settings) -> None:
        """测试未提供 LLM client 时抛出错误."""
        with pytest.raises(ValueError, match="AtomExtractor 必须提供 llm_client 参数"):
            AtomExtractor(mock_settings, llm_client=None)

    def test_extractor_initialization_with_client(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试正常初始化."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client)

        assert extractor.llm_client is not None
        assert extractor.num_parallel == 1
        assert extractor.max_atoms_per_chunk == 10

    def test_extract_atoms_basic(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试基本 Atom 提取."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client, max_atoms_per_chunk=10)

        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Python is a programming language created by Guido van Rossum in 1991.",
            metadata={"title": "Python History"},
        )

        atoms = extractor.extract_atoms(chunk)

        assert len(atoms) > 0
        assert all(atom.chunk_id == chunk.id for atom in atoms)
        assert all(atom.question for atom in atoms)
        assert atoms[0].question == "What is Python?"

    def test_extract_atoms_with_limit(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试 Atom 数量限制."""
        # 模拟返回很多问题
        mock_llm_client.generate_content.return_value = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"]'

        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client, max_atoms_per_chunk=3)

        chunk = Chunk(id="chunk-1", content="Test content" * 100)

        atoms = extractor.extract_atoms(chunk)

        # 应该被限制为 3 个
        assert len(atoms) == 3

    def test_extract_atoms_empty_chunk(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试空 Chunk."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client)

        chunk = Chunk(id="empty-chunk", content="")

        atoms = extractor.extract_atoms(chunk)

        assert len(atoms) == 0
        # 不应该调用 LLM
        mock_llm_client.generate_content.assert_not_called()

    def test_extract_atoms_batch_single_thread(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试单线程批量提取."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client, num_parallel=1)

        chunks = [Chunk(id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}") for i in range(3)]

        all_atoms = extractor.extract_atoms_batch(chunks, show_progress=False)

        assert len(all_atoms) > 0
        # 每个 chunk 应该提取 3 个 atoms（根据 mock 返回）
        assert len(all_atoms) == 9  # 3 chunks * 3 atoms

    def test_extract_atoms_batch_multi_thread(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试多线程批量提取."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client, num_parallel=2)

        chunks = [Chunk(id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}") for i in range(3)]

        all_atoms = extractor.extract_atoms_batch(chunks, show_progress=False)

        assert len(all_atoms) > 0
        assert len(all_atoms) == 9


class TestAtomExtractorErrorHandling:
    """测试 AtomExtractor 错误处理."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建模拟配置."""
        import os

        os.environ["LOCAL_DEV_MODE"] = "true"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["NEO4J_PASSWORD"] = "test-password"

        settings = Settings.from_local_env()
        return settings

    @pytest.fixture
    def mock_llm_client_with_error(self) -> Any:
        """创建会抛出异常的模拟 LLM 客户端."""
        client = MagicMock()
        client.generate_content.side_effect = Exception("LLM API Error")
        return client

    def test_extract_atoms_with_llm_error(self, mock_settings: Settings, mock_llm_client_with_error: Any) -> None:
        """测试 LLM 调用失败."""
        extractor = AtomExtractor(mock_settings, llm_client=mock_llm_client_with_error)

        chunk = Chunk(id="chunk-1", content="Test content")

        atoms = extractor.extract_atoms(chunk)

        # 出错时应该返回空列表
        assert len(atoms) == 0

    def test_parse_response_with_invalid_json(self, mock_settings: Settings) -> None:
        """测试解析无效 JSON."""
        client = MagicMock()
        client.generate_content.return_value = "This is not JSON"

        extractor = AtomExtractor(mock_settings, llm_client=client)

        chunk = Chunk(id="chunk-1", content="Test content")

        atoms = extractor.extract_atoms(chunk)

        # 解析失败时返回空列表
        assert len(atoms) == 0

    def test_parse_response_with_embedded_json(self, mock_settings: Settings) -> None:
        """测试解析嵌入在文本中的 JSON."""
        client = MagicMock()
        # LLM 可能返回带说明的响应
        client.generate_content.return_value = 'Here are the questions: ["Question 1?", "Question 2?"]'

        extractor = AtomExtractor(mock_settings, llm_client=client)

        chunk = Chunk(id="chunk-1", content="Test content")

        atoms = extractor.extract_atoms(chunk)

        # 应该能提取出 JSON 数组
        assert len(atoms) == 2
        assert atoms[0].question == "Question 1?"

    def test_extract_atoms_batch_with_partial_failures(self, mock_settings: Settings) -> None:
        """测试批量处理时部分失败."""
        client = MagicMock()
        # 第一个成功，第二个失败，第三个成功
        client.generate_content.side_effect = [
            '["Q1?"]',
            Exception("Error"),
            '["Q3?"]',
        ]

        extractor = AtomExtractor(mock_settings, llm_client=client, num_parallel=1)

        chunks = [Chunk(id=f"chunk-{i}", content=f"Content {i}") for i in range(3)]

        all_atoms = extractor.extract_atoms_batch(chunks, show_progress=False)

        # 应该有 2 个成功的 atoms（第1个和第3个chunk）
        assert len(all_atoms) == 2

