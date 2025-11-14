"""测试文档切分服务."""

from typing import Any

import pytest

from pikee.infrastructure.config.settings import Settings
from pikee.pipeline.models.document import Document
from pikee.pipeline.services.chunker import DocumentChunker, SimpleChunker


class TestSimpleChunker:
    """测试 SimpleChunker 类."""

    def test_simple_chunker_initialization(self) -> None:
        """测试 SimpleChunker 初始化."""
        chunker = SimpleChunker(chunk_size=500, chunk_overlap=100)

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunk_document_basic(self) -> None:
        """测试基本文档切分."""
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)

        document = Document(
            id="test-doc-1",
            title="测试文档",
            content="这是一个测试文档。" * 50,  # 重复多次以产生多个 chunks
            file_path="/test/doc.txt",
        )

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        assert all(chunk.document_id == document.id for chunk in chunks)
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.char_count > 0 for chunk in chunks)

    def test_chunk_empty_document(self) -> None:
        """测试空文档切分."""
        chunker = SimpleChunker()

        document = Document(id="empty-doc", content="")

        chunks = chunker.chunk_document(document)

        assert len(chunks) == 0

    def test_chunk_documents_batch(self) -> None:
        """测试批量文档切分."""
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)

        documents = [Document(id=f"doc-{i}", content=f"文档内容 {i}。" * 30) for i in range(3)]

        all_chunks = chunker.chunk_documents(documents, show_progress=False)

        assert len(all_chunks) > 0
        # 每个文档应该产生多个 chunks
        assert len(all_chunks) >= len(documents)


class TestDocumentChunker:
    """测试 DocumentChunker 类."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建模拟配置."""
        # 使用环境变量模式避免连接 Apollo
        import os

        os.environ["LOCAL_DEV_MODE"] = "true"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["NEO4J_PASSWORD"] = "test-password"

        settings = Settings.from_local_env()
        return settings

    def test_document_chunker_initialization(self, mock_settings: Settings) -> None:
        """测试 DocumentChunker 初始化."""
        chunker = DocumentChunker(mock_settings)

        assert chunker.chunk_size == mock_settings.chunk_size
        assert chunker.chunk_overlap == mock_settings.chunk_overlap
        assert not chunker.enable_llm

    def test_chunk_document_basic_mode(self, mock_settings: Settings) -> None:
        """测试基本模式文档切分."""
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20)

        document = Document(
            id="test-doc-1", title="测试文档", content="这是一个测试文档的内容。" * 50, file_path="/test/doc.txt"
        )

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        assert all(chunk.document_id == document.id for chunk in chunks)
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.index >= 0 for chunk in chunks)
        assert all(chunk.start_index >= 0 for chunk in chunks)
        assert all(chunk.end_index > chunk.start_index for chunk in chunks)

    def test_chunk_metadata_preservation(self, mock_settings: Settings) -> None:
        """测试元数据保留."""
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20)

        document = Document(
            id="test-doc-2",
            title="元数据测试",
            content="测试元数据保留。" * 20,
            file_path="/test/meta.txt",
            metadata={"author": "测试作者", "category": "测试类别"},
        )

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["title"] == document.title
            assert chunk.metadata["source"] == document.file_path

    def test_chunk_empty_document(self, mock_settings: Settings) -> None:
        """测试空文档切分."""
        chunker = DocumentChunker(mock_settings)

        document = Document(id="empty-doc", content="")

        chunks = chunker.chunk_document(document)

        assert len(chunks) == 0

    def test_chunk_documents_batch(self, mock_settings: Settings) -> None:
        """测试批量文档切分."""
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20)

        documents = [Document(id=f"doc-{i}", title=f"文档{i}", content=f"这是文档 {i} 的内容。" * 30) for i in range(3)]

        all_chunks = chunker.chunk_documents(documents, show_progress=False)

        assert len(all_chunks) > 0
        assert len(all_chunks) >= len(documents)

        # 验证 chunk 索引的连续性
        for doc_id in [f"doc-{i}" for i in range(3)]:
            doc_chunks = [c for c in all_chunks if c.document_id == doc_id]
            indices = [c.index for c in doc_chunks]
            assert indices == list(range(len(indices)))

    def test_custom_separators(self, mock_settings: Settings) -> None:
        """测试自定义分隔符."""
        custom_separators = ["\n\n", "\n", ". "]
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20, separators=custom_separators)

        document = Document(id="test-doc-3", content="第一段。\n\n第二段。\n\n第三段。")

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        assert chunker.separators == custom_separators


class TestDocumentChunkerErrorHandling:
    """测试 DocumentChunker 错误处理."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """创建模拟配置."""
        import os

        os.environ["LOCAL_DEV_MODE"] = "true"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["NEO4J_PASSWORD"] = "test-password"

        settings = Settings.from_local_env()
        return settings

    def test_chunk_document_with_invalid_params(self, mock_settings: Settings) -> None:
        """测试无效参数."""
        with pytest.raises(ValueError):
            # chunk_overlap 大于 chunk_size
            DocumentChunker(mock_settings, chunk_size=50, chunk_overlap=100)

    def test_chunk_document_whitespace_only(self, mock_settings: Settings) -> None:
        """测试仅包含空白字符的文档."""
        chunker = DocumentChunker(mock_settings)

        document = Document(id="whitespace-doc", content="   \n\n\t\t   ")

        chunks = chunker.chunk_document(document)

        # 空白字符会被 strip，应返回空列表
        assert len(chunks) == 0

    def test_chunk_document_very_long_content(self, mock_settings: Settings) -> None:
        """测试超长文档."""
        chunker = DocumentChunker(mock_settings, chunk_size=500, chunk_overlap=50)

        # 生成一个很长的文档
        long_content = "这是一段很长的文本内容。" * 1000
        document = Document(id="long-doc", content=long_content)

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        # 验证所有 chunk 的大小都合理
        for chunk in chunks:
            assert len(chunk.content) <= 600  # chunk_size + 一些容差
            assert chunk.document_id == document.id

    def test_chunk_document_special_characters(self, mock_settings: Settings) -> None:
        """测试特殊字符."""
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20)

        special_content = '测试特殊字符：\n\n😀🎉🔥\n\n<html></html>\n\n{"json": true}'
        document = Document(id="special-doc", content=special_content)

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 0
        # 验证特殊字符被正确保留
        all_content = "".join(chunk.content for chunk in chunks)
        assert "😀" in all_content
        assert "<html>" in all_content
        assert '{"json"' in all_content

    def test_chunk_documents_with_partial_failures(self, mock_settings: Settings) -> None:
        """测试批量处理中部分文档失败."""
        chunker = DocumentChunker(mock_settings, chunk_size=100, chunk_overlap=20)

        documents = [
            Document(id="doc-1", content="正常文档内容" * 10),
            Document(id="doc-2", content=""),  # 空文档
            Document(id="doc-3", content="正常文档内容" * 10),
        ]

        all_chunks = chunker.chunk_documents(documents, show_progress=False)

        # 应该只有 doc-1 和 doc-3 的 chunks
        assert len(all_chunks) > 0
        doc_ids = {chunk.document_id for chunk in all_chunks}
        assert "doc-1" in doc_ids
        assert "doc-3" in doc_ids
        assert "doc-2" not in doc_ids  # 空文档被跳过

    def test_chunk_document_single_sentence(self, mock_settings: Settings) -> None:
        """测试单句文档."""
        chunker = DocumentChunker(mock_settings, chunk_size=1000, chunk_overlap=0)

        document = Document(id="single-sentence", content="这是一句话。")

        chunks = chunker.chunk_document(document)

        assert len(chunks) == 1
        # 注意：句号是分隔符，会被移除（keep_separator=False）
        assert chunks[0].content == "这是一句话"
        assert chunks[0].index == 0


class TestDocumentChunkerLLMMode:
    """测试 DocumentChunker LLM 增强模式."""

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
        from unittest.mock import MagicMock

        client = MagicMock()
        # 模拟不同的响应
        client.generate_content.side_effect = [
            "This is a summary of the first chunk.",  # 第一个 chunk 摘要
            "思考: Split at line 10\n\n<result>\n<chunk>\n<endline>10</endline>\n<summary>First part summary</summary>\n</chunk>\n<chunk>\n<summary>Second part summary</summary>\n</chunk>\n</result>",  # 重切分
            "Final summary of the last chunk.",  # 最后 chunk 摘要
        ]
        return client

    def test_chunk_with_llm_disabled_without_client(self, mock_settings: Settings) -> None:
        """测试启用 LLM 但未提供 client."""
        with pytest.raises(ValueError, match="启用 LLM 增强模式时必须提供 llm_client 参数"):
            # 在初始化时就会抛出错误
            DocumentChunker(mock_settings, enable_llm=True, llm_client=None)

    def test_chunk_with_llm_mode_initialization(self, mock_settings: Settings, mock_llm_client: Any) -> None:
        """测试 LLM 模式初始化."""
        chunker = DocumentChunker(mock_settings, enable_llm=True, llm_client=mock_llm_client)

        assert chunker.enable_llm is True
        assert chunker.llm_client is not None
        assert chunker.prompts is not None
