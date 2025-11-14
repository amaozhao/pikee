"""测试 Chunk 数据模型."""

from datetime import datetime

import pytest

from pikee.pipeline.models.chunk import Chunk


class TestChunk:
    """测试 Chunk 模型."""

    def test_chunk_initialization_with_defaults(self) -> None:
        """测试使用默认值初始化 Chunk."""
        chunk = Chunk()

        assert chunk.id  # UUID 自动生成
        assert chunk.document_id == ""
        assert chunk.content == ""
        assert chunk.summary == ""
        assert chunk.index == 0
        assert chunk.char_count == 0
        assert chunk.start_index == 0
        assert chunk.end_index == 0
        assert chunk.metadata == {}
        assert isinstance(chunk.created_at, datetime)

    def test_chunk_initialization_with_values(self) -> None:
        """测试使用指定值初始化 Chunk."""
        chunk = Chunk(
            id="test-chunk-1",
            document_id="doc-1",
            content="这是测试内容",
            summary="测试摘要",
            index=0,
            char_count=7,
            start_index=0,
            end_index=7,
            metadata={"key": "value"},
        )

        assert chunk.id == "test-chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "这是测试内容"
        assert chunk.summary == "测试摘要"
        assert chunk.index == 0
        assert chunk.char_count == 7
        assert chunk.start_index == 0
        assert chunk.end_index == 7
        assert chunk.metadata == {"key": "value"}

    def test_chunk_auto_calculate_char_count(self) -> None:
        """测试自动计算字符数."""
        chunk = Chunk(content="测试内容12345")

        assert chunk.char_count == 9

    def test_chunk_char_count_manual_override(self) -> None:
        """测试手动设置字符数."""
        chunk = Chunk(content="测试内容", char_count=100)

        # 手动设置的值优先
        assert chunk.char_count == 100

    def test_chunk_to_dict(self) -> None:
        """测试转换为字典."""
        chunk = Chunk(
            id="test-chunk-1",
            document_id="doc-1",
            content="测试内容",
            summary="摘要",
            index=0,
            char_count=4,
            start_index=0,
            end_index=4,
            metadata={"key": "value"},
        )

        chunk_dict = chunk.to_dict()

        assert chunk_dict["id"] == "test-chunk-1"
        assert chunk_dict["document_id"] == "doc-1"
        assert chunk_dict["content"] == "测试内容"
        assert chunk_dict["summary"] == "摘要"
        assert chunk_dict["index"] == 0
        assert chunk_dict["char_count"] == 4
        assert chunk_dict["start_index"] == 0
        assert chunk_dict["end_index"] == 4
        assert chunk_dict["metadata"] == {"key": "value"}
        assert "created_at" in chunk_dict

    def test_chunk_from_dict(self) -> None:
        """测试从字典创建 Chunk."""
        data = {
            "id": "test-chunk-1",
            "document_id": "doc-1",
            "content": "测试内容",
            "summary": "摘要",
            "index": 0,
            "char_count": 4,
            "start_index": 0,
            "end_index": 4,
            "metadata": {"key": "value"},
            "created_at": "2025-01-01T00:00:00",
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "test-chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "测试内容"
        assert chunk.summary == "摘要"
        assert chunk.index == 0
        assert chunk.char_count == 4
        assert isinstance(chunk.created_at, datetime)

    def test_chunk_to_dict_from_dict_roundtrip(self) -> None:
        """测试字典转换的往返一致性."""
        original_chunk = Chunk(
            id="test-chunk-1",
            document_id="doc-1",
            content="测试内容",
            summary="摘要",
            index=5,
            metadata={"key": "value"},
        )

        chunk_dict = original_chunk.to_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        assert restored_chunk.id == original_chunk.id
        assert restored_chunk.document_id == original_chunk.document_id
        assert restored_chunk.content == original_chunk.content
        assert restored_chunk.summary == original_chunk.summary
        assert restored_chunk.index == original_chunk.index
        assert restored_chunk.metadata == original_chunk.metadata

    def test_chunk_empty_content(self) -> None:
        """测试空内容的 Chunk."""
        chunk = Chunk(content="")

        assert chunk.content == ""
        assert chunk.char_count == 0

    def test_chunk_with_long_content(self) -> None:
        """测试长内容的 Chunk."""
        long_content = "测试内容" * 1000
        chunk = Chunk(content=long_content)

        assert chunk.char_count == len(long_content)
        assert chunk.content == long_content

