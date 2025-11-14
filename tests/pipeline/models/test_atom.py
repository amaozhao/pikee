"""测试 Atom 数据模型."""

from datetime import datetime

import pytest

from pikee.pipeline.models.atom import Atom


class TestAtom:
    """测试 Atom 模型."""

    def test_atom_initialization_with_defaults(self) -> None:
        """测试使用默认值初始化 Atom."""
        atom = Atom()

        assert atom.id  # UUID 自动生成
        assert atom.chunk_id == ""
        assert atom.question == ""
        assert atom.answer == ""
        assert atom.metadata == {}
        assert isinstance(atom.created_at, datetime)

    def test_atom_initialization_with_values(self) -> None:
        """测试使用指定值初始化 Atom."""
        atom = Atom(
            id="test-atom-1",
            chunk_id="chunk-1",
            question="什么是 PIKE-RAG?",
            answer="PIKE-RAG 是一个知识图谱 RAG 系统",
            metadata={"source": "doc1"},
        )

        assert atom.id == "test-atom-1"
        assert atom.chunk_id == "chunk-1"
        assert atom.question == "什么是 PIKE-RAG?"
        assert atom.answer == "PIKE-RAG 是一个知识图谱 RAG 系统"
        assert atom.metadata == {"source": "doc1"}

    def test_atom_to_dict(self) -> None:
        """测试转换为字典."""
        atom = Atom(
            id="test-atom-1",
            chunk_id="chunk-1",
            question="测试问题",
            answer="测试答案",
            metadata={"key": "value"},
        )

        atom_dict = atom.to_dict()

        assert atom_dict["id"] == "test-atom-1"
        assert atom_dict["chunk_id"] == "chunk-1"
        assert atom_dict["question"] == "测试问题"
        assert atom_dict["answer"] == "测试答案"
        assert atom_dict["metadata"] == {"key": "value"}
        assert "created_at" in atom_dict

    def test_atom_from_dict(self) -> None:
        """测试从字典创建 Atom."""
        data = {
            "id": "test-atom-1",
            "chunk_id": "chunk-1",
            "question": "测试问题",
            "answer": "测试答案",
            "metadata": {"key": "value"},
            "created_at": "2025-01-01T00:00:00",
        }

        atom = Atom.from_dict(data)

        assert atom.id == "test-atom-1"
        assert atom.chunk_id == "chunk-1"
        assert atom.question == "测试问题"
        assert atom.answer == "测试答案"
        assert isinstance(atom.created_at, datetime)

    def test_atom_to_dict_from_dict_roundtrip(self) -> None:
        """测试字典转换的往返一致性."""
        original_atom = Atom(
            id="test-atom-1",
            chunk_id="chunk-1",
            question="原始问题",
            answer="原始答案",
            metadata={"type": "qa"},
        )

        atom_dict = original_atom.to_dict()
        restored_atom = Atom.from_dict(atom_dict)

        assert restored_atom.id == original_atom.id
        assert restored_atom.chunk_id == original_atom.chunk_id
        assert restored_atom.question == original_atom.question
        assert restored_atom.answer == original_atom.answer
        assert restored_atom.metadata == original_atom.metadata

    def test_atom_empty_qa_pair(self) -> None:
        """测试空问答对."""
        atom = Atom(question="", answer="")

        assert atom.question == ""
        assert atom.answer == ""

    def test_atom_with_long_qa(self) -> None:
        """测试长问答对."""
        long_question = "这是一个很长的问题" * 100
        long_answer = "这是一个很长的答案" * 100

        atom = Atom(question=long_question, answer=long_answer)

        assert atom.question == long_question
        assert atom.answer == long_answer
        assert len(atom.question) == len(long_question)
        assert len(atom.answer) == len(long_answer)

    def test_atom_with_complex_metadata(self) -> None:
        """测试复杂元数据."""
        complex_metadata = {
            "source": "document.pdf",
            "page": 10,
            "confidence": 0.95,
            "tags": ["important", "verified"],
            "nested": {"key1": "value1", "key2": [1, 2, 3]},
        }

        atom = Atom(metadata=complex_metadata)

        assert atom.metadata == complex_metadata
        assert atom.metadata["tags"] == ["important", "verified"]
        assert atom.metadata["nested"]["key2"] == [1, 2, 3]

