"""Atom 数据模型.

定义从 Chunk 中提取的原子知识点结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


@dataclass
class Atom:
    """Atom 模型.

    表示从 Chunk 中提取的原子知识点（通常是问答对）。

    Attributes:
        id: Atom 唯一标识符
        chunk_id: 所属 Chunk ID
        question: 问题
        answer: 答案
        metadata: 其他元数据
        created_at: 创建时间
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    chunk_id: str = ""
    question: str = ""
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典.

        Returns:
            Dict[str, Any]: Atom 字典表示
        """
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """从字典创建 Atom 实例.

        Args:
            data: Atom 数据字典

        Returns:
            Atom: Atom 实例
        """
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)
