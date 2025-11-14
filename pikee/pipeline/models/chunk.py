"""Chunk 数据模型.

定义文档切分后的 Chunk 结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


@dataclass
class Chunk:
    """Chunk 模型.

    表示文档切分后的片段,包含内容、摘要和元数据。

    Attributes:
        id: Chunk 唯一标识符
        document_id: 所属文档 ID
        content: Chunk 内容文本
        summary: Chunk 摘要（可选，由 LLM 生成）
        index: Chunk 在文档中的序号
        char_count: 字符数
        start_index: 在原文档中的起始位置
        end_index: 在原文档中的结束位置
        metadata: 其他元数据
        created_at: 创建时间
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    content: str = ""
    summary: str = ""
    index: int = 0
    char_count: int = 0
    start_index: int = 0
    end_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """初始化后处理，自动计算字符数."""
        if not self.char_count and self.content:
            self.char_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典.

        Returns:
            Dict[str, Any]: Chunk 字典表示
        """
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "summary": self.summary,
            "index": self.index,
            "char_count": self.char_count,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建 Chunk 实例.

        Args:
            data: Chunk 数据字典

        Returns:
            Chunk: Chunk 实例
        """
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)
