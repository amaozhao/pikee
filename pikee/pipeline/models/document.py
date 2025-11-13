"""文档数据模型.

定义文档处理流程中的核心数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class Document:
    """文档模型.

    表示一个完整的文档，包含内容和元数据。

    Attributes:
        id: 文档唯一标识符
        title: 文档标题
        content: 文档内容文本
        file_path: 源文件路径
        file_type: 文件类型（如 pdf, docx）
        file_size: 文件大小（字节）
        page_count: 页数（如果适用）
        metadata: 其他元数据
        created_at: 创建时间
        updated_at: 更新时间
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    content: str = ""
    file_path: str = ""
    file_type: str = ""
    file_size: int = 0
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """初始化后处理，自动设置标题."""
        if not self.title and self.file_path:
            # 从文件路径提取文件名作为标题
            from pathlib import Path

            self.title = Path(self.file_path).stem

    @property
    def word_count(self) -> int:
        """获取文档字数.

        Returns:
            int: 文档字数
        """
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """获取文档字符数.

        Returns:
            int: 文档字符数
        """
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典.

        Returns:
            Dict[str, Any]: 文档字典表示
        """
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建文档实例.

        Args:
            data: 文档数据字典

        Returns:
            Document: 文档实例
        """
        # 处理时间字段
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # 移除计算属性
        data.pop("word_count", None)
        data.pop("char_count", None)

        return cls(**data)

