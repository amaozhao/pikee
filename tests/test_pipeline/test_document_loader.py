"""DocumentLoader 测试用例."""

import json
from pathlib import Path
from typing import Generator

import pytest

from pikee.pipeline.models.document import Document
from pikee.pipeline.services.document_loader import DocumentLoader


@pytest.fixture
def test_files_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """创建测试文件目录.

    Args:
        tmp_path: pytest 提供的临时目录

    Yields:
        Path: 测试文件目录路径
    """
    yield tmp_path


@pytest.fixture
def sample_txt_file(test_files_dir: Path) -> Path:
    """创建示例 TXT 文件.

    Args:
        test_files_dir: 测试文件目录

    Returns:
        Path: TXT 文件路径
    """
    txt_file = test_files_dir / "sample.txt"
    txt_file.write_text(
        "这是一个测试文档。\n\n这是第二段内容。\n包含多行文本。",
        encoding="utf-8",
    )
    return txt_file


@pytest.fixture
def sample_json_file(test_files_dir: Path) -> Path:
    """创建示例 JSON 文件.

    Args:
        test_files_dir: 测试文件目录

    Returns:
        Path: JSON 文件路径
    """
    json_file = test_files_dir / "sample.json"
    data = {
        "title": "测试文档",
        "content": "这是 JSON 格式的文档内容",
        "author": "测试作者",
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return json_file


@pytest.fixture
def sample_csv_file(test_files_dir: Path) -> Path:
    """创建示例 CSV 文件.

    Args:
        test_files_dir: 测试文件目录

    Returns:
        Path: CSV 文件路径
    """
    csv_file = test_files_dir / "sample.csv"
    csv_file.write_text(
        "姓名,年龄,城市\n张三,25,北京\n李四,30,上海\n王五,28,广州",
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def sample_md_file(test_files_dir: Path) -> Path:
    """创建示例 Markdown 文件.

    Args:
        test_files_dir: 测试文件目录

    Returns:
        Path: Markdown 文件路径
    """
    md_file = test_files_dir / "sample.md"
    md_content = """# 标题

这是一个 Markdown 文档。

## 二级标题

- 列表项 1
- 列表项 2
- 列表项 3

代码示例：

```python
def hello():
    print("Hello, World!")
```
"""
    md_file.write_text(md_content, encoding="utf-8")
    return md_file


class TestDocumentLoader:
    """DocumentLoader 测试类."""

    def test_init_default_params(self) -> None:
        """测试默认参数初始化."""
        loader = DocumentLoader()
        assert loader.mode == "elements"
        assert loader.strategy == "fast"
        assert loader.encoding == "utf-8"

    def test_init_custom_params(self) -> None:
        """测试自定义参数初始化."""
        loader = DocumentLoader(mode="single", strategy="hi_res", encoding="gbk")
        assert loader.mode == "single"
        assert loader.strategy == "hi_res"
        assert loader.encoding == "gbk"

    def test_load_txt_file(self, sample_txt_file: Path) -> None:
        """测试加载 TXT 文件."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_txt_file))

        assert isinstance(doc, Document)
        assert doc.title == "sample"
        assert "这是一个测试文档" in doc.content
        assert "第二段内容" in doc.content
        assert doc.file_type == "txt"
        assert doc.file_path == str(sample_txt_file.absolute())
        assert doc.file_size > 0
        assert doc.char_count > 0
        assert doc.word_count > 0

    def test_load_txt_file_with_custom_title(self, sample_txt_file: Path) -> None:
        """测试加载 TXT 文件并指定自定义标题."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_txt_file), title="自定义标题")

        assert doc.title == "自定义标题"
        assert doc.file_type == "txt"

    def test_load_json_file(self, sample_json_file: Path) -> None:
        """测试加载 JSON 文件."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_json_file))

        assert isinstance(doc, Document)
        assert doc.title == "sample"
        assert doc.file_type == "json"
        assert len(doc.content) > 0

    def test_load_csv_file(self, sample_csv_file: Path) -> None:
        """测试加载 CSV 文件."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_csv_file))

        assert isinstance(doc, Document)
        assert doc.title == "sample"
        assert doc.file_type == "csv"
        assert len(doc.content) > 0

    def test_load_markdown_file(self, sample_md_file: Path) -> None:
        """测试加载 Markdown 文件."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_md_file))

        assert isinstance(doc, Document)
        assert doc.title == "sample"
        assert doc.file_type == "md"
        assert "标题" in doc.content
        assert "Markdown" in doc.content

    def test_load_file_not_found(self) -> None:
        """测试文件不存在的情况."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            loader.load("nonexistent_file.txt")

    def test_load_unsupported_format(self, test_files_dir: Path) -> None:
        """测试不支持的文件格式."""
        unsupported_file = test_files_dir / "test.xyz"
        unsupported_file.write_text("test content", encoding="utf-8")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            loader.load(str(unsupported_file))

    def test_document_metadata(self, sample_txt_file: Path) -> None:
        """测试文档元数据."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_txt_file))

        assert "source" in doc.metadata
        assert "file_name" in doc.metadata
        assert doc.metadata["file_name"] == "sample.txt"
        assert "document_count" in doc.metadata

    def test_document_properties(self, sample_txt_file: Path) -> None:
        """测试文档属性."""
        loader = DocumentLoader()
        doc = loader.load(str(sample_txt_file))

        # 测试 word_count 和 char_count 属性
        assert doc.word_count > 0
        assert doc.char_count > 0
        assert doc.char_count >= doc.word_count

        # 测试 to_dict 方法
        doc_dict = doc.to_dict()
        assert "id" in doc_dict
        assert "title" in doc_dict
        assert "content" in doc_dict
        assert "word_count" in doc_dict
        assert "char_count" in doc_dict

    def test_get_supported_formats(self) -> None:
        """测试获取支持的格式列表."""
        formats = DocumentLoader.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "txt" in formats
        assert "pdf" in formats
        assert "docx" in formats
        assert "md" in formats
        # 确保列表已排序
        assert formats == sorted(formats)

    def test_document_to_dict_and_from_dict(self, sample_txt_file: Path) -> None:
        """测试文档序列化和反序列化."""
        loader = DocumentLoader()
        original_doc = loader.load(str(sample_txt_file))

        # 转换为字典
        doc_dict = original_doc.to_dict()

        # 从字典重建
        restored_doc = Document.from_dict(doc_dict)

        # 验证关键字段
        assert restored_doc.id == original_doc.id
        assert restored_doc.title == original_doc.title
        assert restored_doc.content == original_doc.content
        assert restored_doc.file_type == original_doc.file_type
        assert restored_doc.file_size == original_doc.file_size


class TestDocumentModel:
    """Document 模型测试类."""

    def test_document_creation(self) -> None:
        """测试文档创建."""
        doc = Document(
            title="测试文档",
            content="测试内容",
            file_path="/path/to/file.txt",
            file_type="txt",
        )

        assert doc.title == "测试文档"
        assert doc.content == "测试内容"
        assert doc.file_type == "txt"
        assert doc.id is not None  # 自动生成 ID
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_document_auto_title_from_path(self) -> None:
        """测试从文件路径自动生成标题."""
        doc = Document(
            content="测试内容",
            file_path="/path/to/my_document.pdf",
            file_type="pdf",
        )

        assert doc.title == "my_document"

    def test_document_word_count(self) -> None:
        """测试字数统计."""
        doc = Document(content="Hello world this is a test")
        assert doc.word_count == 6

    def test_document_char_count(self) -> None:
        """测试字符数统计."""
        doc = Document(content="Hello world")
        assert doc.char_count == 11

    def test_document_to_dict(self) -> None:
        """测试转换为字典."""
        doc = Document(
            title="测试",
            content="内容",
            file_type="txt",
            metadata={"key": "value"},
        )

        doc_dict = doc.to_dict()

        assert isinstance(doc_dict, dict)
        assert doc_dict["title"] == "测试"
        assert doc_dict["content"] == "内容"
        assert doc_dict["file_type"] == "txt"
        assert "word_count" in doc_dict
        assert "char_count" in doc_dict

    def test_document_from_dict(self) -> None:
        """测试从字典创建文档."""
        data = {
            "id": "test-id",
            "title": "测试",
            "content": "内容",
            "file_type": "txt",
            "file_size": 100,
            "metadata": {"key": "value"},
        }

        doc = Document.from_dict(data)

        assert doc.id == "test-id"
        assert doc.title == "测试"
        assert doc.content == "内容"
        assert doc.file_type == "txt"
        assert doc.file_size == 100

