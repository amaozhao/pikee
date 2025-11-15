"""文档加载服务.

基于 LangChain 和 Unstructured 实现多格式文档加载。
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document as LangChainDocument

from pikee.pipeline.models.document import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """统一的文档加载器.

    使用 Unstructured 处理复杂格式（PDF, DOCX, XLSX 等），
    使用专用加载器处理简单格式（TXT, CSV, JSON）以获得更好的性能。

    支持的格式：
        - PDF (.pdf)
        - Word (.docx, .doc)
        - Excel (.xlsx, .xls)
        - PowerPoint (.pptx, .ppt)
        - Markdown (.md)
        - HTML (.html, .htm)
        - Text (.txt)
        - CSV (.csv)
        - JSON (.json)

    Examples:
        >>> loader = DocumentLoader()
        >>> doc = loader.load("report.pdf")
        >>> print(doc.title, doc.char_count)

        >>> # 使用高精度模式
        >>> loader = DocumentLoader(strategy="hi_res")
        >>> doc = loader.load("scanned.pdf")
    """

    # 支持的文件格式
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".md",
        ".html",
        ".htm",
        ".txt",
        ".csv",
        ".json",
    }

    # 简单格式使用专用加载器（性能更好）
    SIMPLE_LOADERS = {
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(self, mode: str = "elements", strategy: str = "fast", encoding: str = "utf-8"):
        """初始化文档加载器.

        Args:
            mode: Unstructured 的模式
                - "single": 整个文档作为一个元素
                - "elements": 按元素分割（推荐）
                - "paged": 按页分割
            strategy: Unstructured 的策略
                - "fast": 快速模式（推荐）
                - "hi_res": 高精度模式（更慢但更准确）
                - "ocr_only": 只使用 OCR
            encoding: 文本文件编码，默认 utf-8
        """
        self.mode = mode
        self.strategy = strategy
        self.encoding = encoding
        logger.info(f"初始化 DocumentLoader: mode={mode}, strategy={strategy}, encoding={encoding}")

    def load(self, file_path: str, title: Optional[str] = None) -> Document:
        """加载文档.

        Args:
            file_path: 文档文件路径
            title: 文档标题，如果不提供则使用文件名

        Returns:
            Document: 加载的文档对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        path = Path(file_path)

        # 验证文件存在
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 验证文件格式
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"不支持的文件格式: {suffix}")
            raise ValueError(f"不支持的文件格式: {suffix}。支持的格式: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}")

        logger.info(f"开始加载文档: {file_path} (格式: {suffix})")

        try:
            # 加载 LangChain 文档
            langchain_docs = self._load_documents(path, suffix)
            logger.info(f"加载完成: {len(langchain_docs)} 个文档片段")

            # 转换为自定义 Document 对象
            document = self._convert_to_document(langchain_docs, path, title)
            logger.info(f"文档转换完成: {document.title}, 字符数: {document.char_count}, 字数: {document.word_count}")

            return document

        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {e}")
            raise

    def _load_documents(self, path: Path, suffix: str) -> List[LangChainDocument]:
        """使用 LangChain 加载器加载文档.

        Args:
            path: 文件路径
            suffix: 文件后缀

        Returns:
            List[LangChainDocument]: LangChain 文档列表
        """
        # 简单格式使用专用加载器
        if suffix in self.SIMPLE_LOADERS:
            loader_class = self.SIMPLE_LOADERS[suffix]
            if suffix == ".txt":
                loader = loader_class(str(path), encoding=self.encoding)
            elif suffix == ".json":
                # JSONLoader 需要指定 jq_schema
                loader = loader_class(str(path), jq_schema=".", text_content=False)
            else:
                loader = loader_class(str(path))
            logger.debug(f"使用专用加载器: {loader_class.__name__}")
        else:
            # 复杂格式使用 UnstructuredFileLoader
            # 处理pdf文档时，需要考虑同步处理表格和图片，暂时考虑ocr解析图片内容
            loader = UnstructuredFileLoader(str(path), mode=self.mode, strategy=self.strategy)
            logger.debug(f"使用 UnstructuredFileLoader: mode={self.mode}")

        return loader.load()

    def _convert_to_document(
        self, langchain_docs: List[LangChainDocument], path: Path, title: Optional[str] = None
    ) -> Document:
        """将 LangChain 文档列表转换为单个 Document 对象.

        Args:
            langchain_docs: LangChain 文档列表
            path: 文件路径
            title: 文档标题

        Returns:
            Document: 转换后的文档对象
        """
        # 合并所有文档内容
        content = "\n\n".join(doc.page_content for doc in langchain_docs)

        # 提取元数据（使用第一个文档的元数据）
        metadata = langchain_docs[0].metadata if langchain_docs else {}

        # 获取文件大小
        file_size = path.stat().st_size

        # 推断页数（如果元数据中有）
        page_count = None
        if "page" in metadata:
            # 如果有 page 字段，页数就是文档片段数量
            page_count = len(langchain_docs)
        elif "total_pages" in metadata:
            page_count = metadata.get("total_pages")

        # 创建 Document 对象
        document = Document(
            title=title or path.stem,
            content=content,
            file_path=str(path.absolute()),
            file_type=path.suffix[1:],  # 去掉前面的点
            file_size=file_size,
            page_count=page_count,
            metadata={**metadata, "source": str(path), "file_name": path.name, "document_count": len(langchain_docs)},
        )

        return document

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的文件格式列表.

        Returns:
            List[str]: 支持的文件格式列表（不含点）
        """
        return sorted([ext[1:] for ext in cls.SUPPORTED_EXTENSIONS])
