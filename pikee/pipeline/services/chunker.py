"""文档切分服务.

基于 LLM 的智能文档切分,支持生成摘要和语义边界识别。
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from pikee.infrastructure.config.settings import Settings
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.models.document import Document
from pikee.pipeline.services.prompts import ChunkingPrompts

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ", "，", ", ", " ", ""]


class DocumentChunker:
    """文档切分器.

    基于 RecursiveCharacterTextSplitter 的文档切分服务,
    支持基本切分和 LLM 增强的智能切分。

    基本切分模式:
        使用配置的分隔符和大小参数进行文本切分。

    LLM 增强模式:
        在基本切分基础上,使用 LLM 进行:
        - 语义边界识别
        - Chunk 摘要生成
        - 智能重切分

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(content="长文本内容...")
        >>> chunks = chunker.chunk_document(document)
        >>> print(f"切分为 {len(chunks)} 个 chunks")
    """

    def __init__(
        self,
        settings: Settings,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        enable_llm: bool = False,
        llm_client: Optional[Any] = None,
    ) -> None:
        """初始化文档切分器.

        Args:
            settings: 应用配置实例
            chunk_size: Chunk 大小（字符数）,默认使用配置中的值
            chunk_overlap: Chunk 重叠大小（字符数）,默认使用配置中的值
            separators: 分隔符列表,默认使用预定义分隔符
            length_function: 长度计算函数,默认为 len
            enable_llm: 是否启用 LLM 增强模式
            llm_client: LLM 客户端实例（启用 LLM 模式时需要）
        """
        self.settings = settings
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.length_function = length_function
        self.enable_llm = enable_llm
        self.llm_client = llm_client
        self.prompts = ChunkingPrompts()

        # 验证 LLM 配置
        if self.enable_llm and not self.llm_client:
            raise ValueError("启用 LLM 增强模式时必须提供 llm_client 参数")

        # 创建基础分割器
        self._base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            keep_separator=False,
            add_start_index=True,
            strip_whitespace=True,
        )

        logger.info(
            f"DocumentChunker 初始化完成: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, enable_llm={self.enable_llm}"
        )

    def chunk_document(self, document: Document, show_progress: bool = False) -> List[Chunk]:
        """切分文档为多个 Chunks.

        Args:
            document: 待切分的文档
            show_progress: 是否显示进度条

        Returns:
            List[Chunk]: 切分后的 Chunks 列表

        Raises:
            ValueError: 当文档内容为空时
        """
        if not document.content or not document.content.strip():
            logger.warning(f"文档 {document.id} 内容为空,跳过切分")
            return []

        logger.info(f"开始切分文档: {document.id}, 长度: {len(document.content)} 字符")

        try:
            if self.enable_llm and self.llm_client:
                chunks = self._chunk_with_llm(document, show_progress)
            else:
                chunks = self._chunk_basic(document, show_progress)

            logger.info(f"文档 {document.id} 切分完成: {len(chunks)} 个 chunks")
            return chunks

        except Exception as e:
            logger.error(f"文档 {document.id} 切分失败: {e}", exc_info=True)
            raise

    def chunk_documents(self, documents: List[Document], show_progress: bool = True) -> List[Chunk]:
        """批量切分多个文档.

        Args:
            documents: 待切分的文档列表
            show_progress: 是否显示进度条

        Returns:
            List[Chunk]: 所有文档的 Chunks 列表
        """
        all_chunks: List[Chunk] = []

        iterator = tqdm(documents, desc="切分文档") if show_progress else documents

        for document in iterator:
            try:
                chunks = self.chunk_document(document, show_progress=False)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"文档 {document.id} 切分失败: {e}")
                continue

        logger.info(f"批量切分完成: {len(documents)} 个文档, 共 {len(all_chunks)} 个 chunks")
        return all_chunks

    def _chunk_basic(self, document: Document, show_progress: bool = False) -> List[Chunk]:
        """基本切分模式.

        使用 RecursiveCharacterTextSplitter 进行简单切分。

        Args:
            document: 待切分的文档
            show_progress: 是否显示进度条

        Returns:
            List[Chunk]: 切分后的 Chunks 列表
        """
        text = document.content.strip()

        # 使用基础分割器切分
        langchain_docs = self._base_splitter.create_documents(
            texts=[text], metadatas=[{"document_id": document.id, **document.metadata}]
        )

        # 转换为 Chunk 对象
        chunks: List[Chunk] = []
        for idx, lc_doc in enumerate(langchain_docs):
            chunk = Chunk(
                document_id=document.id,
                content=lc_doc.page_content,
                index=idx,
                char_count=len(lc_doc.page_content),
                start_index=lc_doc.metadata.get("start_index", 0),
                end_index=lc_doc.metadata.get("start_index", 0) + len(lc_doc.page_content),
                metadata={"source": document.file_path, "title": document.title, **lc_doc.metadata},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_with_llm(self, document: Document, show_progress: bool = False) -> List[Chunk]:
        """LLM 增强切分模式.

        使用 LLM 进行智能切分和摘要生成。

        工作流程:
            1. 使用基础分割器进行初步切分
            2. 对第一个 chunk 生成摘要
            3. 迭代处理剩余 chunks:
               - 如果只剩一个 chunk,生成最终摘要
               - 如果有多个 chunks,让 LLM 评估前两个 chunk 的切分边界
               - 根据 LLM 建议重新切分
            4. 为每个 chunk 生成摘要

        Args:
            document: 待切分的文档
            show_progress: 是否显示进度条

        Returns:
            List[Chunk]: 切分后的 Chunks 列表（包含摘要）

        Raises:
            RuntimeError: 当 LLM 客户端未初始化时
        """
        if not self.llm_client:
            raise RuntimeError("LLM 增强模式需要提供 llm_client")

        text = document.content.strip()
        chunks: List[Chunk] = []

        # 第一步: 生成第一个 chunk 的摘要
        chunk_summary = self._generate_first_chunk_summary(text, document)

        # 第二步: 初步切分
        text_chunks = self._base_splitter.split_text(text)

        chunk_index = 0
        position = 0

        # 第三步: 迭代处理
        while text_chunks:
            if len(text_chunks) == 1:
                # 最后一个 chunk: 生成优化的摘要
                chunk_summary = self._generate_last_chunk_summary(text_chunks[0], chunk_summary, document)

                chunk = Chunk(
                    document_id=document.id,
                    content=text_chunks[0],
                    summary=chunk_summary,
                    index=chunk_index,
                    char_count=len(text_chunks[0]),
                    start_index=position,
                    end_index=position + len(text_chunks[0]),
                    metadata={"source": document.file_path, "title": document.title, "has_llm_summary": True},
                )
                chunks.append(chunk)

                logger.debug(f"最后一个 chunk 已添加 (index={chunk_index}, length={len(text_chunks[0])})")
                break

            else:
                # 多个 chunks: 让 LLM 重新评估切分边界
                result = self._resplit_and_generate_summary(text, text_chunks, chunk_summary, document)

                new_chunk_content, new_summary, next_summary, dropped_len = result

                # 跳过空 chunk
                if not new_chunk_content or len(new_chunk_content.strip()) == 0:
                    logger.debug("跳过空的重切分 chunk")
                    chunk_summary = next_summary
                    # 合并前两个 chunk 重新处理
                    if len(text_chunks) >= 2:
                        text_chunks = [text_chunks[0] + text_chunks[1]] + text_chunks[2:]
                    continue

                # 添加 chunk
                chunk = Chunk(
                    document_id=document.id,
                    content=new_chunk_content,
                    summary=new_summary,
                    index=chunk_index,
                    char_count=len(new_chunk_content),
                    start_index=position,
                    end_index=position + len(new_chunk_content),
                    metadata={"source": document.file_path, "title": document.title, "has_llm_summary": True},
                )
                chunks.append(chunk)

                logger.debug(f"Chunk 已添加 (index={chunk_index}, length={len(new_chunk_content)})")

                # 更新状态
                chunk_index += 1
                position += dropped_len
                text = text[dropped_len:].strip()
                chunk_summary = next_summary

                # 重新切分剩余文本
                text_chunks = self._base_splitter.split_text(text)

        return chunks

    def _generate_first_chunk_summary(self, text: str, document: Document) -> str:
        """生成第一个 chunk 的摘要.

        Args:
            text: 文档文本
            document: 文档对象

        Returns:
            str: 第一个 chunk 的摘要
        """
        # 获取第一个 chunk 的内容
        chunks = self._base_splitter.split_text(text)
        if not chunks:
            return ""

        first_chunk_start = text.find(chunks[0])
        content_for_summary = text[: first_chunk_start + len(chunks[0])]

        # 构建提示
        prompt = self.prompts.build_first_chunk_summary(content_for_summary, document)

        # 调用 LLM
        try:
            response = self.llm_client.generate_content(prompt)
            summary = self._parse_summary_response(response)
            logger.debug(f"第一个 chunk 摘要已生成: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"生成第一个 chunk 摘要失败: {e}")
            return ""

    def _generate_last_chunk_summary(self, chunk: str, prev_summary: str, document: Document) -> str:
        """生成最后一个 chunk 的优化摘要.

        Args:
            chunk: Chunk 内容
            prev_summary: 前一个 chunk 的摘要
            document: 文档对象

        Returns:
            str: 优化后的摘要
        """
        prompt = self.prompts.build_last_chunk_summary(chunk, prev_summary, document)

        try:
            response = self.llm_client.generate_content(prompt)
            summary = self._parse_summary_response(response)
            logger.debug(f"最后 chunk 摘要已生成: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"生成最后 chunk 摘要失败: {e}")
            return prev_summary

    def _resplit_and_generate_summary(
        self, text: str, chunks: List[str], chunk_summary: str, document: Document
    ) -> Tuple[str, str, str, int]:
        """重新切分并生成摘要.

        让 LLM 评估前两个 chunk 的切分边界,并生成新的摘要。

        Args:
            text: 原始文本
            chunks: 当前的 chunks 列表
            chunk_summary: 当前 chunk 的摘要
            document: 文档对象

        Returns:
            Tuple[str, str, str, int]: (新chunk内容, 新chunk摘要, 下一个chunk摘要, 已处理的字符数)
        """
        if len(chunks) < 2:
            raise ValueError("重切分需要至少 2 个 chunks")

        # 获取前两个 chunk 的内容
        text_to_resplit = text[: len(chunks[0]) + len(chunks[1])]

        # 构建提示
        prompt = self.prompts.build_resplit(text_to_resplit, chunk_summary, document)

        try:
            response = self.llm_client.generate_content(prompt)
            result = self._parse_resplit_response(response, text_to_resplit)

            new_chunk, new_summary, next_summary, split_pos = result

            logger.debug(
                f"重切分完成: new_chunk_len={len(new_chunk)}, "
                f"split_pos={split_pos}, next_summary={next_summary[:30]}..."
            )

            return new_chunk, new_summary, next_summary, split_pos

        except Exception as e:
            logger.error(f"重切分失败: {e}")
            # 降级: 使用第一个 chunk
            return chunks[0], chunk_summary, chunk_summary, len(chunks[0])

    def _parse_summary_response(self, response: str) -> str:
        """解析摘要响应.

        Args:
            response: LLM 响应

        Returns:
            str: 提取的摘要
        """
        return response.strip()

    def _parse_resplit_response(self, response: str, original_text: str) -> Tuple[str, str, str, int]:
        """解析重切分响应.

        Args:
            response: LLM 响应
            original_text: 原始文本

        Returns:
            Tuple[str, str, str, int]: (第一部分内容, 第一部分摘要, 第二部分摘要, 切分位置)

        Raises:
            ValueError: 当解析失败时
        """
        import re

        # 提取 endline
        endline_match = re.search(r"<endline>(\d+)</endline>", response)
        if not endline_match:
            raise ValueError("无法解析 endline")

        end_line_num = int(endline_match.group(1))

        # 提取两个摘要
        summary_matches = re.findall(r"<summary>(.*?)</summary>", response, re.DOTALL)
        if len(summary_matches) < 2:
            raise ValueError("无法解析摘要")

        first_summary = summary_matches[0].strip()
        second_summary = summary_matches[1].strip()

        # 根据行号切分文本
        lines = original_text.split("\n")
        if end_line_num > len(lines):
            end_line_num = len(lines)

        first_part = "\n".join(lines[:end_line_num])
        split_position = len(first_part)

        return first_part, first_summary, second_summary, split_position


class SimpleChunker:
    """简单文档切分器.

    提供基于固定大小的简单切分,不使用 LLM。
    适合快速处理或不需要语义边界识别的场景。

    Examples:
        >>> chunker = SimpleChunker(chunk_size=1000, chunk_overlap=200)
        >>> document = Document(content="长文本...")
        >>> chunks = chunker.chunk_document(document)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
    ) -> None:
        """初始化简单切分器.

        Args:
            chunk_size: Chunk 大小（字符数）
            chunk_overlap: Chunk 重叠大小（字符数）
            separators: 分隔符列表,默认使用预定义分隔符
            length_function: 长度计算函数,默认为 len
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.length_function = length_function

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            keep_separator=False,
            add_start_index=True,
            strip_whitespace=True,
        )

        logger.info(f"SimpleChunker 初始化完成: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk_document(self, document: Document) -> List[Chunk]:
        """切分文档为多个 Chunks.

        Args:
            document: 待切分的文档

        Returns:
            List[Chunk]: 切分后的 Chunks 列表
        """
        if not document.content or not document.content.strip():
            logger.warning(f"文档 {document.id} 内容为空,跳过切分")
            return []

        text = document.content.strip()

        # 切分
        langchain_docs = self._splitter.create_documents(
            texts=[text], metadatas=[{"document_id": document.id, **document.metadata}]
        )

        # 转换为 Chunk 对象
        chunks: List[Chunk] = []
        for idx, lc_doc in enumerate(langchain_docs):
            chunk = Chunk(
                document_id=document.id,
                content=lc_doc.page_content,
                index=idx,
                char_count=len(lc_doc.page_content),
                start_index=lc_doc.metadata.get("start_index", 0),
                end_index=lc_doc.metadata.get("start_index", 0) + len(lc_doc.page_content),
                metadata={"source": document.file_path, "title": document.title, **lc_doc.metadata},
            )
            chunks.append(chunk)

        logger.info(f"文档 {document.id} 切分完成: {len(chunks)} 个 chunks")
        return chunks

    def chunk_documents(self, documents: List[Document], show_progress: bool = True) -> List[Chunk]:
        """批量切分多个文档.

        Args:
            documents: 待切分的文档列表
            show_progress: 是否显示进度条

        Returns:
            List[Chunk]: 所有文档的 Chunks 列表
        """
        all_chunks: List[Chunk] = []

        iterator = tqdm(documents, desc="切分文档") if show_progress else documents

        for document in iterator:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"文档 {document.id} 切分失败: {e}")
                continue

        logger.info(f"批量切分完成: {len(documents)} 个文档, 共 {len(all_chunks)} 个 chunks")
        return all_chunks
