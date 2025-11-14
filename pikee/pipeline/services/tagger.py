"""Atom 提取服务.

从 Chunk 中提取原子问题（Atom），用于细粒度检索。
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from tqdm import tqdm

from pikee.infrastructure.config.settings import Settings
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.services.prompts import AtomExtractionPrompts

logger = logging.getLogger(__name__)


class AtomExtractor:
    """Atom 提取器.

    从 Chunk 中使用 LLM 提取原子问题（Atom）。

    原子问题特点:
        - 完整的问题形式（不是关键词）
        - 包含必要实体，独立可理解
        - 可以被源 Chunk 直接回答
        - 用于精确检索匹配

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> extractor = AtomExtractor(settings, llm_client=client)
        >>> chunk = Chunk(content="长文本内容...")
        >>> atoms = extractor.extract_atoms(chunk)
        >>> print(f"提取了 {len(atoms)} 个原子问题")
    """

    def __init__(
        self, settings: Settings, llm_client: Optional[Any] = None, num_parallel: int = 1, max_atoms_per_chunk: int = 10
    ) -> None:
        """初始化 Atom 提取器.

        Args:
            settings: 应用配置实例
            llm_client: LLM 客户端实例（必需）
            num_parallel: 并行处理数量，1 为单线程
            max_atoms_per_chunk: 每个 Chunk 最多提取的 Atom 数量

        Raises:
            ValueError: 当 llm_client 未提供时
        """
        if not llm_client:
            raise ValueError("AtomExtractor 必须提供 llm_client 参数")

        self.settings = settings
        self.llm_client = llm_client
        self.num_parallel = num_parallel
        self.max_atoms_per_chunk = max_atoms_per_chunk
        self.prompts = AtomExtractionPrompts()

        logger.info(
            f"AtomExtractor 初始化完成: num_parallel={self.num_parallel}, "
            f"max_atoms_per_chunk={self.max_atoms_per_chunk}"
        )

    def extract_atoms(self, chunk: Chunk) -> List[Atom]:
        """从单个 Chunk 提取 Atoms.

        Args:
            chunk: 待提取的 Chunk

        Returns:
            List[Atom]: 提取的 Atom 列表
        """
        if not chunk.content or not chunk.content.strip():
            logger.warning(f"Chunk {chunk.id} 内容为空，跳过 Atom 提取")
            return []

        try:
            # 构建提示
            prompt = self.prompts.build_atom_extraction(
                content=chunk.content, chunk_id=chunk.id, title=chunk.metadata.get("title", "")
            )

            # 调用 LLM
            response = self.llm_client.generate_content(prompt)

            # 解析响应
            questions = self._parse_response(response)

            # 限制数量
            if len(questions) > self.max_atoms_per_chunk:
                logger.debug(f"Chunk {chunk.id} 提取了 {len(questions)} 个问题，限制为 {self.max_atoms_per_chunk} 个")
                questions = questions[: self.max_atoms_per_chunk]

            # 转换为 Atom 对象
            atoms: List[Atom] = []
            for question in questions:
                if question.strip():
                    atom = Atom(
                        chunk_id=chunk.id,
                        question=question.strip(),
                        answer="",  # Answer 可以后续填充
                        metadata={
                            "source": chunk.metadata.get("source", ""),
                            "title": chunk.metadata.get("title", ""),
                            "document_id": chunk.document_id,
                        },
                    )
                    atoms.append(atom)

            logger.debug(f"从 Chunk {chunk.id} 提取了 {len(atoms)} 个 Atoms")
            return atoms

        except Exception as e:
            logger.error(f"从 Chunk {chunk.id} 提取 Atoms 失败: {e}", exc_info=True)
            return []

    def extract_atoms_batch(self, chunks: List[Chunk], show_progress: bool = True) -> List[Atom]:
        """批量提取 Atoms.

        Args:
            chunks: Chunk 列表
            show_progress: 是否显示进度条

        Returns:
            List[Atom]: 所有提取的 Atoms
        """
        if not chunks:
            logger.warning("输入 chunks 为空")
            return []

        logger.info(f"开始批量提取 Atoms: {len(chunks)} 个 chunks, 并行度={self.num_parallel}")

        if self.num_parallel == 1:
            return self._single_thread_extract(chunks, show_progress)
        else:
            return self._multi_thread_extract(chunks, show_progress)

    def _single_thread_extract(self, chunks: List[Chunk], show_progress: bool) -> List[Atom]:
        """单线程提取.

        Args:
            chunks: Chunk 列表
            show_progress: 是否显示进度条

        Returns:
            List[Atom]: 所有提取的 Atoms
        """
        all_atoms: List[Atom] = []

        iterator = tqdm(chunks, desc="提取 Atoms") if show_progress else chunks

        for chunk in iterator:
            try:
                atoms = self.extract_atoms(chunk)
                all_atoms.extend(atoms)
            except Exception as e:
                logger.error(f"处理 Chunk {chunk.id} 失败: {e}")
                continue

        logger.info(f"单线程提取完成: 处理 {len(chunks)} 个 chunks, 提取 {len(all_atoms)} 个 Atoms")
        return all_atoms

    def _multi_thread_extract(self, chunks: List[Chunk], show_progress: bool) -> List[Atom]:
        """多线程并行提取.

        Args:
            chunks: Chunk 列表
            show_progress: 是否显示进度条

        Returns:
            List[Atom]: 所有提取的 Atoms
        """
        all_atoms: List[Atom] = []
        pbar = tqdm(total=len(chunks), desc="提取 Atoms") if show_progress else None

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            # 提交所有任务
            future_to_chunk = {executor.submit(self.extract_atoms, chunk): chunk for chunk in chunks}

            # 收集结果
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    atoms = future.result()
                    all_atoms.extend(atoms)
                except Exception as e:
                    logger.error(f"处理 Chunk {chunk.id} 失败: {e}")

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        logger.info(f"多线程提取完成: 处理 {len(chunks)} 个 chunks, 提取 {len(all_atoms)} 个 Atoms")
        return all_atoms

    def _parse_response(self, response: str) -> List[str]:
        """解析 LLM 响应.

        Args:
            response: LLM 响应文本

        Returns:
            List[str]: 问题列表

        Raises:
            ValueError: 当解析失败时
        """
        try:
            # 尝试直接解析 JSON
            questions = json.loads(response.strip())

            if isinstance(questions, list):
                return [str(q) for q in questions if q]

            raise ValueError("响应不是列表格式")

        except json.JSONDecodeError:
            # 尝试提取 JSON 数组
            import re

            json_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if json_match:
                try:
                    questions = json.loads(json_match.group())
                    return [str(q) for q in questions if q]
                except json.JSONDecodeError:
                    pass

            logger.error(f"无法解析响应为 JSON: {response[:200]}")
            raise ValueError(f"无法解析 LLM 响应: {response[:100]}")


class SimpleAtomExtractor:
    """简单的 Atom 提取器.

    不使用 LLM，基于规则提取简单的问题。
    适合测试或不需要高质量 Atom 的场景。

    Examples:
        >>> extractor = SimpleAtomExtractor()
        >>> chunk = Chunk(content="Python is a programming language.")
        >>> atoms = extractor.extract_atoms(chunk)
    """

    def __init__(self, max_atoms_per_chunk: int = 5) -> None:
        """初始化简单提取器.

        Args:
            max_atoms_per_chunk: 每个 Chunk 最多提取的 Atom 数量
        """
        self.max_atoms_per_chunk = max_atoms_per_chunk
        logger.info(f"SimpleAtomExtractor 初始化完成: max_atoms_per_chunk={self.max_atoms_per_chunk}")

    def extract_atoms(self, chunk: Chunk) -> List[Atom]:
        """从 Chunk 提取简单的 Atoms.

        使用规则提取关键句子作为问题。

        Args:
            chunk: 待提取的 Chunk

        Returns:
            List[Atom]: 提取的 Atom 列表
        """
        if not chunk.content or not chunk.content.strip():
            return []

        # 简单规则：将每个句子转换为 "What is mentioned about X?" 形式
        sentences = [s.strip() for s in chunk.content.split("。") if s.strip()]
        sentences = sentences[: self.max_atoms_per_chunk]

        atoms: List[Atom] = []
        for idx, sentence in enumerate(sentences):
            # 生成简单问题
            question = f"What information is provided in statement {idx + 1}?"

            atom = Atom(
                chunk_id=chunk.id,
                question=question,
                answer=sentence,
                metadata={"extraction_method": "rule_based", "document_id": chunk.document_id},
            )
            atoms.append(atom)

        return atoms

    def extract_atoms_batch(self, chunks: List[Chunk], show_progress: bool = True) -> List[Atom]:
        """批量提取 Atoms.

        Args:
            chunks: Chunk 列表
            show_progress: 是否显示进度条

        Returns:
            List[Atom]: 所有提取的 Atoms
        """
        all_atoms: List[Atom] = []

        iterator = tqdm(chunks, desc="提取 Atoms (规则)") if show_progress else chunks

        for chunk in iterator:
            atoms = self.extract_atoms(chunk)
            all_atoms.extend(atoms)

        logger.info(f"规则提取完成: 处理 {len(chunks)} 个 chunks, 提取 {len(all_atoms)} 个 Atoms")
        return all_atoms
