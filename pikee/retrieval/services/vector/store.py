"""向量检索服务.

提供对 Chunk 和 Atom 向量的检索接口。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.qdrant import QdrantClient
from pikee.infrastructure.llm.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索服务.

    提供统一的检索接口，支持 Chunk 和 Atom 两种检索模式。

    检索流程:
        1. 将查询文本转换为向量
        2. 在 Qdrant 中检索相似向量
        3. 返回结果

    Examples:
        >>> retriever = VectorRetriever(settings, qdrant_client, embedder)
        >>> results = retriever.search_chunks("Python 的特点是什么？", top_k=10)
        >>> results = retriever.search_atoms("什么是 GIL？", top_k=5)
    """

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedder: BaseEmbedder,
    ) -> None:
        """初始化向量检索服务.

        Args:
            settings: 应用配置
            qdrant_client: Qdrant 客户端
            embedder: Embedding 服务
        """
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedder = embedder

        self.chunk_collection = settings.qdrant_collection_chunks
        self.atom_collection = settings.qdrant_collection_atoms

        logger.info("VectorRetriever 初始化完成")

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_by: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索相似的 Chunks.

        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值（可选）
            filter_by: 过滤条件（可选，如 {"document_id": "doc123"}）

        Returns:
            List[Dict]: Chunk 列表，每个包含:
                - chunk_id: Chunk ID
                - document_id: 文档 ID
                - content: 内容
                - summary: 摘要
                - title: 标题
                - source: 来源
                - score: 相似度分数

        Examples:
            >>> results = retriever.search_chunks("Python 特点", top_k=5)
            >>> for chunk in results:
            >>>     print(f"{chunk['title']}: {chunk['score']:.3f}")
        """
        try:
            # 1. 生成查询向量
            query_vector = self.embedder.embed_text(query)

            # 2. 检索
            raw_results = self.qdrant_client.search_vectors(
                collection_name=self.chunk_collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_by,
            )

            # 3. 转换结果
            results = []
            for chunk_id, score, payload in raw_results:
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id", chunk_id),
                        "document_id": payload.get("document_id", ""),
                        "content": payload.get("content", ""),
                        "summary": payload.get("summary", ""),
                        "title": payload.get("title", ""),
                        "source": payload.get("source", ""),
                        "index": payload.get("index", -1),
                        "score": score,
                    }
                )

            logger.debug(f"检索到 {len(results)} 个 Chunks")
            return results

        except Exception as e:
            logger.error(f"Chunk 检索失败: {e}", exc_info=True)
            return []

    def search_atoms(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_by: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索相似的 Atoms.

        Args:
            query: 查询文本（问题）
            top_k: 返回结果数量
            score_threshold: 分数阈值（可选）
            filter_by: 过滤条件（可选）

        Returns:
            List[Dict]: Atom 列表，每个包含:
                - atom_id: Atom ID
                - question: 问题
                - answer: 答案
                - source_chunk_id: 源 Chunk ID
                - document_id: 文档 ID
                - title: 标题
                - score: 相似度分数

        Examples:
            >>> results = retriever.search_atoms("什么是 GIL？", top_k=3)
            >>> for atom in results:
            >>>     print(f"Q: {atom['question']}")
            >>>     print(f"A: {atom['answer']}")
        """
        try:
            # 1. 生成查询向量
            query_vector = self.embedder.embed_text(query)

            # 2. 检索
            raw_results = self.qdrant_client.search_vectors(
                collection_name=self.atom_collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_by,
            )

            # 3. 转换结果
            results = []
            for atom_id, score, payload in raw_results:
                results.append(
                    {
                        "atom_id": payload.get("atom_id", atom_id),
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                        "source_chunk_id": payload.get("source_chunk_id", ""),
                        "document_id": payload.get("document_id", ""),
                        "title": payload.get("title", ""),
                        "score": score,
                    }
                )

            logger.debug(f"检索到 {len(results)} 个 Atoms")
            return results

        except Exception as e:
            logger.error(f"Atom 检索失败: {e}", exc_info=True)
            return []

    def hybrid_search(
        self,
        query: str,
        top_k_chunks: int = 5,
        top_k_atoms: int = 5,
        score_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """混合检索（同时检索 Chunks 和 Atoms）.

        Args:
            query: 查询文本
            top_k_chunks: Chunk 返回数量
            top_k_atoms: Atom 返回数量
            score_threshold: 分数阈值

        Returns:
            Tuple[List[Dict], List[Dict]]: (Chunks, Atoms)

        Examples:
            >>> chunks, atoms = retriever.hybrid_search("Python 特点", top_k_chunks=3, top_k_atoms=3)
            >>> print(f"找到 {len(chunks)} 个 Chunks 和 {len(atoms)} 个 Atoms")
        """
        chunks = self.search_chunks(query, top_k=top_k_chunks, score_threshold=score_threshold)
        atoms = self.search_atoms(query, top_k=top_k_atoms, score_threshold=score_threshold)

        logger.info(f"混合检索完成: {len(chunks)} Chunks + {len(atoms)} Atoms")
        return chunks, atoms


def create_vector_retriever(
    settings: Settings,
    qdrant_client: QdrantClient,
    embedder: BaseEmbedder,
) -> VectorRetriever:
    """创建向量检索服务（便捷函数）.

    Args:
        settings: 应用配置
        qdrant_client: Qdrant 客户端
        embedder: Embedding 服务

    Returns:
        VectorRetriever: 检索服务实例

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> from pikee.infrastructure.llm import get_embedder
        >>> from pikee.infrastructure.database import get_qdrant_client
        >>>
        >>> settings = get_settings()
        >>> embedder = get_embedder(settings, provider="fastembed")
        >>> qdrant_client = get_qdrant_client(settings)
        >>> retriever = create_vector_retriever(settings, qdrant_client, embedder)
    """
    return VectorRetriever(settings, qdrant_client, embedder)

