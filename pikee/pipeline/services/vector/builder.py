"""向量数据库构建服务.

将 Chunks 和 Atoms 构建为双向量数据库。
"""

import logging
from typing import List, Optional

from tqdm import tqdm

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database import get_qdrant_client
from pikee.infrastructure.database.qdrant import QdrantClient
from pikee.infrastructure.llm.embedder import BaseEmbedder
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk

logger = logging.getLogger(__name__)


class VectorDatabaseBuilder:
    """向量数据库构建器.

    将 Chunks 和 Atoms 转换为向量并存储到 Qdrant。

    工作流程:
        1. 创建两个 Collection（chunks 和 atoms）
        2. 批量生成 Chunk 向量
        3. 批量生成 Atom 向量
        4. 插入到 Qdrant

    Examples:
        >>> builder = VectorDatabaseBuilder(settings, qdrant_client, embedder)
        >>> builder.build_chunk_store(chunks)
        >>> builder.build_atom_store(atoms)
    """

    def __init__(
        self, settings: Settings, qdrant_client: QdrantClient, embedder: BaseEmbedder, batch_size: Optional[int] = None
    ) -> None:
        """初始化向量数据库构建器.

        Args:
            settings: 应用配置
            qdrant_client: Qdrant 客户端
            embedder: Embedding 服务
            batch_size: 批处理大小（考虑512M内存限制）
        """
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedder = embedder

        # 根据512M内存限制计算批处理大小
        if batch_size is None:
            # 假设每个向量1536维（4字节/float），约6KB
            # 加上文本内容，保守估计每个item 10KB
            # 512MB / 10KB ≈ 50,000，但为了安全使用更小的值
            self.batch_size = 50
        else:
            self.batch_size = batch_size

        self.chunk_collection = settings.qdrant_collection_chunks
        self.atom_collection = settings.qdrant_collection_atoms

        logger.info(f"VectorDatabaseBuilder 初始化完成: batch_size={self.batch_size}, vector_dim={embedder.dimension}")

    def build_chunk_store(
        self, chunks: List[Chunk], recreate_collection: bool = False, show_progress: bool = True
    ) -> bool:
        """构建 Chunk 向量存储.

        Args:
            chunks: Chunk 列表
            recreate_collection: 是否重新创建 Collection
            show_progress: 是否显示进度条

        Returns:
            bool: 是否构建成功
        """
        if not chunks:
            logger.warning("Chunks 列表为空，跳过构建")
            return False

        logger.info(f"开始构建 Chunk 向量存储: {len(chunks)} 个 chunks")

        try:
            # 1. 创建或重建 Collection
            if recreate_collection and self.qdrant_client.collection_exists(self.chunk_collection):
                logger.info(f"删除现有 Collection: {self.chunk_collection}")
                self.qdrant_client.delete_collection(self.chunk_collection)

            if not self.qdrant_client.collection_exists(self.chunk_collection):
                logger.info(f"创建 Collection: {self.chunk_collection}")
                self.qdrant_client.create_collection(
                    collection_name=self.chunk_collection, vector_size=self.embedder.dimension, distance="Cosine"
                )

            # 2. 分批处理
            total = len(chunks)
            successful = 0

            for i in tqdm(range(0, total, self.batch_size), desc="构建 Chunk 向量", disable=not show_progress):
                end = min(i + self.batch_size, total)
                batch_chunks = chunks[i:end]

                # 生成向量
                texts = [chunk.content for chunk in batch_chunks]
                vectors = self.embedder.embed_texts(texts)

                # 构建 payloads
                payloads = []
                ids = []
                for chunk in batch_chunks:
                    payload = {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "summary": chunk.summary,
                        "title": chunk.metadata.get("title", ""),
                        "source": chunk.metadata.get("source", ""),
                        "index": chunk.index,
                    }
                    payloads.append(payload)
                    ids.append(chunk.id)

                # 插入
                success = self.qdrant_client.insert_vectors(
                    collection_name=self.chunk_collection,
                    vectors=vectors,
                    payloads=payloads,
                    ids=ids,
                    batch_size=self.batch_size,
                )

                if success:
                    successful += len(batch_chunks)

            logger.info(f"Chunk 向量存储构建完成: {successful}/{total} 成功")
            return successful == total

        except Exception as e:
            logger.error(f"构建 Chunk 向量存储失败: {e}", exc_info=True)
            return False

    def build_atom_store(
        self, atoms: List[Atom], recreate_collection: bool = False, show_progress: bool = True
    ) -> bool:
        """构建 Atom 向量存储.

        Args:
            atoms: Atom 列表
            recreate_collection: 是否重新创建 Collection
            show_progress: 是否显示进度条

        Returns:
            bool: 是否构建成功
        """
        if not atoms:
            logger.warning("Atoms 列表为空，跳过构建")
            return False

        logger.info(f"开始构建 Atom 向量存储: {len(atoms)} 个 atoms")

        try:
            # 1. 创建或重建 Collection
            if recreate_collection and self.qdrant_client.collection_exists(self.atom_collection):
                logger.info(f"删除现有 Collection: {self.atom_collection}")
                self.qdrant_client.delete_collection(self.atom_collection)

            if not self.qdrant_client.collection_exists(self.atom_collection):
                logger.info(f"创建 Collection: {self.atom_collection}")
                self.qdrant_client.create_collection(
                    collection_name=self.atom_collection, vector_size=self.embedder.dimension, distance="Cosine"
                )

            # 2. 分批处理
            total = len(atoms)
            successful = 0

            for i in tqdm(range(0, total, self.batch_size), desc="构建 Atom 向量", disable=not show_progress):
                end = min(i + self.batch_size, total)
                batch_atoms = atoms[i:end]

                # 生成向量（使用问题）
                texts = [atom.question for atom in batch_atoms]
                vectors = self.embedder.embed_texts(texts)

                # 构建 payloads
                payloads = []
                ids = []
                for atom in batch_atoms:
                    payload = {
                        "atom_id": atom.id,
                        "question": atom.question,
                        "answer": atom.answer,
                        "source_chunk_id": atom.chunk_id,  # 关键！指向源 Chunk
                        "document_id": atom.metadata.get("document_id", ""),
                        "title": atom.metadata.get("title", ""),
                    }
                    payloads.append(payload)
                    ids.append(atom.id)

                # 插入
                success = self.qdrant_client.insert_vectors(
                    collection_name=self.atom_collection,
                    vectors=vectors,
                    payloads=payloads,
                    ids=ids,
                    batch_size=self.batch_size,
                )

                if success:
                    successful += len(batch_atoms)

            logger.info(f"Atom 向量存储构建完成: {successful}/{total} 成功")
            return successful == total

        except Exception as e:
            logger.error(f"构建 Atom 向量存储失败: {e}", exc_info=True)
            return False

    def build_all(
        self, chunks: List[Chunk], atoms: List[Atom], recreate_collections: bool = False, show_progress: bool = True
    ) -> bool:
        """一次性构建 Chunk 和 Atom 向量存储.

        Args:
            chunks: Chunk 列表
            atoms: Atom 列表
            recreate_collections: 是否重新创建 Collections
            show_progress: 是否显示进度条

        Returns:
            bool: 是否全部构建成功
        """
        logger.info("开始构建双向量数据库")

        chunk_success = self.build_chunk_store(
            chunks=chunks, recreate_collection=recreate_collections, show_progress=show_progress
        )

        atom_success = self.build_atom_store(
            atoms=atoms, recreate_collection=recreate_collections, show_progress=show_progress
        )

        if chunk_success and atom_success:
            logger.info("双向量数据库构建成功")
            # 输出统计信息
            self._print_statistics()
            return True
        else:
            logger.error("双向量数据库构建失败")
            return False

    def _print_statistics(self) -> None:
        """打印统计信息."""
        chunk_info = self.qdrant_client.get_collection_info(self.chunk_collection)
        atom_info = self.qdrant_client.get_collection_info(self.atom_collection)

        if chunk_info:
            logger.info(f"Chunk Collection: {chunk_info['vectors_count']} 个向量")
        if atom_info:
            logger.info(f"Atom Collection: {atom_info['vectors_count']} 个向量")


def build_vector_database(
    settings: Settings, chunks: List[Chunk], atoms: List[Atom], embedder: BaseEmbedder, recreate: bool = False
) -> bool:
    """构建向量数据库（便捷函数）.

    Args:
        settings: 应用配置
        chunks: Chunk 列表
        atoms: Atom 列表
        embedder: Embedding 服务
        recreate: 是否重新创建

    Returns:
        bool: 是否构建成功

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> from pikee.infrastructure.llm import get_embedder
        >>> from pikee.infrastructure.database import get_qdrant_client
        >>>
        >>> settings = get_settings()
        >>> embedder = get_embedder(settings, provider="fastembed")
        >>> qdrant_client = get_qdrant_client(settings)
        >>>
        >>> builder = VectorDatabaseBuilder(settings, qdrant_client, embedder)
        >>> builder.build_all(chunks, atoms, recreate_collections=True)
    """
    qdrant_client = get_qdrant_client(settings)
    builder = VectorDatabaseBuilder(settings, qdrant_client, embedder)

    return builder.build_all(chunks, atoms, recreate_collections=recreate)
