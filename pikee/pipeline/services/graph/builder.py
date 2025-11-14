"""知识图谱构建服务.

将 Document、Chunk、Atom 构建为 Neo4j 知识图谱。
"""

import logging
from typing import List

from pikee.infrastructure.config.settings import Settings
from pikee.infrastructure.database.neo4j import Neo4jClient
from pikee.pipeline.models.atom import Atom
from pikee.pipeline.models.chunk import Chunk
from pikee.pipeline.models.document import Document

logger = logging.getLogger(__name__)


class GraphBuilder:
    """知识图谱构建器.

    将 Document、Chunk、Atom 构建为 Neo4j 知识图谱，形成以下结构：
    (:Document)-[:CONTAINS]->(:Chunk)-[:HAS_ATOM]->(:Atom)

    Examples:
        >>> settings = get_settings()
        >>> neo4j_client = get_neo4j_client(settings)
        >>> builder = GraphBuilder(settings, neo4j_client)
        >>> builder.build_graph(document, chunks, atoms)
        >>> neo4j_client.close()
    """

    def __init__(self, settings: Settings, neo4j_client: Neo4jClient) -> None:
        """初始化图谱构建器.

        Args:
            settings: 应用配置
            neo4j_client: Neo4j 客户端
        """
        self.settings = settings
        self.neo4j_client = neo4j_client

        logger.info("GraphBuilder 初始化完成")

    def build_graph(self, document: Document, chunks: List[Chunk], atoms: List[Atom]) -> bool:
        """构建完整的知识图谱.

        Args:
            document: 文档对象
            chunks: Chunk 列表
            atoms: Atom 列表

        Returns:
            是否构建成功
        """
        logger.info(f"开始构建知识图谱: document={document.id}, chunks={len(chunks)}, atoms={len(atoms)}")

        try:
            # 1. 创建 Document 节点
            success = self.neo4j_client.create_document(
                document_id=document.id,
                title=document.title,
                file_path=document.file_path,
                file_type=document.file_type,
                metadata=document.metadata,
            )

            if not success:
                logger.error("创建 Document 节点失败")
                return False

            # 2. 批量创建 Chunk 节点
            chunk_data = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "summary": chunk.summary,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]

            chunks_created = self.neo4j_client.create_chunks_batch(chunk_data)
            if chunks_created != len(chunks):
                logger.warning(f"部分 Chunk 创建失败: {chunks_created}/{len(chunks)}")

            # 3. 批量创建 Atom 节点
            atom_data = [
                {
                    "atom_id": atom.id,
                    "chunk_id": atom.chunk_id,
                    "question": atom.question,
                    "answer": atom.answer,
                    "metadata": atom.metadata,
                }
                for atom in atoms
            ]

            atoms_created = self.neo4j_client.create_atoms_batch(atom_data)
            if atoms_created != len(atoms):
                logger.warning(f"部分 Atom 创建失败: {atoms_created}/{len(atoms)}")

            logger.info(
                f"知识图谱构建完成: {document.id} "
                f"(Chunks: {chunks_created}/{len(chunks)}, "
                f"Atoms: {atoms_created}/{len(atoms)})"
            )
            return True

        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}", exc_info=True)
            return False

    def build_document_graph(self, document: Document, chunks: List[Chunk]) -> bool:
        """仅构建 Document 和 Chunks 的图谱（不包含 Atoms）.

        Args:
            document: 文档对象
            chunks: Chunk 列表

        Returns:
            是否构建成功
        """
        logger.info(f"开始构建文档图谱: document={document.id}, chunks={len(chunks)}")

        try:
            # 创建 Document 节点
            success = self.neo4j_client.create_document(
                document_id=document.id,
                title=document.title,
                file_path=document.file_path,
                file_type=document.file_type,
                metadata=document.metadata,
            )

            if not success:
                return False

            # 批量创建 Chunk 节点
            chunk_data = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "summary": chunk.summary,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]

            chunks_created = self.neo4j_client.create_chunks_batch(chunk_data)

            logger.info(f"文档图谱构建完成: {document.id} (Chunks: {chunks_created}/{len(chunks)})")
            return chunks_created > 0

        except Exception as e:
            logger.error(f"文档图谱构建失败: {e}", exc_info=True)
            return False


def build_knowledge_graph(settings: Settings, document: Document, chunks: List[Chunk], atoms: List[Atom]) -> bool:
    """构建知识图谱（便捷函数）.

    Args:
        settings: 应用配置
        document: 文档对象
        chunks: Chunk 列表
        atoms: Atom 列表

    Returns:
        是否构建成功

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> success = build_knowledge_graph(settings, document, chunks, atoms)
        >>> print(f"构建{'成功' if success else '失败'}")
    """
    from pikee.infrastructure.database.neo4j import get_neo4j_client

    neo4j_client = get_neo4j_client(settings)
    builder = GraphBuilder(settings, neo4j_client)

    try:
        return builder.build_graph(document, chunks, atoms)
    finally:
        neo4j_client.close()
