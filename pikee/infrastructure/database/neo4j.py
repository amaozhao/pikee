"""Neo4j 图数据库客户端封装.

提供知识图谱的基本操作。
"""

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

from pikee.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j 客户端封装.

    提供知识图谱的基本操作，包括节点创建、关系构建和查询。

    Examples:
        >>> settings = get_settings()
        >>> client = Neo4jClient(settings)
        >>> client.create_document(document_id="doc1", title="测试文档")
        >>> client.close()
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 Neo4j 客户端.

        Args:
            settings: 应用配置
        """
        self.settings = settings
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database

        # 创建驱动
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

        logger.info(f"Neo4jClient 初始化完成: uri={self.uri}, database={self.database}")

    def close(self) -> None:
        """关闭连接."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j 连接已关闭")

    def _execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 Cypher 查询.

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        parameters = parameters or {}

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except ServiceUnavailable as e:
            logger.error(f"Neo4j 服务不可用: {e}")
            raise
        except Exception as e:
            logger.error(f"执行查询失败: {query}, 错误: {e}")
            raise

    # ========== Document 操作 ==========

    def create_document(
        self,
        document_id: str,
        title: str,
        file_path: str = "",
        file_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """创建文档节点.

        Args:
            document_id: 文档ID
            title: 文档标题
            file_path: 文件路径
            file_type: 文件类型
            metadata: 其他元数据

        Returns:
            是否创建成功
        """
        metadata = metadata or {}

        query = """
        CREATE (d:Document {
            document_id: $document_id,
            title: $title,
            file_path: $file_path,
            file_type: $file_type,
            created_at: datetime(),
            metadata: $metadata
        })
        RETURN d
        """

        try:
            self._execute_query(
                query,
                {
                    "document_id": document_id,
                    "title": title,
                    "file_path": file_path,
                    "file_type": file_type,
                    "metadata": metadata,
                },
            )
            logger.info(f"创建文档节点成功: {document_id}")
            return True
        except Exception as e:
            logger.error(f"创建文档节点失败: {e}")
            return False

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档节点.

        Args:
            document_id: 文档ID

        Returns:
            文档数据，如果不存在返回 None
        """
        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d
        """

        try:
            result = self._execute_query(query, {"document_id": document_id})
            return result[0].get("d") if result else None
        except Exception as e:
            logger.error(f"获取文档节点失败: {e}")
            return None

    def delete_document(self, document_id: str) -> bool:
        """删除文档及其所有关联节点.

        Args:
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        query = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_ATOM]->(a:Atom)
        DETACH DELETE d, c, a
        """

        try:
            self._execute_query(query, {"document_id": document_id})
            logger.info(f"删除文档节点成功: {document_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档节点失败: {e}")
            return False

    # ========== Chunk 操作 ==========

    def create_chunk(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        summary: str = "",
        index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """创建 Chunk 节点并关联到 Document.

        Args:
            chunk_id: Chunk ID
            document_id: 所属文档ID
            content: Chunk 内容
            summary: 摘要
            index: 索引
            metadata: 其他元数据

        Returns:
            是否创建成功
        """
        metadata = metadata or {}

        query = """
        // 创建 Chunk 节点
        CREATE (c:Chunk {
            chunk_id: $chunk_id,
            document_id: $document_id,
            content: $content,
            summary: $summary,
            index: $index,
            created_at: datetime(),
            metadata: $metadata
        })

        // 关联到 Document
        WITH c
        MATCH (d:Document {document_id: $document_id})
        CREATE (d)-[:CONTAINS]->(c)

        RETURN c
        """

        try:
            self._execute_query(
                query,
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content": content,
                    "summary": summary,
                    "index": index,
                    "metadata": metadata,
                },
            )
            logger.debug(f"创建 Chunk 节点成功: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"创建 Chunk 节点失败: {e}")
            return False

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取 Chunk 节点.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk 数据，如果不存在返回 None
        """
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        RETURN c
        """

        try:
            result = self._execute_query(query, {"chunk_id": chunk_id})
            return result[0].get("c") if result else None
        except Exception as e:
            logger.error(f"获取 Chunk 节点失败: {e}")
            return None

    # ========== Atom 操作 ==========

    def create_atom(
        self,
        atom_id: str,
        chunk_id: str,
        question: str,
        answer: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """创建 Atom 节点并关联到 Chunk.

        Args:
            atom_id: Atom ID
            chunk_id: 来源 Chunk ID
            question: 原子问题
            answer: 答案
            metadata: 其他元数据

        Returns:
            是否创建成功
        """
        metadata = metadata or {}

        query = """
        // 创建 Atom 节点
        CREATE (a:Atom {
            atom_id: $atom_id,
            chunk_id: $chunk_id,
            question: $question,
            answer: $answer,
            created_at: datetime(),
            metadata: $metadata
        })

        // 关联到 Chunk
        WITH a
        MATCH (c:Chunk {chunk_id: $chunk_id})
        CREATE (c)-[:HAS_ATOM]->(a)

        RETURN a
        """

        try:
            self._execute_query(
                query,
                {
                    "atom_id": atom_id,
                    "chunk_id": chunk_id,
                    "question": question,
                    "answer": answer,
                    "metadata": metadata,
                },
            )
            logger.debug(f"创建 Atom 节点成功: {atom_id}")
            return True
        except Exception as e:
            logger.error(f"创建 Atom 节点失败: {e}")
            return False

    def get_atom(self, atom_id: str) -> Optional[Dict[str, Any]]:
        """获取 Atom 节点.

        Args:
            atom_id: Atom ID

        Returns:
            Atom 数据，如果不存在返回 None
        """
        query = """
        MATCH (a:Atom {atom_id: $atom_id})
        RETURN a
        """

        try:
            result = self._execute_query(query, {"atom_id": atom_id})
            return result[0].get("a") if result else None
        except Exception as e:
            logger.error(f"获取 Atom 节点失败: {e}")
            return None

    # ========== 批量操作 ==========

    def create_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """批量创建 Chunk 节点.

        Args:
            chunks: Chunk 数据列表
            batch_size: 批处理大小

        Returns:
            成功创建的数量
        """
        total = len(chunks)
        success_count = 0

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            query = """
            UNWIND $chunks AS chunk
            CREATE (c:Chunk {
                chunk_id: chunk.chunk_id,
                document_id: chunk.document_id,
                content: chunk.content,
                summary: chunk.summary,
                index: chunk.index,
                created_at: datetime(),
                metadata: chunk.metadata
            })

            WITH c, chunk
            MATCH (d:Document {document_id: chunk.document_id})
            CREATE (d)-[:CONTAINS]->(c)

            RETURN count(c) as created
            """

            try:
                result = self._execute_query(query, {"chunks": batch})
                if result:
                    success_count += result[0].get("created", 0)
                logger.debug(f"批量创建 Chunk: {i + len(batch)}/{total}")
            except Exception as e:
                logger.error(f"批量创建 Chunk 失败: {e}")

        logger.info(f"批量创建 Chunk 完成: {success_count}/{total}")
        return success_count

    def create_atoms_batch(
        self,
        atoms: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """批量创建 Atom 节点.

        Args:
            atoms: Atom 数据列表
            batch_size: 批处理大小

        Returns:
            成功创建的数量
        """
        total = len(atoms)
        success_count = 0

        for i in range(0, total, batch_size):
            batch = atoms[i : i + batch_size]

            query = """
            UNWIND $atoms AS atom
            CREATE (a:Atom {
                atom_id: atom.atom_id,
                chunk_id: atom.chunk_id,
                question: atom.question,
                answer: atom.answer,
                created_at: datetime(),
                metadata: atom.metadata
            })

            WITH a, atom
            MATCH (c:Chunk {chunk_id: atom.chunk_id})
            CREATE (c)-[:HAS_ATOM]->(a)

            RETURN count(a) as created
            """

            try:
                result = self._execute_query(query, {"atoms": batch})
                if result:
                    success_count += result[0].get("created", 0)
                logger.debug(f"批量创建 Atom: {i + len(batch)}/{total}")
            except Exception as e:
                logger.error(f"批量创建 Atom 失败: {e}")

        logger.info(f"批量创建 Atom 完成: {success_count}/{total}")
        return success_count

    # ========== 查询操作 ==========

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """获取文档的所有 Chunks.

        Args:
            document_id: 文档ID

        Returns:
            Chunk 列表
        """
        query = """
        MATCH (d:Document {document_id: $document_id})-[:CONTAINS]->(c:Chunk)
        RETURN c
        ORDER BY c.index
        """

        try:
            return self._execute_query(query, {"document_id": document_id})
        except Exception as e:
            logger.error(f"获取文档 Chunks 失败: {e}")
            return []

    def get_atoms_by_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """获取 Chunk 的所有 Atoms.

        Args:
            chunk_id: Chunk ID

        Returns:
            Atom 列表
        """
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:HAS_ATOM]->(a:Atom)
        RETURN a
        """

        try:
            return self._execute_query(query, {"chunk_id": chunk_id})
        except Exception as e:
            logger.error(f"获取 Chunk Atoms 失败: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, int]:
        """获取图谱统计信息.

        Returns:
            统计信息字典
        """
        query = """
        MATCH (d:Document) WITH count(d) as documents
        MATCH (c:Chunk) WITH documents, count(c) as chunks
        MATCH (a:Atom) WITH documents, chunks, count(a) as atoms
        RETURN documents, chunks, atoms
        """

        try:
            result = self._execute_query(query)
            if result:
                return result[0]
            return {"documents": 0, "chunks": 0, "atoms": 0}
        except Exception as e:
            logger.error(f"获取图谱统计信息失败: {e}")
            return {"documents": 0, "chunks": 0, "atoms": 0}


def get_neo4j_client(settings: Settings) -> Neo4jClient:
    """获取 Neo4j 客户端（便捷函数）.

    Args:
        settings: 应用配置

    Returns:
        Neo4j 客户端实例

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> client = get_neo4j_client(settings)
        >>> # 使用完毕后记得关闭
        >>> client.close()
    """
    return Neo4jClient(settings)

