"""Qdrant 向量数据库客户端封装.

提供向量存储的 CRUD 操作接口。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.http import models

from pikee.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class QdrantClient:
    """Qdrant 客户端封装.

    提供向量数据库的基本操作。

    特点:
        - 连接管理
        - Collection CRUD
        - 向量插入/检索
        - 错误处理

    Examples:
        >>> settings = get_settings()
        >>> client = QdrantClient(settings)
        >>> client.create_collection("test_collection", vector_size=384)
        >>> client.insert_vectors("test_collection", vectors, payloads, ids)
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """初始化 Qdrant 客户端.

        Args:
            settings: 应用配置
            **kwargs: 传递给 QdrantClient 的额外参数
        """
        self.settings = settings
        self.url = settings.qdrant_url

        # 初始化 Qdrant 客户端
        self._client = QdrantClientBase(url=self.url, **kwargs)

        logger.info(f"QdrantClient 初始化完成: url={self.url}")

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine", on_disk_payload: bool = True
    ) -> bool:
        """创建 Collection.

        Args:
            collection_name: Collection 名称
            vector_size: 向量维度
            distance: 距离度量 ("Cosine", "Euclid", "Dot")
            on_disk_payload: 是否将 payload 存储在磁盘上

        Returns:
            bool: 是否创建成功
        """
        try:
            # 检查是否已存在
            if self.collection_exists(collection_name):
                logger.warning(f"Collection {collection_name} 已存在，跳过创建")
                return True

            # 创建 Collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=getattr(models.Distance, distance.upper())
                ),
                on_disk_payload=on_disk_payload,
            )

            logger.info(f"Collection {collection_name} 创建成功: vector_size={vector_size}, distance={distance}")
            return True

        except Exception as e:
            logger.error(f"创建 Collection {collection_name} 失败: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """检查 Collection 是否存在.

        Args:
            collection_name: Collection 名称

        Returns:
            bool: 是否存在
        """
        try:
            collections = self._client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"检查 Collection {collection_name} 失败: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """删除 Collection.

        Args:
            collection_name: Collection 名称

        Returns:
            bool: 是否删除成功
        """
        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} 删除成功")
            return True
        except Exception as e:
            logger.error(f"删除 Collection {collection_name} 失败: {e}")
            return False

    def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> bool:
        """批量插入向量.

        Args:
            collection_name: Collection 名称
            vectors: 向量列表
            payloads: Payload 列表（元数据）
            ids: ID 列表（可选，不提供则自动生成）
            batch_size: 批处理大小

        Returns:
            bool: 是否插入成功

        Raises:
            ValueError: 当向量和 payload 数量不匹配时
        """
        if len(vectors) != len(payloads):
            raise ValueError(f"向量数量 ({len(vectors)}) 与 payload 数量 ({len(payloads)}) 不匹配")

        if ids and len(ids) != len(vectors):
            raise ValueError(f"ID 数量 ({len(ids)}) 与向量数量 ({len(vectors)}) 不匹配")

        try:
            total = len(vectors)
            logger.info(f"开始插入 {total} 个向量到 {collection_name}, batch_size={batch_size}")

            # 分批插入
            for i in range(0, total, batch_size):
                end = min(i + batch_size, total)
                batch_vectors = vectors[i:end]
                batch_payloads = payloads[i:end]
                batch_ids = ids[i:end] if ids else None

                # 构造 points
                points = [
                    models.PointStruct(
                        id=batch_ids[j] if batch_ids else None, vector=batch_vectors[j], payload=batch_payloads[j]
                    )
                    for j in range(len(batch_vectors))
                ]

                # 上传
                self._client.upsert(collection_name=collection_name, points=points)

                logger.debug(f"已插入 {end}/{total} 个向量")

            logger.info(f"成功插入 {total} 个向量到 {collection_name}")
            return True

        except Exception as e:
            logger.error(f"插入向量到 {collection_name} 失败: {e}", exc_info=True)
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """检索相似向量.

        Args:
            collection_name: Collection 名称
            query_vector: 查询向量
            limit: 返回结果数量
            score_threshold: 分数阈值（可选）
            filter_conditions: 过滤条件（可选）

        Returns:
            List[Tuple[str, float, Dict]]: (ID, 分数, payload) 列表
        """
        try:
            # 构建过滤器
            query_filter = None
            if filter_conditions:
                # 简单的 must 过滤器
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(key=k, match=models.MatchValue(value=v))
                        for k, v in filter_conditions.items()
                    ]
                )

            # 搜索
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            # 转换结果
            return [(str(r.id), r.score, r.payload) for r in results]

        except Exception as e:
            logger.error(f"检索向量失败: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """获取 Collection 信息.

        Args:
            collection_name: Collection 名称

        Returns:
            Optional[Dict]: Collection 信息
        """
        try:
            info = self._client.get_collection(collection_name=collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name,
                },
            }
        except Exception as e:
            logger.error(f"获取 Collection {collection_name} 信息失败: {e}")
            return None

    def count_vectors(self, collection_name: str) -> int:
        """统计 Collection 中的向量数量.

        Args:
            collection_name: Collection 名称

        Returns:
            int: 向量数量
        """
        try:
            info = self._client.get_collection(collection_name=collection_name)
            return info.vectors_count or 0
        except Exception as e:
            logger.error(f"统计 Collection {collection_name} 向量数量失败: {e}")
            return 0

    def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """删除指定的向量.

        Args:
            collection_name: Collection 名称
            ids: 要删除的向量 ID 列表

        Returns:
            bool: 是否删除成功
        """
        try:
            self._client.delete(collection_name=collection_name, points_selector=models.PointIdsList(points=ids))
            logger.info(f"成功从 {collection_name} 删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def close(self) -> None:
        """关闭客户端连接."""
        try:
            self._client.close()
            logger.info("QdrantClient 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 QdrantClient 连接失败: {e}")


def get_qdrant_client(settings: Settings, **kwargs: Any) -> QdrantClient:
    """获取 Qdrant 客户端实例（便捷函数）.

    Args:
        settings: 应用配置
        **kwargs: 传递给 QdrantClient 的额外参数

    Returns:
        QdrantClient: Qdrant 客户端实例

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> client = get_qdrant_client(settings)
        >>> client.create_collection("test", vector_size=384)
    """
    return QdrantClient(settings, **kwargs)
