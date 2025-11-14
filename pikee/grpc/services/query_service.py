"""查询 gRPC 服务.

实现查询相关的 gRPC 接口。
"""

import logging
from typing import Any

# import grpc
# from pikee.grpc.generated import query_pb2, query_pb2_grpc
from pikee.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class QueryService:
    """查询服务实现.

    实现 QueryService 的 gRPC 接口。

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> service = QueryService(settings)
    """

    def __init__(self, settings: Settings) -> None:
        """初始化服务.

        Args:
            settings: 应用配置
        """
        self.settings = settings
        logger.info("QueryService 初始化完成")

    # TODO: 实现 gRPC 服务方法
    # def Search(self, request, context):
    #     """搜索查询."""
    #     pass

    # def Answer(self, request, context):
    #     """问答查询."""
    #     pass

