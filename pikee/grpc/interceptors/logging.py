"""日志拦截器.

记录 gRPC 请求和响应日志。
"""

import logging
import time
from typing import Any, Callable

import grpc

logger = logging.getLogger(__name__)


class LoggingInterceptor(grpc.ServerInterceptor):
    """日志拦截器.

    记录每个 gRPC 请求的详细信息。

    Examples:
        >>> interceptor = LoggingInterceptor()
        >>> server = grpc.server(..., interceptors=[interceptor])
    """

    def __init__(self) -> None:
        """初始化拦截器."""
        logger.info("LoggingInterceptor 初始化完成")

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """拦截服务调用.

        Args:
            continuation: 继续调用的回调
            handler_call_details: 调用详情

        Returns:
            RPC 方法处理器
        """
        method = handler_call_details.method
        start_time = time.time()

        logger.info(f"gRPC 请求开始: {method}")

        try:
            result = continuation(handler_call_details)
            elapsed_time = time.time() - start_time
            logger.info(f"gRPC 请求完成: {method}, 耗时: {elapsed_time:.3f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"gRPC 请求失败: {method}, 耗时: {elapsed_time:.3f}s, 错误: {e}")
            raise

