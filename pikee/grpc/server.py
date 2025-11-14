"""gRPC 服务器.

提供 gRPC 服务器的创建和启动功能。
"""

import logging
from concurrent import futures

import grpc

from pikee.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


def create_grpc_server(settings: Settings, max_workers: int = 10) -> grpc.Server:
    """创建 gRPC 服务器.

    Args:
        settings: 应用配置
        max_workers: 最大工作线程数

    Returns:
        gRPC 服务器实例

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> server = create_grpc_server(settings)
        >>> server.add_insecure_port("[::]:50051")
        >>> server.start()
        >>> server.wait_for_termination()
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # TODO: 添加服务注册
    # add_DocumentServiceServicer_to_server(DocumentService(), server)
    # add_QueryServiceServicer_to_server(QueryService(), server)

    logger.info(f"gRPC 服务器创建完成，最大工作线程数：{max_workers}")

    return server


async def serve_async(settings: Settings, port: int = 50051, max_workers: int = 10) -> None:
    """异步启动 gRPC 服务器.

    Args:
        settings: 应用配置
        port: 监听端口
        max_workers: 最大工作线程数

    Examples:
        >>> import asyncio
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> asyncio.run(serve_async(settings))
    """
    server = grpc.aio.server()

    # TODO: 添加异步服务注册
    # add_DocumentServiceServicer_to_server(AsyncDocumentService(), server)

    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"gRPC 服务器启动，监听端口：{port}")

    await server.start()
    await server.wait_for_termination()


def serve(settings: Settings, port: int = 50051, max_workers: int = 10) -> None:
    """同步启动 gRPC 服务器.

    Args:
        settings: 应用配置
        port: 监听端口
        max_workers: 最大工作线程数

    Examples:
        >>> from pikee.infrastructure.config.settings import get_settings
        >>> settings = get_settings()
        >>> serve(settings, port=50051)
    """
    server = create_grpc_server(settings, max_workers=max_workers)
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"gRPC 服务器启动，监听端口：{port}")

    server.start()
    server.wait_for_termination()
