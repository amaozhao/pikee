"""gRPC 接口层.

提供 gRPC 服务接口，支持高性能 RPC 调用。
"""

from pikee.grpc.server import create_grpc_server

__all__ = ["create_grpc_server"]

