"""FastAPI 应用入口.

启动 HTTP API 服务。
"""

from pathlib import Path

from fastapi import FastAPI

from pikee.infrastructure.config.settings import get_settings
from pikee.infrastructure.utils.logger import setup_logger

# 初始化配置
settings = get_settings()

# 配置根日志记录器（应用启动时执行一次）
logger = setup_logger(
    name="pikee",  # 根日志记录器名称
    settings=settings,
    log_file=Path("logs/pikee.log") if not settings.debug else None,  # 生产环境写文件
)

logger.info(f"应用启动: {settings.app_name} v{settings.app_version}")
logger.info(f"调试模式: {settings.debug}")
logger.info(f"日志级别: {settings.log_level}")

# 创建 FastAPI 应用
app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)


@app.get("/")
async def root() -> dict:
    """健康检查接口.

    Returns:
        应用状态信息
    """
    logger.info("健康检查请求")
    return {"app": settings.app_name, "version": settings.app_version, "status": "healthy"}


@app.on_event("startup")
async def startup_event() -> None:
    """应用启动事件."""
    logger.info("FastAPI 应用启动完成")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """应用关闭事件."""
    logger.info("FastAPI 应用关闭")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "pikee.main:app", host="0.0.0.0", port=8000, reload=settings.debug, log_level=settings.log_level.lower()
    )
