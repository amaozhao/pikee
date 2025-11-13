"""配置管理模块.

提供基于 Apollo 配置中心的两层配置加载机制。

使用示例:
    ```python
    from pikee.infrastructure.config import get_settings

    # 获取配置
    settings = get_settings()

    # 使用配置
    print(settings.openai_api_key)
    print(settings.neo4j_uri)
    ```

动态刷新配置:
    ```python
    from pikee.infrastructure.config import reload_settings

    # 重新加载配置（适用于管理接口）
    settings = reload_settings()
    ```
"""

from pikee.infrastructure.config.settings import (
    ConfigManager,
    LocalSettings,
    Settings,
    get_config_manager,
    get_settings,
    reload_settings,
)

__all__ = ["Settings", "LocalSettings", "ConfigManager", "get_settings", "get_config_manager", "reload_settings"]
