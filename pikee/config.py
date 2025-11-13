"""全局配置模块（向后兼容）.

为了方便导入，提供顶层的配置访问接口。

推荐使用方式:
    ```python
    from pikee.config import get_settings

    settings = get_settings()
    ```

或者:
    ```python
    from pikee.infrastructure.config import get_settings

    settings = get_settings()
    ```
"""

from pikee.infrastructure.config import (
    ConfigManager,
    LocalSettings,
    Settings,
    get_config_manager,
    get_settings,
    reload_settings,
)

__all__ = [
    "Settings",
    "LocalSettings",
    "ConfigManager",
    "get_settings",
    "get_config_manager",
    "reload_settings",
]

