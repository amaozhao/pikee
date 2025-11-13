# 配置管理模块

基于 Apollo 配置中心和 pydantic-settings 的两层配置加载机制。

## 架构设计

```
┌─────────────────────────────────────┐
│   本地 .env（Apollo 连接信息）       │
│   • APOLLO_META_SERVER_ADDRESS      │
│   • APOLLO_APP_ID                   │
│   • APOLLO_CLUSTER                  │
│   • LOCAL_DEV_MODE                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│     Apollo 配置中心                  │
│   • openai_api_key                  │
│   • neo4j_password                  │
│   • 所有业务配置                     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   应用配置（Settings）               │
│   • 类型安全                        │
│   • 自动验证                        │
│   • 便捷访问                        │
└─────────────────────────────────────┘
```

## 快速开始

### 1. 创建 .env 文件

```bash
# .env
APOLLO_META_SERVER_ADDRESS=http://localhost:8080
APOLLO_APP_ID=pike-rag
APOLLO_CLUSTER=default
APOLLO_ENV=DEV
APOLLO_NAMESPACES=application

# 本地开发模式（不连接 Apollo）
LOCAL_DEV_MODE=false
```

### 2. 使用配置

```python
from pikee.infrastructure.config import get_settings

# 获取配置（自动初始化）
settings = get_settings()

# 使用配置
print(settings.openai_api_key)
print(settings.neo4j_uri)
print(settings.qdrant_url)
```

### 3. 在 FastAPI 中使用

```python
from fastapi import FastAPI, Depends
from pikee.infrastructure.config import get_settings, Settings

app = FastAPI()

@app.get("/config/status")
async def config_status(settings: Settings = Depends(get_settings)):
    """获取配置状态"""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
    }

@app.post("/admin/reload-config")
async def reload_config():
    """重新加载配置（管理接口）"""
    from pikee.infrastructure.config import reload_settings
    
    settings = reload_settings()
    return {"message": "配置已重新加载"}
```

## 配置项说明

### 基础配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `app_name` | str | "PIKE-RAG v2" | 应用名称 |
| `app_version` | str | "2.0.0" | 应用版本 |
| `debug` | bool | False | 调试模式 |
| `log_level` | str | "INFO" | 日志级别 |

### LLM 配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `openai_api_key` | str | **必填** | OpenAI API Key |
| `openai_api_base` | str | "https://api.openai.com/v1" | API Base URL |
| `openai_model` | str | "gpt-4-turbo" | 模型名称 |
| `openai_temperature` | float | 0.7 | 温度参数 (0.0-2.0) |
| `openai_max_tokens` | int | 4096 | 最大 token 数 |

### 数据库配置

#### Qdrant (向量数据库)

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `qdrant_url` | str | "http://localhost:6333" | 服务地址 |
| `qdrant_collection_chunks` | str | "pike_chunks" | Chunk 集合名 |
| `qdrant_collection_atoms` | str | "pike_atoms" | Atom 集合名 |
| `qdrant_vector_size` | int | 1536 | 向量维度 |

#### Neo4j (图数据库)

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `neo4j_uri` | str | "bolt://localhost:7687" | 连接 URI |
| `neo4j_user` | str | "neo4j" | 用户名 |
| `neo4j_password` | str | **必填** | 密码 |
| `neo4j_database` | str | "neo4j" | 数据库名 |

#### Redis (缓存)

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `redis_url` | str | "redis://localhost:6379" | 连接 URL |

### 重试配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `max_retry_attempts` | int | 3 | 最大重试次数 |
| `retry_backoff_base` | float | 2.0 | 退避基数 |
| `retry_max_wait` | int | 60 | 最大等待时间（秒） |

### 文档处理配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `chunk_size` | int | 1000 | 切分大小 |
| `chunk_overlap` | int | 200 | 重叠大小 |

### 监控配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `enable_sentry` | bool | False | 启用 Sentry |
| `sentry_dsn` | str | None | Sentry DSN |

## 使用模式

### 模式 1: 生产环境（Apollo）

```bash
# .env
APOLLO_META_SERVER_ADDRESS=http://apollo.example.com:8080
APOLLO_APP_ID=pike-rag
APOLLO_CLUSTER=prod
LOCAL_DEV_MODE=false
```

配置从 Apollo 配置中心加载。

### 模式 2: 本地开发（环境变量）

```bash
# .env
LOCAL_DEV_MODE=true

# 直接配置业务参数
OPENAI_API_KEY=sk-xxx
NEO4J_PASSWORD=password
```

不连接 Apollo，从环境变量读取。

## 高级用法

### 自定义配置管理器

```python
from pikee.infrastructure.config import ConfigManager

# 创建自定义管理器
manager = ConfigManager()

# 初始化配置
settings = manager.initialize()

# 重新加载
settings = manager.reload()
```

### 访问 Apollo 客户端

```python
from pikee.infrastructure.config import get_config_manager

manager = get_config_manager()

# 访问 Apollo 客户端
if manager.apollo_client:
    value = manager.apollo_client.get_value("custom_key")
```

## 故障排查

### Q: Apollo 连接失败

**A:** 检查以下项：
1. `APOLLO_META_SERVER_ADDRESS` 是否正确
2. 网络连通性
3. 临时使用 `LOCAL_DEV_MODE=true` 跳过 Apollo

### Q: 配置项读取为 None

**A:** 检查以下项：
1. Apollo 配置中心是否配置该项
2. 配置项名称是否正确（不区分大小写）
3. 查看日志中的配置加载信息

### Q: 必填配置项缺失

**A:** 确保以下配置已设置：
- `openai_api_key`
- `neo4j_password`

## 最佳实践

1. **生产环境**：使用 Apollo 配置中心
2. **开发环境**：使用 `LOCAL_DEV_MODE=true`
3. **敏感信息**：不要提交到代码仓库
4. **配置变更**：通过 Apollo 管理，无需重启应用
5. **日志记录**：启用 DEBUG 日志查看配置加载过程

## 相关文档

- [Apollo配置管理详解](../../../docs/Apollo配置管理详解.md)
- [PIKE-RAG生产级实现方案](../../../docs/PIKE-RAG生产级实现方案.md)

