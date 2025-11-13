# Apollo 配置管理详解

> 基于 Apollo + pydantic-settings 的集中配置方案

---

## 1. 配置架构

### 1.1 配置分层

```
┌─────────────────────────────────────┐
│   本地 .env（仅 Apollo 连接信息）    │
│   • APOLLO_META_SERVER              │
│   • APOLLO_APP_ID                   │
│   • APOLLO_SECRET                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│     Apollo Config Service            │
│   • 所有业务配置                     │
│   • 环境隔离（dev/test/prod）        │
│   • 配置版本管理                     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   应用启动时加载 + 实时监听变更      │
└─────────────────────────────────────┘
```

---

## 2. 依赖安装

```toml
[tool.poetry.dependencies]
pydantic = "^2.10.0"
pydantic-settings = "^2.6.0"
pyapollo = "^1.1.0"
```

---

## 3. 配置实现

### 3.1 本地配置类 (LocalSettings)

```python
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class LocalSettings(BaseSettings):
    """本地环境变量（仅 Apollo 连接）"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Apollo 配置
    apollo_app_id: str = "pike-rag"
    apollo_cluster: str = "default"
    apollo_meta_server: str = "http://localhost:8080"
    apollo_namespace: str = "application"
    apollo_secret: str | None = None
    
    # 环境
    env: str = "dev"  # dev, test, prod
    
    # 本地开发模式（跳过 Apollo）
    local_dev_mode: bool = False
```

### 3.2 Apollo 客户端封装

```python
from pyapollo import ApolloClient
from typing import Any

class ApolloConfig:
    """Apollo 配置客户端"""
    
    def __init__(self, local_settings: LocalSettings):
        self.local_settings = local_settings
        self.client: ApolloClient | None = None
        self._cache: dict[str, Any] = {}
        
        if not local_settings.local_dev_mode:
            self._init_client()
    
    def _init_client(self):
        """初始化 Apollo"""
        self.client = ApolloClient(
            app_id=self.local_settings.apollo_app_id,
            cluster=self.local_settings.apollo_cluster,
            config_server_url=self.local_settings.apollo_meta_server,
            secret=self.local_settings.apollo_secret
        )
        self.client.start()  # 启动配置监听
        logger.info("Apollo client started")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        if self.client:
            value = self.client.get_value(
                key, 
                default, 
                namespace=self.local_settings.apollo_namespace
            )
            self._cache[key] = value
            return value
        else:
            # 本地开发模式，从环境变量读取
            return os.getenv(key.upper(), default)
```

### 3.3 应用配置类 (Settings)

```python
class Settings(BaseSettings):
    """应用配置（从 Apollo 加载）"""
    
    # 基础
    app_name: str = "PIKE-RAG v2"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4-turbo"
    openai_temperature: float = 0.7
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_chunks: str = "pike_chunks"
    qdrant_collection_atoms: str = "pike_atoms"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # 重试配置
    max_retry_attempts: int = 3
    retry_backoff_base: float = 2.0
    retry_max_wait: int = 60
    
    @classmethod
    def from_apollo(cls, apollo: ApolloConfig):
        """从 Apollo 加载所有配置"""
        config_keys = [
            "openai_api_key",
            "openai_model",
            "openai_temperature",
            "qdrant_url",
            "qdrant_collection_chunks",
            "neo4j_uri",
            "neo4j_password",
            "redis_url",
            "max_retry_attempts",
            # ... 其他配置项
        ]
        
        config_dict = {}
        for key in config_keys:
            value = apollo.get(key)
            if value is not None:
                config_dict[key] = value
        
        return cls(**config_dict)
```

### 3.4 全局初始化

```python
# 初始化配置
local_settings = LocalSettings()
apollo_config = ApolloConfig(local_settings)
settings = Settings.from_apollo(apollo_config)

# 动态刷新配置
def reload_config():
    """重新加载配置"""
    global settings
    settings = Settings.from_apollo(apollo_config)
```

---

## 4. 环境变量文件

### 4.1 .env（本地）

```bash
# Apollo 配置中心连接信息
APOLLO_APP_ID=pike-rag
APOLLO_CLUSTER=default
APOLLO_META_SERVER=http://config.example.com:8080
APOLLO_NAMESPACE=application
APOLLO_SECRET=your-secret-key

# 环境标识
ENV=dev

# 本地开发模式（设为 true 时不连接 Apollo）
LOCAL_DEV_MODE=false
```

### 4.2 Apollo 配置中心（application namespace）

```yaml
# LLM 配置
openai_api_key: "sk-xxxxx"
openai_api_base: "https://api.openai.com/v1"
openai_model: "gpt-4-turbo"
openai_temperature: 0.7

# Qdrant 配置
qdrant_url: "http://qdrant-service:6333"
qdrant_collection_chunks: "pike_chunks"
qdrant_collection_atoms: "pike_atoms"

# Neo4j 配置
neo4j_uri: "bolt://neo4j-service:7687"
neo4j_user: "neo4j"
neo4j_password: "prod_password_here"
neo4j_database: "neo4j"

# Redis 配置
redis_url: "redis://redis-service:6379"

# 重试配置
max_retry_attempts: 3
retry_backoff_base: 2.0
retry_max_wait: 60

# 监控配置
enable_sentry: true
sentry_dsn: "https://xxx@sentry.io/xxx"
```

---

## 5. 使用示例

### 5.1 在代码中使用配置

```python
from app.config import settings

# 直接使用
llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=settings.openai_temperature
)

# Qdrant 连接
client = QdrantClient(url=settings.qdrant_url)
```

### 5.2 动态刷新配置（可选）

```python
from app.config import reload_config

@app.post("/admin/reload-config")
async def admin_reload():
    """管理员接口：重新加载配置"""
    reload_config()
    return {"message": "Configuration reloaded"}
```

---

## 6. 不同环境配置

### 6.1 开发环境

```bash
# .env.dev
APOLLO_CLUSTER=dev
ENV=dev
LOCAL_DEV_MODE=true  # 本地开发不连 Apollo
```

### 6.2 测试环境

```bash
# .env.test
APOLLO_CLUSTER=test
APOLLO_META_SERVER=http://apollo-test.example.com:8080
ENV=test
```

### 6.3 生产环境

```bash
# .env.prod
APOLLO_CLUSTER=prod
APOLLO_META_SERVER=http://apollo-prod.example.com:8080
APOLLO_SECRET=prod-secret-key
ENV=prod
```

---

## 7. 优势总结

| 优势 | 说明 |
|------|------|
| **集中管理** | 所有配置在 Apollo 统一维护 |
| **环境隔离** | dev/test/prod 配置独立 |
| **动态刷新** | 配置变更无需重启应用 |
| **安全** | 敏感信息不存本地代码 |
| **审计** | Apollo 记录配置变更历史 |
| **降级** | Apollo 不可用时使用缓存 |

---

## 8. 故障排查

### 8.1 Apollo 连接失败

**现象**：应用启动时提示无法连接 Apollo

**解决**：
1. 检查 `APOLLO_META_SERVER` 地址是否正确
2. 检查网络连通性
3. 设置 `LOCAL_DEV_MODE=true` 临时使用本地配置

### 8.2 配置项读取失败

**现象**：某个配置项为 None

**解决**：
1. 检查 Apollo 配置中心是否配置该项
2. 检查 namespace 是否正确
3. 查看应用日志中的配置加载信息

---

**版本**: 1.0  
**更新时间**: 2025年

