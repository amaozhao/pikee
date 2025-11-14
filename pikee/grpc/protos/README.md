# Protocol Buffer 定义

本目录存放所有 `.proto` 文件定义。

## 目录结构

```
protos/
├── common.proto          # 通用类型定义
├── document.proto        # 文档处理服务定义
└── query.proto          # 查询服务定义
```

## 生成代码

使用以下命令生成 Python 代码：

```bash
# 生成 Python gRPC 代码
python -m grpc_tools.protoc \
    -I. \
    --python_out=../generated \
    --grpc_python_out=../generated \
    *.proto
```

## Proto 文件说明

### common.proto
定义通用数据类型：
- Metadata：元数据
- Status：状态码
- Pagination：分页参数

### document.proto
定义文档处理相关服务：
- DocumentService：文档处理服务
  - ProcessDocument：处理文档
  - GetDocumentStatus：查询处理状态

### query.proto
定义查询相关服务：
- QueryService：查询服务
  - Search：搜索查询
  - Answer：问答查询

