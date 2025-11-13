# 知识图谱在 PIKE-RAG 中的应用详解

## 📚 目录
- [1. 概述](#1-概述)
- [2. PIKE-RAG 中的知识图谱概念](#2-pike-rag-中的知识图谱概念)
- [3. 异构知识图谱的数据结构](#3-异构知识图谱的数据结构)
- [4. 知识图谱的构建流程](#4-知识图谱的构建流程)
- [5. 知识图谱的检索机制](#5-知识图谱的检索机制)
- [6. 数据流转详解](#6-数据流转详解)
- [7. 代码实现解析](#7-代码实现解析)
- [8. 配置示例](#8-配置示例)
- [9. 总结与最佳实践](#9-总结与最佳实践)

---

## 1. 概述

### 1.1 什么是 PIKE-RAG 的知识图谱？

PIKE-RAG 采用了一种创新的**异构知识图谱（Heterogeneous Knowledge Graph）**设计，与传统的实体-关系图谱不同，它通过**双层知识表示结构**来组织和检索信息：

```
传统知识图谱:                    PIKE-RAG 异构知识图谱:
                                 
实体 --关系--> 实体               Chunk (粗粒度) ←→ Atom (细粒度)
   \                                  |                    |
    \--关系--> 实体                  完整文档片段         原子问题
                                     |                    |
                                  向量存储              向量存储
                                  Chunk Store          Atom Store
```

### 1.2 核心优势

1. **多粒度检索**: 细粒度检索（Atom）+ 粗粒度返回（Chunk）
2. **语义对齐**: 原子问题（Atom）与用户问题在语义上天然对齐
3. **上下文完整**: 最终返回完整的 Chunk，保证上下文信息不丢失
4. **支持多跳推理**: 通过 Atom 之间的关联实现知识链接
5. **可解释性强**: 每个检索结果都可追溯到具体的原子问题

---

## 2. PIKE-RAG 中的知识图谱概念

### 2.1 异构知识图谱的两个层次

#### 层次 1: Chunk 层（文档块层）

```
┌─────────────────────────────────────────────┐
│ Chunk (文档块)                              │
├─────────────────────────────────────────────┤
│ • 完整的文档片段（500-1000字）              │
│ • 包含丰富的上下文信息                      │
│ • 用于最终提供给 LLM 的内容                 │
│ • 每个 Chunk 有唯一 ID                      │
└─────────────────────────────────────────────┘

示例 Chunk:
{
  "chunk_id": "chunk_001",
  "title": "2020年奥斯卡最佳影片",
  "content": "《寄生虫》(Parasite)是由韩国导演奉俊昊执导的2019年黑色喜剧惊悚片。
             该片在2020年第92届奥斯卡金像奖上获得最佳影片、最佳导演、最佳原创剧本和
             最佳国际影片四项大奖。奉俊昊导演1969年出生于韩国大邱市..."
}
```

#### 层次 2: Atom 层（原子问题层）

```
┌─────────────────────────────────────────────┐
│ Atom (原子问题)                             │
├─────────────────────────────────────────────┤
│ • 从 Chunk 中提取的细粒度知识点             │
│ • 以"问题"形式表示                          │
│ • 每个 Atom 关联到源 Chunk ID               │
│ • 用于精确的语义检索                        │
└─────────────────────────────────────────────┘

从上述 Chunk 提取的 Atoms:
1. "2020年奥斯卡最佳影片是哪部电影？"
2. "《寄生虫》的导演是谁？"
3. "奉俊昊导演出生在哪里？"
4. "奉俊昊导演出生在哪一年？"
5. "《寄生虫》在奥斯卡获得了哪些奖项？"
```

### 2.2 知识图谱的图结构

虽然表面上是双层向量存储，但实际形成了一个图结构：

```
                    知识图谱结构视图

                [用户问题: "奉俊昊出生地?"]
                           ↓
                    【检索 Atom Store】
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   [Atom 3]           [Atom 7]           [Atom 12]
"奉俊昊出生在哪？"  "奉俊昊国籍？"    "奉俊昊成长经历？"
   (score: 0.95)      (score: 0.82)      (score: 0.78)
        ↓                  ↓                  ↓
   source_chunk_id    source_chunk_id    source_chunk_id
        ↓                  ↓                  ↓
   [Chunk 001] ←─────[Chunk 001]       [Chunk 005]
        ↓
   【返回完整 Chunk】
   "《寄生虫》...奉俊昊1969年出生于韩国大邱市..."
```

这种结构中：
- **节点**: Chunk 和 Atom 是两类不同的节点
- **边**: Atom 到 Chunk 的 `source_chunk_id` 关系
- **隐式关联**: 属于同一 Chunk 的 Atoms 之间有隐式关联

---

## 3. 异构知识图谱的数据结构

### 3.1 核心数据类

#### AtomRetrievalInfo - 原子检索信息

```python
@dataclass
class AtomRetrievalInfo:
    atom_query: str              # 用于检索的查询
    atom: str                    # 检索到的原子问题
    source_chunk_title: str      # 源文档块的标题
    source_chunk: str            # 源文档块的完整内容
    source_chunk_id: str         # 源文档块的ID (关键关联字段)
    retrieval_score: float       # 检索相似度分数
    atom_embedding: List[float]  # 原子问题的向量表示
```

**关键字段说明**:
- `source_chunk_id`: 这是 Atom 和 Chunk 之间的**核心关联字段**，形成图谱的"边"
- `atom`: 原子问题本身，是知识图谱中的细粒度"节点"
- `source_chunk`: 完整文档内容，是知识图谱中的粗粒度"节点"

### 3.2 双向量存储架构

```
┌──────────────────────────────────────────────────────────┐
│              ChunkAtomRetriever                          │
│                                                          │
│  ┌────────────────────┐      ┌─────────────────────┐   │
│  │  _chunk_store      │      │  _atom_store        │   │
│  │  (Chroma)          │      │  (Chroma)           │   │
│  ├────────────────────┤      ├─────────────────────┤   │
│  │ Document:          │      │ Document:           │   │
│  │ - page_content:    │      │ - page_content:     │   │
│  │     完整 Chunk 内容│      │     原子问题文本    │   │
│  │ - metadata:        │◄─────┤ - metadata:         │   │
│  │   - id             │ 关联 │   - source_chunk_id │   │
│  │   - title          │      │   - title           │   │
│  │   - atom_questions │      │                     │   │
│  └────────────────────┘      └─────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**存储说明**:

1. **_chunk_store (Chunk 向量存储)**
   ```python
   Document(
       page_content="完整的文档片段内容...",
       metadata={
           "id": "chunk_001",
           "title": "文档标题",
           "atom_questions_str": "问题1\n问题2\n问题3"
       }
   )
   ```

2. **_atom_store (Atom 向量存储)**
   ```python
   Document(
       page_content="奉俊昊导演出生在哪里？",  # 单个原子问题
       metadata={
           "source_chunk_id": "chunk_001",     # 指向源 Chunk
           "title": "2020年奥斯卡最佳影片"
       }
   )
   ```

### 3.3 NetworkX 图增强（可选）

虽然当前实现主要使用向量存储，但 PIKE-RAG 保留了 NetworkX 的图遍历能力：

```python
class NetworkxMixin:
    def _get_subgraph_by_entity(
        self, 
        graph: nx.Graph, 
        entities: Iterable, 
        neighbor_layer: int = 1
    ) -> nx.Graph:
        """
        根据给定的实体提取子图，包含指定跳数内的邻居节点
        
        Args:
            graph: 完整的知识图谱
            entities: 起始实体集合
            neighbor_layer: 扩展的邻居层数
        
        Returns:
            过滤后的子图
        """
```

**使用场景**: 
- 多跳推理: 从一个 Atom 出发，遍历其相关的 Chunk 和其他 Atoms
- 实体关联分析: 找到与特定实体相关的所有知识节点

---

## 4. 知识图谱的构建流程

完整的知识图谱构建需要经过三个主要阶段：

```
原始文档
    ↓
┌─────────────────────────────────────────┐
│ 阶段 1: 文档切分 (Chunking)            │
│ 工具: pikerag/workflows/chunking.py    │
└─────────────────────────────────────────┘
    ↓
文档块 (Chunks)
    ↓
┌─────────────────────────────────────────┐
│ 阶段 2: 原子问题提取 (Tagging)         │
│ 工具: pikerag/workflows/tagging.py     │
└─────────────────────────────────────────┘
    ↓
带原子问题的文档块 (Tagged Chunks)
    ↓
┌─────────────────────────────────────────┐
│ 阶段 3: 向量数据库构建                 │
│ 工具: ChunkAtomRetriever 自动加载      │
└─────────────────────────────────────────┘
    ↓
异构知识图谱 (Chunk Store + Atom Store)
```

### 4.1 阶段 1: 文档切分 (Chunking)

#### 目的
将大型文档切分成合适大小的 Chunk，保持语义完整性

#### 执行命令
```bash
python examples/chunking.py examples/biology/configs/chunking.yml
```

#### 核心代码流程

**文件**: `pikerag/workflows/chunking.py`

```python
class ChunkingWorkflow:
    def __init__(self, yaml_config: dict):
        # 1. 初始化 LLM 客户端
        self._init_llm_client()
        
        # 2. 初始化分割器
        self._init_splitter()  # 通常使用 LLMPoweredRecursiveSplitter
        
    def run(self):
        for doc_name, input_path, output_path in self._file_infos:
            # 3. 加载原始文档
            doc_loader = get_loader(file_path=input_path)
            docs = doc_loader.load()
            
            # 4. 执行文档切分
            chunk_docs = self._splitter.transform_documents(docs)
            
            # 5. 保存切分结果到 pickle 文件
            with open(output_path, "wb") as fout:
                pickle.dump(chunk_docs, fout)
```

#### 输出示例

**输出文件**: `data/biology/chunks.pkl`

```python
[
    Document(
        page_content="细胞是生物体结构和功能的基本单位。真核细胞包含细胞核、
                     细胞质和细胞膜等结构。细胞核是遗传信息的储存中心...",
        metadata={
            "filename": "biology_textbook.pdf",
            "page": 15
        }
    ),
    Document(
        page_content="线粒体被称为细胞的能量工厂，负责进行细胞呼吸产生ATP。
                     线粒体具有双层膜结构，外膜光滑，内膜向内折叠形成嵴...",
        metadata={
            "filename": "biology_textbook.pdf",
            "page": 16
        }
    ),
    # ... 更多 chunks
]
```

#### 配置示例

**文件**: `examples/biology/configs/chunking.yml`

```yaml
splitter:
  module_path: pikerag.document_transformers.splitter
  class_name: LLMPoweredRecursiveSplitter
  args:
    chunk_size: 1000        # 每个 chunk 的目标大小
    chunk_overlap: 200      # chunk 之间的重叠字符数
    separators:             # 分隔符优先级
      - "\n\n"
      - "\n"
      - "。"
      - "！"
      - "？"
```

---

### 4.2 阶段 2: 原子问题提取 (Tagging)

#### 目的
使用 LLM 从每个 Chunk 中提取多个可回答的原子问题（Atoms）

#### 执行命令
```bash
python examples/tagging.py examples/hotpotqa/configs/tagging.yml
```

#### 核心代码流程

**文件**: `pikerag/workflows/tagging.py`

```python
class TaggingWorkflow:
    def __init__(self, yaml_config: dict):
        # 1. 初始化 LLM 客户端
        self._init_llm_client()
        
        # 2. 初始化 Tagger
        self._tagger = LLMPoweredTagger(
            llm_client=self._client,
            tagging_protocol=self._tagging_protocol,  # 定义如何提取问题
            tag_name="atom_questions"
        )
        
    def run(self):
        # 3. 加载切分后的文档
        docs = self._load_func(**self._load_args)  
        # 例如: load_chunks_from_jsonl()
        
        # 4. 对每个 Chunk 提取原子问题
        tagged_docs = self._tagger.transform_documents(docs)
        
        # 5. 保存带标签的文档
        self._save_func(tagged_docs, **self._save_args)
        # 例如: save_chunks_to_jsonl()
```

**文件**: `pikerag/document_transformers/tagger/llm_powered_tagger.py`

```python
class LLMPoweredTagger:
    def _get_tags_info(self, content: str, **metadata) -> List[str]:
        # 1. 使用协议构建 Prompt
        messages = self._tagging_protocol.process_input(
            content=content, 
            **metadata
        )
        
        # 2. 调用 LLM 生成原子问题
        response = self._llm_client.generate_content_with_messages(
            messages=messages,
            **self._llm_config
        )
        
        # 3. 解析 LLM 输出为问题列表
        return self._tagging_protocol.parse_output(
            content=response, 
            **metadata
        )
```

#### Prompt 模板

**文件**: `pikerag/prompts/tagging/atom_question_tagging.py`

```python
atom_question_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at content "
                   "understanding and asking question."),
        ("user", """
# Task
Your task is to extract as many questions as possible that are 
relevant and can be answered by the given content. Please try to 
be diverse and avoid extracting duplicated or similar questions. 
Make sure your question contain necessary entity names and avoid 
to use pronouns like it, he, she, they, the company, the person etc.

# Output Format
Output your answers line by line, with each question on a new line, 
without itemized symbols or numbers.

# Content
{content}

# Output:
""".strip()),
    ]
)
```

#### LLM 交互示例

**输入到 LLM**:
```
Content: 《寄生虫》(Parasite)是由韩国导演奉俊昊执导的2019年黑色喜剧惊悚片。
该片在2020年第92届奥斯卡金像奖上获得最佳影片、最佳导演、最佳原创剧本和
最佳国际影片四项大奖。奉俊昊导演1969年出生于韩国大邱市。
```

**LLM 输出**:
```
《寄生虫》的导演是谁？
《寄生虫》是哪一年的电影？
《寄生虫》在2020年奥斯卡获得了哪些奖项？
《寄生虫》获得了多少个奥斯卡奖项？
奉俊昊导演出生在哪里？
奉俊昊导演出生在哪一年？
2020年奥斯卡最佳影片是哪部电影？
2020年奥斯卡最佳导演是谁？
```

#### 输出示例

**输出文件**: `data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl`

```json
{
  "chunk_id": "chunk_001",
  "title": "2020年奥斯卡最佳影片",
  "content": "《寄生虫》(Parasite)是由韩国导演奉俊昊执导的2019年黑色喜剧惊悚片...",
  "atom_questions": [
    "《寄生虫》的导演是谁？",
    "《寄生虫》是哪一年的电影？",
    "《寄生虫》在2020年奥斯卡获得了哪些奖项？",
    "奉俊昊导演出生在哪里？",
    "奉俊昊导演出生在哪一年？"
  ]
}
```

#### 配置示例

**文件**: `examples/hotpotqa/configs/tagging.yml`

```yaml
# 输入文档加载
ori_doc_loading:
  module: pikerag.utils.data_protocol_utils
  name: load_chunks_from_jsonl
  args:
    jsonl_chunk_path: data/hotpotqa/dev_500_retrieval_contexts_as_chunks.jsonl

# 输出文档保存
tagged_doc_saving:
  module: pikerag.utils.data_protocol_utils
  name: save_chunks_to_jsonl
  args:
    dump_path: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl

# Tagger 设置
tagger:
  tagging_protocol:
    module_path: pikerag.prompts.tagging
    attr_name: atom_question_tagging_protocol
  tag_name: atom_questions

# LLM 设置
llm_client:
  module_path: pikerag.llm_client
  class_name: AzureOpenAIClient
  llm_config:
    model: gpt-4
    temperature: 0.7
```

---

### 4.3 阶段 3: 向量数据库构建

#### 目的
自动加载 Tagged Chunks，构建双向量存储（Chunk Store + Atom Store）

#### 特点
这个阶段**不需要单独执行**，在 QA 工作流初始化时自动完成

#### 核心代码流程

**文件**: `pikerag/knowledge_retrievers/chunk_atom_retriever.py`

```python
class ChunkAtomRetriever:
    def _load_vector_store(self):
        # 1. 加载带原子问题的 Chunks
        doc_ids, docs = load_ids_and_chunks(
            filepath="data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl",
            atom_tag="atom_questions"
        )
        
        # 2. 构建 Chunk 向量存储
        self._chunk_store: Chroma = load_vector_store(
            collection_name="hotpotqa_chunk",
            persist_directory="data/vector_stores/hotpotqa",
            embedding=self.embedding_func,
            documents=docs,         # List[Document]
            ids=doc_ids,           # List[str]
            exist_ok=True
        )
        
        # 3. 加载 Atoms（从同一个 JSONL 文件）
        atom_ids, atoms = load_ids_and_atoms(
            filepath="data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl",
            atom_tag="atom_questions"
        )
        
        # 4. 构建 Atom 向量存储
        self._atom_store: Chroma = load_vector_store(
            collection_name="hotpotqa_atom",
            persist_directory="data/vector_stores/hotpotqa",
            embedding=self.embedding_func,
            documents=atoms,        # List[Document]
            ids=atom_ids,          # None (自动生成)
            exist_ok=True
        )
```

#### 数据加载函数详解

**文件**: `pikerag/utils/data_protocol_utils.py`

**函数 1: load_ids_and_chunks**

```python
def load_ids_and_chunks(
    filepath: str, 
    atom_tag: str = "atom_questions"
) -> Tuple[List[str], List[Document]]:
    """
    从 JSONL 文件加载 Chunks，构建 Chunk Store
    
    Returns:
        chunk_ids: ['chunk_001', 'chunk_002', ...]
        chunk_docs: [Document(...), Document(...), ...]
    """
    chunk_ids: List[str] = []
    chunk_docs: List[Document] = []
    
    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            chunk_ids.append(chunk_dict["chunk_id"])
            
            chunk_docs.append(
                Document(
                    page_content=chunk_dict["content"],
                    metadata={
                        "id": chunk_dict["chunk_id"],
                        "title": chunk_dict["title"],
                        # 将 atom_questions 列表转为字符串保存
                        "atom_questions_str": "\n".join(chunk_dict[atom_tag])
                    }
                )
            )
    
    return chunk_ids, chunk_docs
```

**函数 2: load_ids_and_atoms**

```python
def load_ids_and_atoms(
    filepath: str, 
    atom_tag: str
) -> Tuple[None, List[Document]]:
    """
    从 JSONL 文件加载 Atoms，构建 Atom Store
    
    Returns:
        None: atom_ids 自动生成
        atom_docs: [Document(...), Document(...), ...]
    """
    atom_docs: List[Document] = []
    
    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            # 遍历每个 chunk 的所有原子问题
            for atom in chunk_dict[atom_tag]:
                atom = atom.strip()
                if len(atom) > 0:
                    atom_docs.append(
                        Document(
                            page_content=atom,  # 单个原子问题
                            metadata={
                                # 关键: 记录源 Chunk ID
                                "source_chunk_id": chunk_dict["chunk_id"],
                                "title": chunk_dict["title"]
                            }
                        )
                    )
    
    return None, atom_docs
```

#### 向量化与存储

```python
# Chroma 会自动对每个 Document 的 page_content 进行向量化
embedding_func = AzureOpenAIEmbedding()  # 例如使用 text-embedding-ada-002

# Chunk Store 中存储的向量
chunk_vector = embedding_func.embed_query(chunk.page_content)
# 向量维度: 例如 1536 (ada-002)

# Atom Store 中存储的向量
atom_vector = embedding_func.embed_query(atom.page_content)
# 向量维度: 例如 1536 (ada-002)
```

#### 最终数据库结构

```
data/vector_stores/hotpotqa/
├── chroma.sqlite3                          # Chroma 数据库文件
└── collections/
    ├── hotpotqa_chunk/                     # Chunk 集合
    │   ├── vectors.bin                     # Chunk 向量
    │   └── metadata.json                   # Chunk 元数据
    └── hotpotqa_atom/                      # Atom 集合
        ├── vectors.bin                     # Atom 向量
        └── metadata.json                   # Atom 元数据
```

#### 数据统计示例

假设处理 HotpotQA dev_500 数据集：

```
原始问题数: 500
检索上下文段落数: 5,000
↓ [Chunking]
Chunks 数量: 5,000
↓ [Tagging]
Atoms 总数: 25,000 (平均每个 Chunk 5 个 Atom)
↓ [Vector Store]
Chunk Store: 5,000 个向量
Atom Store: 25,000 个向量
```

---

## 5. 知识图谱的检索机制

ChunkAtomRetriever 提供了三种检索方法，对应不同的检索策略：

```
用户问题
    ↓
┌──────────────────────────────────────────────────┐
│          ChunkAtomRetriever 检索方法             │
├──────────────────────────────────────────────────┤
│ 方法 1: retrieve_atom_info_through_atom()       │
│   → 通过 Atom Store 检索                        │
│                                                  │
│ 方法 2: retrieve_atom_info_through_chunk()      │
│   → 通过 Chunk Store 检索，返回最佳 Atom        │
│                                                  │
│ 方法 3: retrieve_contents_by_query()            │
│   → 综合检索（方法 1 + 方法 2）                 │
└──────────────────────────────────────────────────┘
    ↓
返回相关 Chunks
```

### 5.1 方法 1: retrieve_atom_info_through_atom()

#### 原理
直接在 Atom Store 中进行向量检索，找到最相关的原子问题，然后通过 `source_chunk_id` 获取源 Chunk

#### 流程图

```
用户问题: "奉俊昊导演出生在哪里？"
    ↓
【步骤 1】向量化问题
    query_embedding = embed("奉俊昊导演出生在哪里？")
    ↓
【步骤 2】在 Atom Store 中检索 Top-K 最相似的 Atoms
    ↓
    Atom 1: "奉俊昊导演出生在哪里？" (score: 0.95, source_chunk_id: chunk_001)
    Atom 2: "奉俊昊导演的国籍是什么？" (score: 0.82, source_chunk_id: chunk_001)
    Atom 3: "奉俊昊导演的成长经历？" (score: 0.78, source_chunk_id: chunk_005)
    ↓
【步骤 3】提取所有唯一的 source_chunk_id
    unique_chunk_ids = ["chunk_001", "chunk_005"]
    ↓
【步骤 4】从 Chunk Store 批量获取 Chunks
    chunks = _chunk_store.get(ids=unique_chunk_ids)
    ↓
【步骤 5】组装 AtomRetrievalInfo 列表
    [
        AtomRetrievalInfo(
            atom_query="奉俊昊导演出生在哪里？",
            atom="奉俊昊导演出生在哪里？",
            source_chunk_id="chunk_001",
            source_chunk="《寄生虫》...奉俊昊1969年出生于韩国大邱市...",
            retrieval_score=0.95
        ),
        ...
    ]
```

#### 代码实现

**文件**: `pikerag/knowledge_retrievers/chunk_atom_retriever.py`

```python
def retrieve_atom_info_through_atom(
    self, 
    queries: Union[List[str], str], 
    retrieve_id: str = "",
    **kwargs
) -> List[AtomRetrievalInfo]:
    """
    通过 Atom Store 检索
    
    Args:
        queries: 单个或多个查询问题
        retrieve_id: 检索标识符（用于日志）
        **kwargs: 可选参数，如 retrieve_k
    
    Returns:
        List[AtomRetrievalInfo]: 检索结果列表
    """
    # 1. 决定 retrieve_k（每个查询返回多少个结果）
    if "retrieve_k" in kwargs:
        retrieve_k = kwargs["retrieve_k"]
    elif isinstance(queries, list) and len(queries) > 1:
        retrieve_k = self.atom_retrieve_k  # 多查询时用较小的 k
    else:
        retrieve_k = self.retrieve_k       # 单查询时用标准 k
    
    # 2. 确保 queries 是列表
    if isinstance(queries, str):
        queries = [queries]
    
    # 3. 对每个 query 在 Atom Store 中检索
    query_atom_score_tuples: List[Tuple[str, Document, float]] = []
    for atom_query in queries:
        for atom_doc, score in self._get_doc_with_query(
            atom_query, 
            self._atom_store,  # 在 Atom Store 中检索
            retrieve_k
        ):
            query_atom_score_tuples.append((atom_query, atom_doc, score))
    
    # 4. 转换为 AtomRetrievalInfo 对象
    return self._atom_info_tuple_to_class(query_atom_score_tuples)


def _atom_info_tuple_to_class(
    self, 
    atom_retrieval_info: List[Tuple[str, Document, float]]
) -> List[AtomRetrievalInfo]:
    """
    将检索结果转换为 AtomRetrievalInfo 对象
    
    核心逻辑:
    1. 提取所有唯一的 source_chunk_id
    2. 批量从 Chunk Store 获取 Chunks
    3. 组装完整的检索信息
    """
    # 1. 提取唯一的 source_chunk_id
    source_chunk_ids: List[str] = list(set([
        doc.metadata["source_chunk_id"] 
        for _, doc, _ in atom_retrieval_info
    ]))
    
    # 2. 批量获取 Chunks（关键: 图谱的边遍历）
    chunk_doc_results: Dict[str, Any] = self._chunk_store.get(
        ids=source_chunk_ids
    )
    
    # 3. 构建 chunk_id -> chunk_content 映射
    chunk_id_to_content = {
        chunk_id: chunk_str
        for chunk_id, chunk_str in zip(
            chunk_doc_results["ids"], 
            chunk_doc_results["documents"]
        )
    }
    
    # 4. 组装 AtomRetrievalInfo
    retrieval_infos: List[AtomRetrievalInfo] = []
    for atom_query, atom_doc, score in atom_retrieval_info:
        source_chunk_id = atom_doc.metadata["source_chunk_id"]
        retrieval_infos.append(
            AtomRetrievalInfo(
                atom_query=atom_query,
                atom=atom_doc.page_content,
                source_chunk_title=atom_doc.metadata.get("title", None),
                source_chunk=chunk_id_to_content[source_chunk_id],  # 获取完整 Chunk
                source_chunk_id=source_chunk_id,
                retrieval_score=score,
                atom_embedding=self.embedding_func.embed_query(atom_doc.page_content)
            )
        )
    
    return retrieval_infos
```

#### 使用示例

```python
retriever = ChunkAtomRetriever(...)

# 单查询检索
results = retriever.retrieve_atom_info_through_atom(
    queries="奉俊昊导演出生在哪里？",
    retrieve_id="Q001"
)

# 多查询检索（用于问题分解场景）
results = retriever.retrieve_atom_info_through_atom(
    queries=[
        "奉俊昊导演出生在哪里？",
        "2020年奥斯卡最佳影片是什么？",
        "《寄生虫》获得了哪些奖项？"
    ],
    retrieve_id="Q001_decomposed"
)

# 访问结果
for info in results:
    print(f"Query: {info.atom_query}")
    print(f"Matched Atom: {info.atom}")
    print(f"Score: {info.retrieval_score}")
    print(f"Source Chunk: {info.source_chunk[:100]}...")
```

---

### 5.2 方法 2: retrieve_atom_info_through_chunk()

#### 原理
在 Chunk Store 中进行向量检索，找到最相关的 Chunks，然后为每个 Chunk 找到与问题最匹配的一个 Atom

#### 流程图

```
用户问题: "奉俊昊导演出生在哪里？"
    ↓
【步骤 1】向量化问题
    query_embedding = embed("奉俊昊导演出生在哪里？")
    ↓
【步骤 2】在 Chunk Store 中检索 Top-K 最相似的 Chunks
    ↓
    Chunk 1: "《寄生虫》...奉俊昊1969年出生于韩国大邱市..." (score: 0.88)
      metadata: {
        "atom_questions_str": "《寄生虫》的导演是谁？\n奉俊昊出生在哪里？\n..."
      }
    Chunk 2: "奉俊昊的早期作品..." (score: 0.75)
      metadata: {
        "atom_questions_str": "奉俊昊的第一部电影？\n奉俊昊的代表作？\n..."
      }
    ↓
【步骤 3】对每个 Chunk，计算问题与其所有 Atoms 的相似度
    Chunk 1 的 Atoms:
      - "《寄生虫》的导演是谁？" → 相似度: 0.72
      - "奉俊昊出生在哪里？" → 相似度: 0.95 ✓ (最高)
      - ...
    
    Chunk 2 的 Atoms:
      - "奉俊昊的第一部电影？" → 相似度: 0.68 ✓ (最高)
      - ...
    ↓
【步骤 4】为每个 Chunk 选择最佳 Atom
    [
        (Chunk 1, "奉俊昊出生在哪里？", 0.95),
        (Chunk 2, "奉俊昊的第一部电影？", 0.68)
    ]
    ↓
【步骤 5】组装 AtomRetrievalInfo 列表
```

#### 代码实现

```python
def retrieve_atom_info_through_chunk(
    self, 
    query: str, 
    retrieve_id: str = ""
) -> List[AtomRetrievalInfo]:
    """
    通过 Chunk Store 检索，返回每个 Chunk 的最佳匹配 Atom
    
    Args:
        query: 查询问题
        retrieve_id: 检索标识符
    
    Returns:
        List[AtomRetrievalInfo]: 检索结果列表
    """
    # 1. 在 Chunk Store 中检索
    chunk_info: List[Tuple[Document, float]] = self._get_doc_with_query(
        query, 
        self._chunk_store,  # 在 Chunk Store 中检索
        self.retrieve_k
    )
    
    # 2. 为每个 Chunk 找到最佳 Atom
    return self._chunk_info_tuple_to_class(
        query=query, 
        chunk_docs=[doc for doc, _ in chunk_info]
    )


def _chunk_info_tuple_to_class(
    self, 
    query: str, 
    chunk_docs: List[Document]
) -> List[AtomRetrievalInfo]:
    """
    为每个 Chunk 计算最佳匹配的 Atom
    """
    # 1. 向量化用户问题
    query_embedding = self.embedding_func.embed_query(query)
    
    # 2. 为每个 Chunk 找到最佳 Atom
    best_hit_atom_infos: List[Tuple[str, float, List[float]]] = []
    
    for chunk_doc in chunk_docs:
        best_atom, best_score, best_embedding = "", 0, []
        
        # 遍历该 Chunk 的所有 Atoms (存储在 metadata 中)
        for atom in chunk_doc.metadata["atom_questions_str"].split("\n"):
            # 向量化 Atom
            atom_embedding = self.embedding_func.embed_query(atom)
            
            # 计算相似度
            score = self.similarity_func(query_embedding, atom_embedding)
            
            # 更新最佳匹配
            if score > best_score:
                best_atom = atom
                best_score = score
                best_embedding = atom_embedding
        
        best_hit_atom_infos.append((best_atom, best_score, best_embedding))
    
    # 3. 组装 AtomRetrievalInfo
    retrieval_infos: List[AtomRetrievalInfo] = []
    for chunk_doc, (atom, score, atom_embedding) in zip(chunk_docs, best_hit_atom_infos):
        retrieval_infos.append(
            AtomRetrievalInfo(
                atom_query=query,
                atom=atom,
                source_chunk_title=chunk_doc.metadata.get("title", None),
                source_chunk=chunk_doc.page_content,
                source_chunk_id=chunk_doc.metadata["id"],
                retrieval_score=score,
                atom_embedding=atom_embedding
            )
        )
    
    return retrieval_infos
```

#### 方法 1 vs 方法 2 对比

| 维度 | 方法 1: through_atom | 方法 2: through_chunk |
|------|---------------------|---------------------|
| **检索对象** | Atom Store | Chunk Store |
| **检索粒度** | 细粒度（原子问题） | 粗粒度（文档块） |
| **精确度** | 高（直接匹配 Atom） | 中（需二次计算） |
| **计算成本** | 低（一次向量检索） | 高（检索 + N 次相似度计算） |
| **适用场景** | 精确问答、问题分解 | 探索性检索、上下文丰富 |

---

### 5.3 方法 3: retrieve_contents_by_query() - 综合检索

#### 原理
结合方法 1 和方法 2，既在 Atom Store 检索，也在 Chunk Store 检索，去重后返回

#### 流程图

```
用户问题
    ↓
【路径 A】通过 Atom Store 检索
    ↓
    Atom 1 → Chunk A
    Atom 2 → Chunk A (去重)
    Atom 3 → Chunk B
    ↓
【路径 B】通过 Chunk Store 检索
    ↓
    Chunk C
    Chunk A (去重)
    ↓
【合并去重】
    ↓
返回: [Chunk A, Chunk B, Chunk C]
```

#### 代码实现

```python
def retrieve_contents_by_query(
    self, 
    query: str, 
    retrieve_id: str = ""
) -> List[str]:
    """
    综合检索: 同时使用 Atom Store 和 Chunk Store
    
    Returns:
        List[str]: 去重后的 Chunk 内容列表
    """
    # 1. 从 Chunk Store 直接检索
    chunk_info: List[Tuple[Document, float]] = self._get_doc_with_query(
        query, 
        self._chunk_store, 
        self.retrieve_k
    )
    chunks = [chunk_doc.page_content for chunk_doc, _ in chunk_info]
    
    # 2. 从 Atom Store 检索，获取源 Chunks
    atom_infos = self.retrieve_atom_info_through_atom(
        queries=query, 
        retrieve_id=retrieve_id
    )
    atom_source_chunks = [atom_info.source_chunk for atom_info in atom_infos]
    
    # 3. 合并并去重
    for chunk in atom_source_chunks:
        if chunk not in chunks:
            chunks.append(chunk)
    
    return chunks
```

#### 使用场景

**基础 QA 工作流**:

```python
# pikerag/workflows/qa.py
class QaWorkflow:
    def answer(self, qa: BaseQaData, question_idx: int) -> dict:
        # 使用综合检索
        reference_chunks = self._retriever.retrieve_contents(
            qa, 
            retrieve_id=f"Q{question_idx:03}"
        )
        
        # 将检索结果提供给 LLM
        messages = self._qa_protocol.process_input(
            content=qa.question, 
            references=reference_chunks
        )
        
        response = self._client.generate_content_with_messages(messages)
        return self._qa_protocol.parse_output(response)
```

---

## 6. 数据流转详解

### 6.1 完整数据流转图

```
┌────────────────────────────────────────────────────────────────┐
│                     离线阶段: 知识图谱构建                     │
└────────────────────────────────────────────────────────────────┘

原始文档 (documents/)
    ↓
    | python examples/chunking.py config.yml
    ↓
文档块 (chunks.jsonl)
[
  {"chunk_id": "chunk_001", "content": "...", "title": "..."},
  {"chunk_id": "chunk_002", "content": "...", "title": "..."}
]
    ↓
    | python examples/tagging.py config.yml
    | (调用 LLM 提取原子问题)
    ↓
带标签文档块 (chunks_with_atoms.jsonl)
[
  {
    "chunk_id": "chunk_001",
    "content": "...",
    "title": "...",
    "atom_questions": ["问题1", "问题2", "问题3"]
  }
]
    ↓
    | 向量化 + 存储
    ↓
┌─────────────────────────────────────────────────┐
│           异构知识图谱 (双向量存储)              │
├─────────────────────────────────────────────────┤
│ Chunk Store                | Atom Store         │
│ - 5,000 chunks             | - 25,000 atoms     │
│ - 向量化的完整文档内容      | - 向量化的原子问题  │
│ - metadata: atom_questions | - metadata:        │
│                            |   source_chunk_id  │
└─────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                     在线阶段: 问答检索                         │
└────────────────────────────────────────────────────────────────┘

用户问题: "奉俊昊导演出生在哪里？"
    ↓
    | QaWorkflow.answer()
    ↓
检索器初始化
    ↓
    | ChunkAtomRetriever.__init__()
    | - 自动加载 Chunk Store
    | - 自动加载 Atom Store
    ↓
执行检索
    ↓
    | retriever.retrieve_contents_by_query(query)
    ↓
┌─────────────────┐          ┌─────────────────┐
│  路径 A:        │          │  路径 B:        │
│  Atom 检索      │          │  Chunk 检索     │
├─────────────────┤          ├─────────────────┤
│ 1. 向量化问题   │          │ 1. 向量化问题   │
│ 2. 检索 Atoms   │          │ 2. 检索 Chunks  │
│ 3. 获取源 Chunks│          │ 3. 直接返回     │
└────────┬────────┘          └────────┬────────┘
         │                            │
         └────────────┬───────────────┘
                      ↓
                合并 + 去重
                      ↓
            返回 Top-K Chunks
            [Chunk A, Chunk B, Chunk C]
                      ↓
            组装 Prompt
                      ↓
        messages = [
          {"role": "system", "content": "..."},
          {"role": "user", "content": 
            "参考资料:\n{Chunk A}\n{Chunk B}\n\n问题: 奉俊昊导演出生在哪里？"
          }
        ]
                      ↓
            调用 LLM
                      ↓
        response = client.generate(messages)
                      ↓
            解析输出
                      ↓
        answer = "奉俊昊导演1969年出生于韩国大邱市"
```

### 6.2 数据结构转换链

```
【阶段 1: 文档切分】
Document (LangChain)
├── page_content: str      # 完整文档块内容
└── metadata: dict
    ├── filename: str
    └── page: int
    ↓ (保存为 pickle)
List[Document] → chunks.pkl

【阶段 2: 原子问题提取】
输入: List[Document] (from chunks.pkl)
    ↓
LLM 提取
    ↓
输出: List[Document]
├── page_content: str      # 完整文档块内容 (不变)
└── metadata: dict
    ├── chunk_id: str      # 新增
    ├── title: str         # 新增
    └── atom_questions: List[str]  # 新增: LLM 提取的问题列表
    ↓ (保存为 JSONL)
[
  {
    "chunk_id": "chunk_001",
    "title": "...",
    "content": "...",
    "atom_questions": ["Q1", "Q2", "Q3"]
  }
] → chunks_with_atoms.jsonl

【阶段 3: 向量数据库构建】

输入: chunks_with_atoms.jsonl
    ↓
load_ids_and_chunks()
    ↓
Chunk Store 数据结构:
List[Document]
├── page_content: str      # 完整文档块内容
└── metadata: dict
    ├── id: str            # chunk_id
    ├── title: str
    └── atom_questions_str: str  # "Q1\nQ2\nQ3" (转为字符串)
    ↓
向量化 + 存储到 Chroma
    ↓

load_ids_and_atoms()
    ↓
Atom Store 数据结构:
List[Document]  (展开: 每个 atom 一个 Document)
├── page_content: str      # 单个原子问题
└── metadata: dict
    ├── source_chunk_id: str  # 指向源 Chunk
    └── title: str
    ↓
向量化 + 存储到 Chroma

【阶段 4: 检索】

用户问题: "奉俊昊导演出生在哪里？"
    ↓
retrieve_atom_info_through_atom()
    ↓
检索结果: List[Tuple[str, Document, float]]
[
  (
    "奉俊昊导演出生在哪里？",  # atom_query
    Document(
      page_content="奉俊昊导演出生在哪里？",
      metadata={"source_chunk_id": "chunk_001", "title": "..."}
    ),
    0.95  # score
  ),
  ...
]
    ↓
_atom_info_tuple_to_class()
    ↓
最终输出: List[AtomRetrievalInfo]
[
  AtomRetrievalInfo(
    atom_query="奉俊昊导演出生在哪里？",
    atom="奉俊昊导演出生在哪里？",
    source_chunk_id="chunk_001",
    source_chunk="《寄生虫》...奉俊昊1969年出生于韩国大邱市...",
    source_chunk_title="2020年奥斯卡最佳影片",
    retrieval_score=0.95,
    atom_embedding=[0.123, 0.456, ...]
  ),
  ...
]
```

### 6.3 关键字段关联图

```
┌──────────────────────────────────────────────────────────┐
│                    数据关联关系                          │
└──────────────────────────────────────────────────────────┘

chunks_with_atoms.jsonl
    ├── chunk_001
    │   ├── content: "..." ─────────────┐
    │   ├── title: "..."                │
    │   └── atom_questions:             │
    │       ├── "问题1" ────────┐       │
    │       ├── "问题2" ────┐   │       │
    │       └── "问题3" ─┐  │   │       │
    │                    │  │   │       │
    ├── chunk_002        │  │   │       │
    │   ├── content: ... │  │   │       │
    │   └── ...          │  │   │       │
    └── ...              │  │   │       │
                         │  │   │       │
        ┌────────────────┘  │   │       │
        │  ┌────────────────┘   │       │
        │  │  ┌─────────────────┘       │
        ↓  ↓  ↓                         ↓
    ┌────────────────┐         ┌──────────────┐
    │  Atom Store    │         │ Chunk Store  │
    ├────────────────┤         ├──────────────┤
    │ Atom Doc 1     │         │ Chunk Doc 1  │
    │ - content: Q1  │─ link ─→│ - id: c_001  │
    │ - meta:        │         │ - content:   │
    │   source_chunk │         │   "..."      │
    │   _id: c_001   │         │ - meta:      │
    │                │         │   atom_...   │
    │ Atom Doc 2     │         │   _str       │
    │ - content: Q2  │─ link ─→│              │
    │ - meta:        │         │              │
    │   source_chunk │         │              │
    │   _id: c_001   │         │              │
    │                │         │              │
    │ Atom Doc 3     │         │              │
    │ - content: Q3  │─ link ─→│              │
    │ - meta:        │         │              │
    │   source_chunk │         │              │
    │   _id: c_001   │         │              │
    └────────────────┘         └──────────────┘
            ↑                          ↑
            │                          │
        检索 Atoms               检索 Chunks
            │                          │
            └─────────┬────────────────┘
                      ↓
            合并返回 Chunks 给 LLM
```

---

## 7. 代码实现解析

### 7.1 核心类关系图

```
┌────────────────────────────────────────────────────┐
│                   工作流层                         │
├────────────────────────────────────────────────────┤
│ QaWorkflow                                         │
│ ├── QaDecompositionWorkflow (问题分解)            │
│ ├── QaIRCoTWorkflow (迭代检索)                    │
│ ├── QaSelfAskWorkflow (自我询问)                  │
│ └── QaIterRetgenWorkflow (迭代生成)               │
└────────────────┬───────────────────────────────────┘
                 │ uses
                 ↓
┌────────────────────────────────────────────────────┐
│                 检索器层                           │
├────────────────────────────────────────────────────┤
│ BaseQaRetriever                                    │
│ ├── ChunkAtomRetriever ★ (核心)                   │
│ ├── QaChunkRetriever                               │
│ └── BM25Retriever                                  │
└────────────┬───────────────────────────────────────┘
             │ uses
             ↓
┌────────────────────────────────────────────────────┐
│                  混入层                            │
├────────────────────────────────────────────────────┤
│ ChromaMixin (向量数据库操作)                       │
│ NetworkxMixin (图遍历操作)                         │
└────────────┬───────────────────────────────────────┘
             │ uses
             ↓
┌────────────────────────────────────────────────────┐
│                 存储层                             │
├────────────────────────────────────────────────────┤
│ Chroma (向量数据库)                                │
│ NetworkX (图数据库)                                │
└────────────────────────────────────────────────────┘
```

### 7.2 ChunkAtomRetriever 完整实现

**文件**: `pikerag/knowledge_retrievers/chunk_atom_retriever.py`

```python
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

@dataclass
class AtomRetrievalInfo:
    """原子检索信息数据类"""
    atom_query: str              # 用户的查询
    atom: str                    # 匹配到的原子问题
    source_chunk_title: str      # 源文档标题
    source_chunk: str            # 源文档完整内容
    source_chunk_id: str         # 源文档ID (关键关联字段)
    retrieval_score: float       # 检索分数
    atom_embedding: List[float]  # Atom 向量


class ChunkAtomRetriever(BaseQaRetriever, ChromaMixin):
    """
    双层向量存储检索器
    
    核心组件:
    - _chunk_store: Chunk 向量存储
    - _atom_store: Atom 向量存储
    
    公开接口:
    - retrieve_atom_info_through_atom(): 通过 Atom 检索
    - retrieve_atom_info_through_chunk(): 通过 Chunk 检索
    - retrieve_contents_by_query(): 综合检索
    - retrieve_contents(): 等价于 retrieve_contents_by_query(qa.question)
    """
    
    name: str = "ChunkAtomRetriever"
    
    def __init__(self, retriever_config: dict, log_dir: str, main_logger):
        super().__init__(retriever_config, log_dir, main_logger)
        
        # 加载双向量存储
        self._load_vector_store()
        
        # 初始化 Chroma 混入
        self._init_chroma_mixin()
        
        # 设置 Atom 检索的 k 值
        self.atom_retrieve_k = retriever_config.get(
            "atom_retrieve_k", 
            self.retrieve_k
        )
    
    def _load_vector_store(self):
        """加载 Chunk Store 和 Atom Store"""
        vector_store_config = self._retriever_config["vector_store"]
        
        # 集合名称
        collection_name = vector_store_config.get("collection_name", self.name)
        doc_collection_name = vector_store_config.get(
            "collection_name_doc", 
            f"{collection_name}_doc"
        )
        atom_collection_name = vector_store_config.get(
            "collection_name_atom", 
            f"{collection_name}_atom"
        )
        
        # 持久化目录
        persist_directory = vector_store_config.get(
            "persist_directory", 
            self._log_dir
        )
        exist_ok = vector_store_config.get("exist_ok", True)
        
        # 加载 Embedding 函数
        embedding_config = vector_store_config.get("embedding_setting", {})
        self.embedding_func = load_embedding_func(
            module_path=embedding_config.get("module_path"),
            class_name=embedding_config.get("class_name"),
            **embedding_config.get("args", {})
        )
        
        # 相似度函数 (余弦相似度)
        self.similarity_func = lambda x, y: (
            np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        )
        
        # 加载 Chunk Store
        loading_configs = vector_store_config["id_document_loading"]
        doc_ids, docs = load_callable(
            module_path=loading_configs["module_path"],
            name=loading_configs["func_name"],
        )(**loading_configs.get("args", {}))
        
        self._chunk_store = load_vector_store(
            collection_name=doc_collection_name,
            persist_directory=persist_directory,
            embedding=self.embedding_func,
            documents=docs,
            ids=doc_ids,
            exist_ok=exist_ok
        )
        
        # 加载 Atom Store
        loading_configs = vector_store_config["id_atom_loading"]
        atom_ids, atoms = load_callable(
            module_path=loading_configs["module_path"],
            name=loading_configs["func_name"],
        )(**loading_configs.get("args", {}))
        
        self._atom_store = load_vector_store(
            collection_name=atom_collection_name,
            persist_directory=persist_directory,
            embedding=self.embedding_func,
            documents=atoms,
            ids=atom_ids,
            exist_ok=exist_ok
        )
    
    # ... (retrieve 方法见前文详解)
```

### 7.3 NetworkxMixin 实现

**文件**: `pikerag/knowledge_retrievers/mixins/networkx_mixin.py`

```python
from typing import Iterable
import networkx as nx

class NetworkxMixin:
    """
    NetworkX 图遍历混入类
    
    提供基于图结构的知识扩展能力
    """
    
    def _init_networkx_mixin(self):
        """初始化图遍历参数"""
        self.entity_neighbor_layer: int = self._retriever_config.get(
            "entity_neighbor_layer", 
            1  # 默认扩展 1 跳邻居
        )
    
    def _get_subgraph_by_entity(
        self, 
        graph: nx.Graph, 
        entities: Iterable, 
        neighbor_layer: int = None
    ) -> nx.Graph:
        """
        根据实体提取子图
        
        算法:
        1. 从给定实体集合开始
        2. 迭代扩展 neighbor_layer 层邻居
        3. 返回包含所有相关节点的子图
        
        Args:
            graph: 完整知识图谱
            entities: 起始实体集合
            neighbor_layer: 扩展层数
        
        Returns:
            过滤后的子图
        
        示例:
            graph: A -- B -- C -- D
                   |
                   E
            
            entities = [A]
            neighbor_layer = 1
            → subgraph nodes = {A, B, E}
            
            neighbor_layer = 2
            → subgraph nodes = {A, B, C, E}
        """
        if neighbor_layer is None:
            neighbor_layer = self.entity_neighbor_layer
        
        # 初始化实体集合
        entity_set = set(entities)
        newly_added = entity_set.copy()
        
        # 迭代扩展邻居
        for layer in range(neighbor_layer):
            tmp_set = set()
            
            # 遍历当前层的所有节点
            for entity in newly_added:
                # 获取邻居节点
                for neighbor in graph.neighbors(entity):
                    if neighbor not in entity_set:
                        tmp_set.add(neighbor)
            
            # 更新
            newly_added = tmp_set
            entity_set.update(newly_added)
        
        # 返回子图
        return graph.subgraph(nodes=entity_set)
```

**使用场景示例**:

```python
# 构建知识图谱
import networkx as nx

# 创建图: Chunk 和 Atom 作为节点
G = nx.Graph()

# 添加 Chunk 节点
G.add_node("chunk_001", type="chunk", content="...")
G.add_node("chunk_002", type="chunk", content="...")

# 添加 Atom 节点
G.add_node("atom_001", type="atom", question="...")
G.add_node("atom_002", type="atom", question="...")
G.add_node("atom_003", type="atom", question="...")

# 添加边: Atom -> Chunk
G.add_edge("atom_001", "chunk_001")  # atom_001 来源于 chunk_001
G.add_edge("atom_002", "chunk_001")
G.add_edge("atom_003", "chunk_002")

# 添加边: Chunk -> Chunk (相关文档)
G.add_edge("chunk_001", "chunk_002")  # 两个文档相关

# 使用 NetworkxMixin 提取子图
mixin = NetworkxMixin()
mixin.entity_neighbor_layer = 2

# 从一个 Atom 出发，找到相关的所有知识
subgraph = mixin._get_subgraph_by_entity(
    graph=G,
    entities=["atom_001"],
    neighbor_layer=2
)

print(subgraph.nodes())
# 输出: ['atom_001', 'chunk_001', 'atom_002', 'chunk_002', 'atom_003']
# 解释: 从 atom_001 出发，1 跳到 chunk_001，2 跳到相关的其他 atoms 和 chunks
```

---

## 8. 配置示例

### 8.1 完整的问题分解工作流配置

**文件**: `examples/hotpotqa/configs/atomic_decompose.yml`

```yaml
# ============================================================
# 实验设置
# ============================================================
experiment_name: atomic_decompose
log_root_dir: logs/hotpotqa
test_jsonl_filename: null
test_rounds: 1

# ============================================================
# 工作流设置
# ============================================================
workflow:
  module_path: pikerag.workflows.qa_decompose
  class_name: QaDecompositionWorkflow
  args:
    max_num_question: 5                    # 最大分解问题数
    question_similarity_threshold: 0.999   # 问题去重阈值

# ============================================================
# 测试数据加载
# ============================================================
test_loading:
  module: pikerag.utils.data_protocol_utils
  name: load_testing_suite
  args:
    filepath: data/hotpotqa/dev_500.jsonl

# ============================================================
# Prompt 协议设置
# ============================================================
decompose_proposal_protocol:
  module_path: pikerag.prompts.decomposition
  protocol_name: question_decompose_protocol

selection_protocol:
  module_path: pikerag.prompts.decomposition
  protocol_name: atom_question_selection_protocol

backup_selection_protocol:
  module_path: pikerag.prompts.decomposition
  protocol_name: chunk_selection_protocol

original_question_answering_protocol:
  module_path: pikerag.prompts.decomposition
  protocol_name: final_qa_protocol

# ============================================================
# LLM 客户端设置
# ============================================================
llm_client:
  module_path: pikerag.llm_client
  class_name: AzureOpenAIClient
  args: {}
  
  llm_config:
    model: gpt-4
    temperature: 0
  
  cache_config:
    location_prefix: null  # 使用 experiment_name
    auto_dump: True

# ============================================================
# 检索器设置 (核心配置)
# ============================================================
retriever:
  module_path: pikerag.knowledge_retrievers
  class_name: ChunkAtomRetriever
  
  args:
    # Chunk 检索参数
    retrieve_k: 8                        # 每次检索返回的 Chunk 数量
    retrieve_score_threshold: 0.5        # 最低相似度阈值
    
    # Atom 检索参数
    atom_retrieve_k: 4                   # 每次检索返回的 Atom 数量
    
    # 向量存储配置
    vector_store:
      # 集合名称
      collection_name: dev_500_atomic_decompose_ada
      
      # 持久化目录
      persist_directory: data/vector_stores/hotpotqa
      
      # Chunk 数据加载
      id_document_loading:
        module_path: pikerag.utils.data_protocol_utils
        func_name: load_ids_and_chunks
        args:
          filepath: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl
          atom_tag: atom_questions
      
      # Atom 数据加载
      id_atom_loading:
        module_path: pikerag.utils.data_protocol_utils
        func_name: load_ids_and_atoms
        args:
          filepath: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl
          atom_tag: atom_questions
      
      # Embedding 设置
      embedding_setting:
        module_path: pikerag.llm_client.azure_open_ai_client
        class_name: AzureOpenAIEmbedding
        args: {}

# ============================================================
# 评估器设置
# ============================================================
evaluator:
  metrics:
    - ExactMatch
    - F1
    - Precision
    - Recall
    - LLM
```

### 8.2 Tagging 配置示例

**文件**: `examples/hotpotqa/configs/tagging.yml`

```yaml
# ============================================================
# 实验设置
# ============================================================
experiment_name: hotpotqa_dev_500
log_root_dir: logs/atom_tagging

# ============================================================
# 文档加载与保存
# ============================================================
ori_doc_loading:
  module: pikerag.utils.data_protocol_utils
  name: load_chunks_from_jsonl
  args:
    jsonl_chunk_path: data/hotpotqa/dev_500_retrieval_contexts_as_chunks.jsonl

tagged_doc_saving:
  module: pikerag.utils.data_protocol_utils
  name: save_chunks_to_jsonl
  args:
    dump_path: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl

# ============================================================
# Tagger 设置
# ============================================================
tagger:
  tagging_protocol:
    module_path: pikerag.prompts.tagging
    attr_name: atom_question_tagging_protocol
  
  tag_name: atom_questions
  
  num_parallel: 1  # 并行处理数量

# ============================================================
# LLM 设置
# ============================================================
llm_client:
  module_path: pikerag.llm_client
  class_name: AzureOpenAIClient
  args: {}
  
  llm_config:
    model: gpt-4
    temperature: 0.7  # 较高温度以增加问题多样性
  
  cache_config:
    location_prefix: null
    auto_dump: True
```

---

## 9. 总结与最佳实践

### 9.1 PIKE-RAG 知识图谱的核心创新

1. **异构双层结构**
   - Chunk 层: 保证上下文完整性
   - Atom 层: 提高检索精确度
   - 两层通过 `source_chunk_id` 关联

2. **语义对齐设计**
   - Atom 以问题形式表示
   - 与用户问题在语义空间天然对齐
   - 提高检索召回率

3. **灵活的检索策略**
   - through_atom: 高精度检索
   - through_chunk: 上下文丰富
   - 综合检索: 平衡精度和覆盖

4. **可扩展的图结构**
   - NetworkxMixin 提供图遍历能力
   - 支持多跳推理和关联发现

### 9.2 最佳实践

#### 1. 文档切分 (Chunking)

**推荐参数**:
```yaml
chunk_size: 800-1200        # 根据领域调整
chunk_overlap: 150-250      # 保证知识不丢失
```

**注意事项**:
- 保持 Chunk 语义完整性
- 避免在句子中间切分
- 使用 LLM 辅助的智能切分

#### 2. 原子问题提取 (Tagging)

**Prompt 设计要点**:
```
✓ 要求多样性: "extract as many questions as possible"
✓ 避免代词: "avoid pronouns like it, he, she"
✓ 包含实体: "contain necessary entity names"
✓ 格式明确: "output line by line"
```

**质量控制**:
- 每个 Chunk 提取 3-7 个 Atoms
- 去重相似问题
- 人工抽查质量

#### 3. 检索参数调优

**通用设置**:
```yaml
retrieve_k: 4-8             # 根据任务复杂度
atom_retrieve_k: 2-4        # 通常小于 retrieve_k
retrieve_score_threshold: 0.3-0.6  # 避免噪声
```

**场景特定**:
- 简单问答: retrieve_k=4, atom_retrieve_k=2
- 多跳推理: retrieve_k=8, atom_retrieve_k=4
- 探索性问题: retrieve_k=10, atom_retrieve_k=5

#### 4. 性能优化

**向量化优化**:
- 使用批量 embedding
- 缓存常用查询
- 选择合适的 embedding 模型 (ada-002, bge-large 等)

**存储优化**:
- 定期清理过期缓存
- 使用 SSD 存储向量数据库
- 考虑分布式部署 (大规模场景)

**检索优化**:
- 使用 approximate nearest neighbor (ANN)
- 预先过滤明显不相关的文档
- 并行检索 Chunk Store 和 Atom Store

### 9.3 常见问题 (FAQ)

**Q1: 为什么不使用传统的实体-关系图谱？**

A: PIKE-RAG 的异构知识图谱更适合文档问答场景:
- 不需要预定义 schema
- LLM 自动提取知识点
- 避免实体识别和关系抽取的误差
- 更灵活，适应多样化的查询

**Q2: Atom 和 Chunk 的比例应该是多少？**

A: 推荐 Atom:Chunk = 3:1 到 7:1
- 比例太低: 检索精度不足
- 比例太高: 增加存储和计算成本
- 根据文档密度调整

**Q3: 如何处理多语言场景？**

A:
- 使用多语言 Embedding 模型 (如 multilingual-e5)
- 分别为每种语言提取 Atoms
- 在 metadata 中标记语言

**Q4: 知识图谱需要多久更新一次？**

A: 根据数据变化频率:
- 静态知识库: 一次性构建
- 周更新: 增量更新 (添加新 Chunks 和 Atoms)
- 实时更新: 流式处理 + 定期重建索引

**Q5: 如何评估知识图谱质量？**

A: 关键指标:
- Atom 质量: 人工评估相关性和多样性
- 检索效果: Recall@K, MRR
- 端到端: QA 任务的 F1/EM 分数

### 9.4 进阶应用

#### 1. 多跳推理增强

```python
# 使用 NetworkxMixin 构建显式图
G = nx.Graph()

# 添加 Chunk-Atom 关系
for chunk_id, atoms in chunks_with_atoms:
    G.add_node(chunk_id, type="chunk")
    for atom in atoms:
        atom_id = hash(atom)
        G.add_node(atom_id, type="atom", question=atom)
        G.add_edge(atom_id, chunk_id)

# 添加 Chunk-Chunk 相似度边
for chunk1, chunk2, similarity in chunk_similarities:
    if similarity > threshold:
        G.add_edge(chunk1, chunk2, weight=similarity)

# 多跳检索
start_atoms = retriever.retrieve_atom_info_through_atom(query)
start_chunk_ids = [info.source_chunk_id for info in start_atoms]

# 扩展 2 跳邻居
subgraph = mixin._get_subgraph_by_entity(G, start_chunk_ids, neighbor_layer=2)
expanded_chunks = [n for n in subgraph.nodes() if G.nodes[n]["type"] == "chunk"]
```

#### 2. 知识图谱可视化

```python
import matplotlib.pyplot as plt
import networkx as nx

# 绘制子图
pos = nx.spring_layout(subgraph)
nx.draw(subgraph, pos, with_labels=True, node_color='lightblue')
plt.show()
```

#### 3. 知识蒸馏

将 LLM 提取的 Atoms 蒸馏为小模型:
```python
# 使用 Atoms 作为训练数据
train_data = [
    {"question": atom, "context": chunk, "answer": extract_answer(atom, chunk)}
    for chunk_id, atoms in tagged_chunks
    for atom in atoms
]

# 微调小模型
model.train(train_data)
```

---

## 📚 参考资源

### 代码文件索引

| 功能 | 文件路径 |
|------|---------|
| Chunking 工作流 | `pikerag/workflows/chunking.py` |
| Tagging 工作流 | `pikerag/workflows/tagging.py` |
| ChunkAtomRetriever | `pikerag/knowledge_retrievers/chunk_atom_retriever.py` |
| NetworkxMixin | `pikerag/knowledge_retrievers/mixins/networkx_mixin.py` |
| Atom Tagging Prompt | `pikerag/prompts/tagging/atom_question_tagging.py` |
| 数据加载工具 | `pikerag/utils/data_protocol_utils.py` |
| QA 工作流 | `pikerag/workflows/qa.py` |
| QA 分解工作流 | `pikerag/workflows/qa_decompose.py` |

### 配置文件索引

| 用途 | 文件路径 |
|------|---------|
| Chunking 配置 | `examples/biology/configs/chunking.yml` |
| Tagging 配置 | `examples/hotpotqa/configs/tagging.yml` |
| QA 配置 | `examples/hotpotqa/configs/atomic_decompose.yml` |
| Retriever 模板 | `pikerag/knowledge_retrievers/templates/ChunkAtomRetriever.yml` |

### 相关文档

- [基础 QA 工作流详解](./基础QA工作流详解.md)
- [ChunkAtomRetriever 详解](./ChunkAtomRetriever详解.md)
- [文档处理与智能切分详解](./文档处理与智能切分详解.md)

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**作者**: PIKE-RAG Team

