# ChunkAtomRetriever 详解：多粒度知识表示

## 📖 核心理念

ChunkAtomRetriever 是 PIKE-RAG 最核心的创新之一，它实现了**多粒度知识表示**策略，解决了传统 RAG 系统的一个关键问题：

### 传统 RAG 的困境

```
问题: "2020年奥斯卡最佳影片的导演出生在哪里？"

传统方法：
    ↓
直接用问题检索文档块
    ↓
❌ 问题：检索精度不高
   - 文档块可能很长（500-1000字）
   - 向量匹配不够精确
   - 可能检索到不相关的内容
```

### ChunkAtomRetriever 的解决方案

```
核心思想：粗粒度存储 + 细粒度检索

【Chunk（粗粒度）】
- 完整的文档片段
- 包含丰富的上下文信息
- 用于最终的 LLM 输入

【Atom（细粒度）】  
- 从 Chunk 提取的原子级知识点
- 以问题形式表示
- 用于精确检索匹配

工作流程：
    用户问题 → 检索相关 Atoms（高精度）
           ↓
    Atoms → 定位源 Chunks（丰富上下文）
           ↓
    返回 Chunks 给 LLM（完整信息）
```

**优势：**
- ✅ 检索精确度高（Atom 级别匹配）
- ✅ 返回上下文丰富（Chunk 级别内容）
- ✅ 支持多跳推理
- ✅ 语义对齐更好

---

## 🏗️ 架构设计

### 数据结构

#### 1. AtomRetrievalInfo - 原子检索信息

```python
@dataclass
class AtomRetrievalInfo:
    atom_query: str              # 用于检索的查询
    atom: str                    # 检索到的原子问题
    source_chunk_title: str      # 源文档块的标题
    source_chunk: str            # 源文档块的完整内容
    source_chunk_id: str         # 源文档块的ID
    retrieval_score: float       # 检索相似度分数
    atom_embedding: List[float]  # 原子问题的向量表示
```

**示例数据：**
```python
AtomRetrievalInfo(
    atom_query="谁导演了《寄生虫》？",
    atom="奉俊昊导演了《寄生虫》",
    source_chunk_title="2020年奥斯卡最佳影片",
    source_chunk="《寄生虫》是由韩国导演奉俊昊执导的黑色喜剧惊悚片...",
    source_chunk_id="chunk_0042",
    retrieval_score=0.92,
    atom_embedding=[0.123, -0.456, ...]
)
```

---

### 双向量存储架构

ChunkAtomRetriever 内部维护**两个独立的向量数据库**：

```
┌─────────────────────────────────────────────────────────────┐
│ ChunkAtomRetriever                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────┐      ┌──────────────────────┐  │
│  │ _chunk_store          │      │ _atom_store          │  │
│  │ (粗粒度向量数据库)     │      │ (细粒度向量数据库)    │  │
│  ├───────────────────────┤      ├──────────────────────┤  │
│  │ • 存储完整文档块       │      │ • 存储原子问题        │  │
│  │ • 包含丰富上下文       │      │ • 精确的知识点        │  │
│  │ • 用于最终输出         │      │ • 用于检索匹配        │  │
│  │                       │      │                      │  │
│  │ Chunk ID: chunk_001   │◄─────┤ Metadata:            │  │
│  │ Content: "..."        │  关联 │   source_chunk_id    │  │
│  │ Metadata: {...}       │      │     = chunk_001      │  │
│  └───────────────────────┘      └──────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 工作流程详解

### 阶段 1：数据准备（离线）

在使用 ChunkAtomRetriever 之前，需要对文档进行预处理：

```
原始文档
    ↓
【步骤 1: 文档切分】chunking.py
    ↓
文档块（Chunks）
    ↓
【步骤 2: 原子问题提取】tagging.py
    ↓
带原子问题的文档块
    ↓
【步骤 3: 构建向量数据库】
    ↓
双向量存储（Chunk Store + Atom Store）
```

#### 详细步骤

##### **步骤 1: 文档切分**

使用 `pikerag/workflows/chunking.py` 进行智能切分：

```python
# 配置示例：examples/biology/configs/chunking.yml
splitter:
  module_path: pikerag.document_transformers.splitter
  class_name: LLMPoweredRecursiveSplitter
  args:
    chunk_size: 1000
    chunk_overlap: 200
```

**输出示例：**
```python
Chunk(
    page_content="《寄生虫》是由韩国导演奉俊昊执导的2019年黑色喜剧惊悚片...",
    metadata={
        "chunk_id": "chunk_001",
        "title": "2020年奥斯卡最佳影片",
        "source": "movie_database.txt"
    }
)
```

---

##### **步骤 2: 原子问题提取**

使用 `pikerag/workflows/tagging.py` 提取原子问题：

**提示模板：** `pikerag/prompts/tagging/atom_question_tagging.py`

```python
atom_question_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at content understanding and asking question."),
        ("user", """
# Task
Your task is to extract as many questions as possible that are relevant and can be 
answered by the given content. Please try to be diverse and avoid extracting duplicated 
or similar questions. Make sure your question contain necessary entity names and avoid 
to use pronouns like it, he, she, they, the company, the person etc.

# Output Format
Output your answers line by line, with each question on a new line, without itemized 
symbols or numbers.

# Content
{content}

# Output:
"""),
    ],
)
```

**LLM 输出示例：**
```
Which movie won the Best Picture at the 2020 Academy Awards?
Who directed the movie Parasite?
What genre is the movie Parasite?
When was the movie Parasite released?
Where is the director Bong Joon-ho from?
```

**数据结构：**
```python
Chunk_with_Atoms = {
    "chunk_id": "chunk_001",
    "title": "2020年奥斯卡最佳影片",
    "content": "《寄生虫》是由韩国导演奉俊昊执导...",
    "atom_questions": [
        "Which movie won the Best Picture at the 2020 Academy Awards?",
        "Who directed the movie Parasite?",
        "What genre is the movie Parasite?",
        ...
    ]
}
```

---

##### **步骤 3: 构建双向量数据库**

在 ChunkAtomRetriever 初始化时自动构建：

```python
def _load_vector_store(self) -> None:
    # 1. 加载 Embedding 函数
    self.embedding_func = load_embedding_func(...)
    
    # 2. 加载文档数据（Chunks）
    # 调用: pikerag.utils.data_protocol_utils.load_ids_and_chunks
    doc_ids, docs = load_callable(...)(
        filepath="data/.../chunks_with_atom_questions.jsonl",
        atom_tag="atom_questions"
    )
    
    # 3. 构建 Chunk 向量存储
    self._chunk_store = load_vector_store(
        collection_name="collection_name_doc",
        documents=docs,  # 包含完整的 Chunk 内容
        ids=doc_ids,
        ...
    )
    
    # 4. 加载原子问题数据（Atoms）
    # 调用: pikerag.utils.data_protocol_utils.load_ids_and_atoms
    atom_ids, atoms = load_callable(...)(
        filepath="data/.../chunks_with_atom_questions.jsonl",
        atom_tag="atom_questions"
    )
    # 这会将每个 Chunk 的原子问题展开为独立的 Document
    
    # 5. 构建 Atom 向量存储
    self._atom_store = load_vector_store(
        collection_name="collection_name_atom",
        documents=atoms,  # 每个 Atom 是一个独立的 Document
        ids=atom_ids,     # 可以为 None，自动生成
        ...
    )
```

**Atom Document 结构：**
```python
Document(
    page_content="Who directed the movie Parasite?",  # 原子问题
    metadata={
        "source_chunk_id": "chunk_001"  # 指向源 Chunk
    }
)
```

---

### 阶段 2：检索（在线）

ChunkAtomRetriever 提供**四种检索接口**：

#### 接口 1: `retrieve_atom_info_through_atom` - 通过原子问题检索

**最核心的检索方法，用于精确匹配**

```python
def retrieve_atom_info_through_atom(
    self, 
    queries: Union[List[str], str],  # 查询问题（可以是单个或多个）
    retrieve_id: str = "",
    **kwargs
) -> List[AtomRetrievalInfo]:
    """通过给定的查询在 _atom_store 中检索相关的原子信息
    
    返回：原子信息 + 对应的源 Chunk 信息
    """
```

**工作流程：**

```python
# 1. 在 Atom Store 中进行向量检索
queries = ["Who directed Parasite?"]
    ↓
atom_store.similarity_search_with_relevance_scores(
    query="Who directed Parasite?",
    k=4  # 检索 top-4
)
    ↓
检索结果（Atom Documents）:
[
    (Document("Who directed the movie Parasite?", 
              metadata={"source_chunk_id": "chunk_001"}), 
     score=0.95),
    (Document("What is the nationality of Bong Joon-ho?",
              metadata={"source_chunk_id": "chunk_001"}),
     score=0.82),
    ...
]

# 2. 提取所有唯一的 source_chunk_id
source_chunk_ids = ["chunk_001", "chunk_003", ...]

# 3. 从 Chunk Store 批量获取对应的完整 Chunk
chunk_results = _chunk_store.get(ids=source_chunk_ids)
    ↓
{
    "ids": ["chunk_001", "chunk_003", ...],
    "documents": ["《寄生虫》是由韩国导演奉俊昊...", "奉俊昊出生于..."],
}

# 4. 组装成 AtomRetrievalInfo 对象
return [
    AtomRetrievalInfo(
        atom_query="Who directed Parasite?",
        atom="Who directed the movie Parasite?",
        source_chunk="《寄生虫》是由韩国导演奉俊昊...",
        source_chunk_id="chunk_001",
        retrieval_score=0.95,
        ...
    ),
    ...
]
```

**代码实现：**

```python
def retrieve_atom_info_through_atom(self, queries: Union[List[str], str], ...) -> List[AtomRetrievalInfo]:
    # A. 决定 retrieve_k
    if isinstance(queries, str):
        queries = [queries]
    retrieve_k = kwargs.get("retrieve_k", self.atom_retrieve_k)
    
    # B. 查询 _atom_store 获取相关原子信息
    query_atom_score_tuples: List[Tuple[str, Document, float]] = []
    for atom_query in queries:
        for atom_doc, score in self._get_doc_with_query(atom_query, self._atom_store, retrieve_k):
            query_atom_score_tuples.append((atom_query, atom_doc, score))
    
    # C. 封装为 AtomRetrievalInfo
    return self._atom_info_tuple_to_class(query_atom_score_tuples)
```

```python
def _atom_info_tuple_to_class(self, atom_retrieval_info: List[Tuple[str, Document, float]]) -> List[AtomRetrievalInfo]:
    # 1. 提取所有唯一的 source_chunk_id
    source_chunk_ids = list(set([
        doc.metadata["source_chunk_id"] 
        for _, doc, _ in atom_retrieval_info
    ]))
    
    # 2. 批量检索对应的源 Chunks
    chunk_doc_results = self._chunk_store.get(ids=source_chunk_ids)
    chunk_id_to_content = {
        chunk_id: chunk_str
        for chunk_id, chunk_str in zip(
            chunk_doc_results["ids"], 
            chunk_doc_results["documents"]
        )
    }
    
    # 3. 组装完整信息
    retrieval_infos = []
    for atom_query, atom_doc, score in atom_retrieval_info:
        source_chunk_id = atom_doc.metadata["source_chunk_id"]
        retrieval_infos.append(
            AtomRetrievalInfo(
                atom_query=atom_query,
                atom=atom_doc.page_content,
                source_chunk_title=atom_doc.metadata.get("title", None),
                source_chunk=chunk_id_to_content[source_chunk_id],
                source_chunk_id=source_chunk_id,
                retrieval_score=score,
                atom_embedding=self.embedding_func.embed_query(atom_doc.page_content),
            )
        )
    
    return retrieval_infos
```

---

#### 接口 2: `retrieve_atom_info_through_chunk` - 通过 Chunk 检索

**先检索 Chunk，再找最匹配的 Atom**

```python
def retrieve_atom_info_through_chunk(
    self, 
    query: str,
    retrieve_id: str = ""
) -> List[AtomRetrievalInfo]:
    """通过给定的查询在 _chunk_store 中检索相关 Chunk，
    然后找出每个 Chunk 中最匹配查询的 Atom
    
    返回：Chunk 信息 + 最佳匹配的 Atom 信息
    """
```

**工作流程：**

```python
query = "Who directed Parasite?"

# 1. 在 Chunk Store 中检索
chunk_infos = _chunk_store.similarity_search_with_relevance_scores(
    query="Who directed Parasite?",
    k=8
)
    ↓
检索到的 Chunks:
[
    Document("《寄生虫》是由韩国导演奉俊昊执导...", 
             metadata={"id": "chunk_001", "atom_questions_str": "..."}),
    Document("奉俊昊1969年出生于韩国...",
             metadata={"id": "chunk_003", "atom_questions_str": "..."}),
    ...
]

# 2. 对每个 Chunk，计算其所有 Atom 与查询的相似度
query_embedding = embed("Who directed Parasite?")

for chunk in chunks:
    atoms = chunk.metadata["atom_questions_str"].split("\n")
    # ["Who directed the movie Parasite?",
    #  "What genre is Parasite?",
    #  ...]
    
    best_atom = None
    best_score = 0
    for atom in atoms:
        atom_embedding = embed(atom)
        score = cosine_similarity(query_embedding, atom_embedding)
        if score > best_score:
            best_atom = atom
            best_score = score
    
    # 返回该 Chunk 及其最佳匹配的 Atom

# 3. 组装成 AtomRetrievalInfo
return [
    AtomRetrievalInfo(
        atom_query="Who directed Parasite?",
        atom="Who directed the movie Parasite?",  # 最佳匹配
        source_chunk="《寄生虫》是由韩国导演奉俊昊...",
        source_chunk_id="chunk_001",
        retrieval_score=0.93,  # Atom 与查询的相似度
        ...
    ),
    ...
]
```

**代码实现：**

```python
def retrieve_atom_info_through_chunk(self, query: str, ...) -> List[AtomRetrievalInfo]:
    # 1. 查询 _chunk_store 获取相关 Chunk
    chunk_info = self._get_doc_with_query(query, self._chunk_store, self.retrieve_k)
    
    # 2. 对每个 Chunk 找最佳匹配的 Atom
    return self._chunk_info_tuple_to_class(
        query=query, 
        chunk_docs=[doc for doc, _ in chunk_info]
    )
```

```python
def _chunk_info_tuple_to_class(self, query: str, chunk_docs: List[Document]) -> List[AtomRetrievalInfo]:
    # 获取查询的向量表示
    query_embedding = self.embedding_func.embed_query(query)
    
    # 对每个 Chunk 计算最佳匹配的 Atom
    best_hit_atom_infos = []
    for chunk_doc in chunk_docs:
        best_atom, best_score, best_embedding = "", 0, []
        
        # 遍历该 Chunk 的所有 Atoms
        for atom in chunk_doc.metadata["atom_questions_str"].split("\n"):
            atom_embedding = self.embedding_func.embed_query(atom)
            score = self.similarity_func(query_embedding, atom_embedding)
            
            if score > best_score:
                best_atom = atom
                best_score = score
                best_embedding = atom_embedding
        
        best_hit_atom_infos.append((best_atom, best_score, best_embedding))
    
    # 组装完整信息
    retrieval_infos = []
    for chunk_doc, (atom, score, atom_embedding) in zip(chunk_docs, best_hit_atom_infos):
        retrieval_infos.append(
            AtomRetrievalInfo(
                atom_query=query,
                atom=atom,
                source_chunk_title=chunk_doc.metadata.get("title", None),
                source_chunk=chunk_doc.page_content,
                source_chunk_id=chunk_doc.metadata["id"],
                retrieval_score=score,
                atom_embedding=atom_embedding,
            )
        )
    
    return retrieval_infos
```

---

#### 接口 3: `retrieve_contents_by_query` - 混合检索返回 Chunk 内容

**结合两种检索方法，返回去重后的 Chunk 列表**

```python
def retrieve_contents_by_query(
    self, 
    query: str,
    retrieve_id: str = ""
) -> List[str]:
    """混合检索：同时通过 Atom Store 和 Chunk Store 检索
    
    返回：去重后的 Chunk 内容列表
    """
```

**工作流程：**

```python
query = "Who directed Parasite?"

# 路径 1: 直接从 Chunk Store 检索
chunks_from_chunk_store = _chunk_store.similarity_search(query, k=8)
chunks_1 = [doc.page_content for doc in chunks_from_chunk_store]

# 路径 2: 通过 Atom Store 检索
atom_infos = retrieve_atom_info_through_atom(query)
chunks_2 = [info.source_chunk for info in atom_infos]

# 合并并去重
all_chunks = chunks_1
for chunk in chunks_2:
    if chunk not in all_chunks:
        all_chunks.append(chunk)

return all_chunks
```

**代码实现：**

```python
def retrieve_contents_by_query(self, query: str, ...) -> List[str]:
    # 1. 从 Chunk Store 直接检索
    chunk_info = self._get_doc_with_query(query, self._chunk_store, self.retrieve_k)
    chunks = [chunk_doc.page_content for chunk_doc, _ in chunk_info]
    
    # 2. 通过 Atom Store 检索
    atom_infos = self.retrieve_atom_info_through_atom(queries=query, retrieve_id=retrieve_id)
    atom_source_chunks = [atom_info.source_chunk for atom_info in atom_infos]
    
    # 3. 合并去重
    for chunk in atom_source_chunks:
        if chunk not in chunks:
            chunks.append(chunk)
    
    return chunks
```

---

#### 接口 4: `retrieve_contents` - 基础接口

```python
def retrieve_contents(self, qa: BaseQaData, retrieve_id: str = "") -> List[str]:
    """继承自 BaseQaRetriever 的接口
    默认使用问题作为查询，调用 retrieve_contents_by_query
    """
    return self.retrieve_contents_by_query(qa.question, retrieve_id)
```

---

## 🎯 使用场景对比

### 场景 1: 简单问答（单跳）

**问题：** "What is the capital of France?"

**使用：** `retrieve_contents_by_query`

```python
# 混合检索，覆盖面广
chunks = retriever.retrieve_contents_by_query("What is the capital of France?")
# 返回: ["France is a country in Europe. Its capital is Paris...", ...]
```

---

### 场景 2: 复杂多跳推理

**问题：** "Where was the director of the 2020 Oscar Best Picture born?"

**使用：** `retrieve_atom_info_through_atom`（在分解工作流中）

```python
# 问题分解后：
sub_questions = [
    "Which movie won the 2020 Oscar Best Picture?",
    "Who directed this movie?",
    "Where was this director born?"
]

# 逐步检索原子信息
for sub_q in sub_questions:
    atom_infos = retriever.retrieve_atom_info_through_atom(sub_q)
    # 每个 atom_info 包含：
    #   - atom: 精确匹配的原子问题
    #   - source_chunk: 完整的上下文
    #   - retrieval_score: 匹配分数
```

**详见：** `pikerag/workflows/qa_decompose.py`

---

## 📊 性能优势分析

### 对比实验

| 检索方法 | HotpotQA EM | 2WikiMultiHopQA EM | MuSiQue EM |
|---------|-------------|-------------------|------------|
| 仅 Chunk 检索 | 71.2% | 68.5% | 42.3% |
| ChunkAtomRetriever | **87.6%** | **82.0%** | **59.6%** |

### 为什么 ChunkAtomRetriever 更好？

#### 1. **检索精度提升**

```
传统 Chunk 检索：
    查询: "Who directed Parasite?"
    匹配对象: 长文档块（500-1000字）
    问题: 向量表示可能模糊，匹配不精确
    
ChunkAtomRetriever:
    查询: "Who directed Parasite?"
    匹配对象: 原子问题 "Who directed the movie Parasite?"
    优势: 问题-问题匹配，语义对齐更精确
```

#### 2. **上下文完整性**

```
传统方法：
    如果检索窗口太小 → 信息不完整
    如果检索窗口太大 → 噪音增多
    
ChunkAtomRetriever:
    通过 Atom 定位到相关 Chunk
    → 既精确又保留完整上下文
```

#### 3. **多跳推理支持**

```
复杂问题: "A的B的C是什么？"

传统方法：
    一次检索 → 可能只找到 A 的信息或 C 的信息
    
ChunkAtomRetriever:
    第1跳: 检索关于 A 的 Atom → 获取 Chunk_A
    第2跳: 检索关于 B 的 Atom → 获取 Chunk_B
    第3跳: 检索关于 C 的 Atom → 获取 Chunk_C
    → 逐步收集所有需要的信息
```

---

## 🔍 深入理解：Atom 的本质

### Atom 是什么？

**Atom（原子问题）** 不是简单的关键词或短语，而是：

1. **可回答的完整问题**
   ```
   ✅ "Who directed the movie Parasite?"
   ❌ "director Parasite"  (不是完整问题)
   ```

2. **包含必要实体**
   ```
   ✅ "Where was Bong Joon-ho born?"
   ❌ "Where was he born?"  (缺少实体名称)
   ```

3. **独立可理解**
   ```
   ✅ "What genre is the movie Parasite?"
   ❌ "What genre is it?"  (需要上下文)
   ```

### 为什么用问题形式？

1. **语义对齐**：用户的查询通常是问题形式
2. **信息密度高**：一个问题包含多个关键实体和关系
3. **检索精确**：问题与问题的匹配比问题与陈述的匹配更准确

### Atom 提取示例

**原始 Chunk：**
```
《寄生虫》是由韩国导演奉俊昊执导的2019年黑色喜剧惊悚片。
该片在2020年第92届奥斯卡金像奖上创造历史，成为首部获得
最佳影片奖的非英语电影。奉俊昊因此片获得最佳导演奖。
```

**提取的 Atoms：**
```
1. Which movie won the Best Picture at the 2020 Academy Awards?
2. Who directed the movie Parasite?
3. What is the nationality of Bong Joon-ho?
4. What genre is the movie Parasite?
5. When was the movie Parasite released?
6. What awards did Bong Joon-ho win for Parasite?
7. What is special about Parasite winning the Best Picture award?
```

**注意：** 每个 Atom 都可以由该 Chunk 直接回答！

---

## 🛠️ 实践指南

### 配置 ChunkAtomRetriever

**完整配置示例：** `examples/hotpotqa/configs/atomic_decompose.yml`

```yaml
retriever:
  module_path: pikerag.knowledge_retrievers
  class_name: ChunkAtomRetriever
  args:
    # Chunk 检索参数
    retrieve_k: 8                    # Chunk Store 返回 top-8
    retrieve_score_threshold: 0.5    # 相似度阈值
    
    # Atom 检索参数
    atom_retrieve_k: 4               # Atom Store 返回 top-4
    
    vector_store:
      collection_name: dev_500_atomic_decompose_ada
      persist_directory: data/vector_stores/hotpotqa
      
      # Chunk 数据加载
      id_document_loading:
        module_path: pikerag.utils.data_protocol_utils
        func_name: load_ids_and_chunks
        args:
          filepath: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl
          atom_tag: atom_questions  # 指定 Atom 字段名
      
      # Atom 数据加载
      id_atom_loading:
        module_path: pikerag.utils.data_protocol_utils
        func_name: load_ids_and_atoms
        args:
          filepath: data/hotpotqa/dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl
          atom_tag: atom_questions
      
      # Embedding 配置
      embedding_setting:
        module_path: pikerag.llm_client.azure_open_ai_client
        class_name: AzureOpenAIEmbedding
        args: {}
```

---

### 数据准备流程

#### Step 1: 文档切分

```bash
python examples/chunking.py examples/biology/configs/chunking.yml
```

**输出：** `data/output/chunks.pkl` 或 `.jsonl`

---

#### Step 2: 原子问题提取

```bash
python examples/tagging.py examples/biology/configs/tagging.yml
```

**配置示例：**
```yaml
tagger:
  tag_name: atom_questions
  tagging_protocol:
    module_path: pikerag.prompts.tagging
    attr_name: atom_question_tagging_protocol

input_doc_setting:
  doc_dir: data/output/
  extensions: [".jsonl"]

output_doc_setting:
  doc_dir: data/output_with_atoms/
  suffix: jsonl
```

**输出：** 
```jsonl
{
  "chunk_id": "chunk_001",
  "title": "Document Title",
  "content": "Full chunk content...",
  "atom_questions": [
    "Question 1?",
    "Question 2?",
    ...
  ]
}
```

---

#### Step 3: 运行问答

```bash
python examples/qa.py examples/hotpotqa/configs/atomic_decompose.yml
```

ChunkAtomRetriever 会在首次运行时自动构建向量数据库。

---

## 💡 高级技巧

### 1. 调优检索参数

```yaml
# 提高召回率（更多结果）
retrieve_k: 16              # 增加 Chunk 数量
atom_retrieve_k: 8          # 增加 Atom 数量

# 提高精确度（更严格）
retrieve_score_threshold: 0.7  # 提高相似度阈值
```

### 2. 自定义 Atom 标签

```python
# 可以为不同类型的知识提取不同类型的 Atoms
atom_tags = {
    "atom_questions": "问题形式的原子知识",
    "atom_facts": "事实陈述形式的原子知识",
    "atom_entities": "实体关系形式的原子知识",
}
```

### 3. 混合使用两种检索方式

```python
# 在分解工作流中
atom_infos = retriever.retrieve_atom_info_through_atom(sub_questions)

# 如果 Atom 检索结果不够
if len(atom_infos) < threshold:
    # 回退到 Chunk 检索
    backup_infos = retriever.retrieve_atom_info_through_chunk(original_question)
    atom_infos.extend(backup_infos)
```

**参见：** `pikerag/workflows/qa_decompose.py` 中的 `_retrieve_atom_info_candidates` 方法

---

## 📝 总结

ChunkAtomRetriever 的核心价值：

### 核心理念
```
粗粒度存储（Chunk）+ 细粒度检索（Atom）= 精确且丰富
```

### 关键优势

1. **检索精度高** 
   - Atom 与查询的语义对齐更好
   - 问题-问题匹配天然准确

2. **上下文完整**
   - 返回完整的 Chunk 内容
   - 保留足够的上下文信息

3. **支持多跳推理**
   - 每个子问题独立检索 Atom
   - 逐步构建推理链

4. **可解释性强**
   - 可以看到匹配的 Atom
   - 理解为什么检索到某个 Chunk

### 适用场景

✅ **适合：**
- 多跳问答
- 复杂推理任务
- 需要高精度检索的场景
- 专业领域知识问答

❌ **不适合：**
- 简单关键词查询（过度设计）
- 实时性要求极高的场景（Atom 提取需要时间）
- 文档频繁更新的场景（需要重新提取 Atoms）

---

## 🚀 下一步学习

现在您已经理解了 ChunkAtomRetriever，建议继续学习：

1. **问题分解工作流** (`qa_decompose.py`)
   - 如何利用 ChunkAtomRetriever 进行多跳推理
   - 迭代检索和信息选择策略

2. **Atom 提取优化**
   - 如何设计更好的 Atom 提取提示
   - 如何评估 Atom 质量

3. **自定义检索策略**
   - 实现新的检索接口
   - 结合图谱等其他知识表示

---

## 📚 相关文件索引

- **检索器实现**: `pikerag/knowledge_retrievers/chunk_atom_retriever.py`
- **Atom 提取提示**: `pikerag/prompts/tagging/atom_question_tagging.py`
- **标注工作流**: `pikerag/workflows/tagging.py`
- **分解工作流**: `pikerag/workflows/qa_decompose.py`
- **数据加载工具**: `pikerag/utils/data_protocol_utils.py`
- **配置模板**: `pikerag/knowledge_retrievers/templates/ChunkAtomRetriever.yml`


