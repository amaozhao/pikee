# PIKE-RAG 基础 QA 工作流详解

## 📋 整体流程图

```
用户执行命令
    ↓
python examples/qa.py examples/hotpotqa/configs/qa_chunk.yml
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 第一阶段：初始化（QaWorkflow.__init__）                      │
├─────────────────────────────────────────────────────────────┤
│ 1. 加载 YAML 配置                                            │
│ 2. 初始化日志系统                                            │
│ 3. 加载测试数据集                                            │
│ 4. 初始化 Agent 组件：                                       │
│    ├── QA 提示协议（Protocol）                               │
│    ├── 知识检索器（Retriever）                               │
│    └── LLM 客户端（Client）                                 │
│ 5. 初始化评估器（Evaluator）                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二阶段：执行（QaWorkflow.run）                            │
├─────────────────────────────────────────────────────────────┤
│ 对每个测试问题循环执行：                                      │
│    ↓                                                        │
│  【QaWorkflow.answer 方法】                                 │
│    ├── Step 1: 检索相关文档（Retriever）                    │
│    ├── Step 2: 构建提示（Protocol.process_input）          │
│    ├── Step 3: LLM 生成回答（Client.generate）             │
│    ├── Step 4: 解析输出（Protocol.parse_output）           │
│    └── Step 5: 评估答案（Evaluator）                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 第三阶段：输出结果                                           │
├─────────────────────────────────────────────────────────────┤
│ 1. 保存所有 QA 结果到 JSONL 文件                            │
│ 2. 生成评估指标报告（EM, F1, Precision, Recall）            │
│ 3. 保存日志文件                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 详细代码调用流程

### **阶段 1：程序启动和配置加载**

#### Step 1.1: 入口点 - `examples/qa.py`

```python
if __name__ == "__main__":
    # 1. 解析命令行参数，获取配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="yaml配置文件路径")
    args = parser.parse_args()
    
    # 2. 加载并处理 YAML 配置
    yaml_config: dict = load_yaml_config(args.config, args)
    # 这一步会：
    #   - 读取 YAML 文件
    #   - 创建日志目录（如 logs/hotpotqa/naive_rag/）
    #   - 设置各种路径
    
    # 3. 加载环境变量（Azure API keys等）
    load_dot_env(env_path=yaml_config.get("dotenv_path", None))
    
    # 4. 动态导入工作流类
    # 从 yaml_config["workflow"]["module_path"] 导入
    # 对于基础 QA: pikerag.workflows.qa.QaWorkflow
    workflow_module = importlib.import_module(yaml_config["workflow"]["module_path"])
    workflow_class = getattr(workflow_module, yaml_config["workflow"]["class_name"])
    
    # 5. 实例化工作流（这会触发所有初始化）
    workflow = workflow_class(yaml_config)
    
    # 6. 运行工作流
    workflow.run()
```

**关键点：** 使用 Python 的动态导入机制，可以通过配置文件灵活切换不同的工作流。

---

### **阶段 2：工作流初始化 - `QaWorkflow.__init__`**

初始化过程包含以下步骤：

```python
def __init__(self, yaml_config: dict) -> None:
    self._yaml_config: dict = yaml_config
    
    # Step 2.1: 初始化日志系统
    self._init_logger()
    
    # Step 2.2: 加载测试数据集
    self._load_testing_suite()
    
    # Step 2.3: 初始化 Agent（核心组件）
    self._init_agent()
    
    # Step 2.4: 初始化评估器
    self._init_evaluator()
    
    # Step 2.5: 初始化 QA 指标表
    self._init_qas_metrics_table()
```

---

#### **Step 2.2: 加载测试数据集** (`_load_testing_suite`)

```python
def _load_testing_suite(self) -> None:
    # 1. 动态导入数据加载函数
    #    对于 HotpotQA: pikerag.utils.data_protocol_utils.load_testing_suite
    test_loading_module = importlib.import_module(
        self._yaml_config["test_loading"]["module"]
    )
    test_loading_func = getattr(
        test_loading_module, 
        self._yaml_config["test_loading"]["name"]
    )
    
    # 2. 调用加载函数，传入参数（如文件路径）
    #    args: {'filepath': 'data/hotpotqa/dev_500.jsonl'}
    self._testing_suite: List[BaseQaData] = test_loading_func(
        **self._yaml_config["test_loading"]["args"]
    )
    
    # 3. 验证数据类型
    assert isinstance(self._testing_suite[0], BaseQaData)
    
    # 4. 记录测试数据数量
    self._num_test: int = len(self._testing_suite)
```

**数据加载函数实际执行：** `pikerag/utils/data_protocol_utils.py`

```python
def load_testing_suite(filepath: str) -> List[GenerationQaData]:
    testing_suite = []
    with jsonlines.open(filepath, "r") as reader:
        for qa in reader:
            # 每一行 JSONL 数据格式：
            # {
            #   "id": "5a7a06935542990198eaf050",
            #   "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            #   "answer_labels": ["Arthur's Magazine"],
            #   "question_type": "comparison",
            #   "metadata": {...}
            # }
            
            testing_suite.append(
                GenerationQaData(
                    question=qa["question"],
                    answer_labels=[str(label) for label in qa["answer_labels"]],
                    metadata=qa["metadata"],
                )
            )
    return testing_suite
```

**数据结构：** `GenerationQaData`

```python
@dataclass
class GenerationQaData(BaseQaData):
    # 标准答案列表（用于评估）
    answer_labels: List[str] = field(default_factory=lambda: [])
    
    # 模型生成的答案
    answer: str = field(default_factory=lambda: "")
    
    def __post_init__(self) -> None:
        # 对答案进行标准化处理（小写、去标点等）
        self.answer_labels = [normalize_answer(answer) for answer in self.answer_labels]
```

---

#### **Step 2.3: 初始化 Agent** (`_init_agent`)

这是最核心的部分，包含三个子组件：

```python
def _init_agent(self) -> None:
    """初始化 Agent 的三大核心组件"""
    
    # 2.3.1: 初始化 QA 通信协议（提示模板）
    self._init_protocol()
    
    # 2.3.2: 初始化知识检索器
    self._init_retriever()
    
    # 2.3.3: 初始化 LLM 客户端
    self._init_llm_client()
```

##### **2.3.1: 初始化 QA 协议** (`_init_protocol`)

```python
def _init_protocol(self) -> None:
    # 从配置加载协议
    # 对于基础 QA: pikerag.prompts.qa.generation_qa_with_reference_protocol
    self._qa_protocol = load_protocol(
        module_path=self._yaml_config["qa_protocol"]["module_path"],
        protocol_name=self._yaml_config["qa_protocol"]["attr_name"],
        partial_values=self._yaml_config["qa_protocol"].get("template_partial", {}),
    )
```

**协议对象包含：**

```python
generation_qa_with_reference_protocol = CommunicationProtocol(
    template=generation_qa_with_reference_template,  # 提示模板
    parser=GenerationQaParser(),                      # 输入/输出解析器
)
```

**提示模板内容：**

```python
generation_qa_with_reference_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),  # 系统提示
        ("user", """
# Task
Your task is to answer a question referring to a given context, if any.
For answering the Question at the end, you need to first read the context provided, 
then give your final answer.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.
{{
    "answer": <A string. Your Answer.>,
    "rationale": <A string. Rationale behind your choice>
}}

# Context, if any
{context_if_any}

# Question
{content}{yes_or_no_limit}

Let's think step by step.
""".strip()),
    ],
    input_variables=["content", "context_if_any", "yes_or_no_limit"],
    partial_variables={"system_prompt": DEFAULT_SYSTEM_PROMPT},
)
```

---

##### **2.3.2: 初始化检索器** (`_init_retriever`)

```python
def _init_retriever(self) -> None:
    retriever_config: dict = self._yaml_config["retriever"]
    
    # 1. 动态加载检索器类
    # 对于基础 QA: pikerag.knowledge_retrievers.QaChunkRetriever
    retriever_class = load_class(
        module_path=retriever_config["module_path"],
        class_name=retriever_config["class_name"],
        base_class=BaseQaRetriever
    )
    
    # 2. 实例化检索器
    self._retriever = retriever_class(
        retriever_config=retriever_config["args"],
        log_dir=self._yaml_config["log_dir"],
        main_logger=self._logger,
    )
```

**检索器初始化过程：** `QaChunkRetriever.__init__`

```python
def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
    super().__init__(retriever_config, log_dir, main_logger)
    
    # Step A: 初始化查询解析器
    # 默认：question_as_query（直接用问题作为查询）
    self._init_query_parser()
    
    # Step B: 加载向量数据库
    # 这会加载预先构建好的 Chroma 向量存储
    self._load_vector_store()
    
    # Step C: 初始化 Chroma 混入（提供向量检索方法）
    self._init_chroma_mixin()
```

**向量数据库加载：**

```python
def _load_vector_store(self) -> None:
    vector_store_config = self._retriever_config["vector_store"]
    
    # 调用辅助函数加载向量存储
    self.vector_store: Chroma = load_vector_store_from_configs(
        vector_store_config=vector_store_config,
        embedding_config=vector_store_config.get("embedding_setting", {}),
        collection_name=vector_store_config.get("collection_name", self.name),
        persist_directory=vector_store_config.get("persist_directory"),
    )
```

```python
def load_vector_store_from_configs(...) -> Chroma:
    # 1. 加载 Embedding 函数（如 Azure OpenAI Embedding）
    embedding = load_embedding_func(
        module_path=embedding_config.get("module_path"),
        class_name=embedding_config.get("class_name"),
        **embedding_config.get("args", {}),
    )
    
    # 2. 加载文档数据
    # 调用：pikerag.utils.data_protocol_utils.load_ids_and_chunks
    loading_configs = vector_store_config["id_document_loading"]
    ids, documents = load_callable(...)(
        **loading_configs.get("args", {})
    )
    # 返回：
    #   ids: ["chunk_0001", "chunk_0002", ...]
    #   documents: [Document(page_content=..., metadata=...), ...]
    
    # 3. 构建或加载向量存储
    vector_store = load_vector_store(
        collection_name, persist_directory, 
        embedding, documents, ids, exist_ok
    )
    return vector_store
```

---

##### **2.3.3: 初始化 LLM 客户端** (`_init_llm_client`)

```python
def _init_llm_client(self) -> None:
    # 1. 创建客户端日志器
    self._client_logger = Logger(
        name="client", 
        dump_mode="a",  # 追加模式
        dump_folder=self._yaml_config["log_dir"]
    )
    
    llm_client_config = self._yaml_config["llm_client"]
    
    # 2. 动态导入 LLM 客户端类
    # 对于 Azure OpenAI: pikerag.llm_client.AzureOpenAIClient
    client_module = importlib.import_module(llm_client_config["module_path"])
    client_class = getattr(client_module, llm_client_config["class_name"])
    
    # 3. 提取 LLM 配置
    self.llm_config = llm_client_config["llm_config"]
    # 例如: {"model": "gpt-4", "temperature": 0}
    
    # 4. 实例化客户端
    self._client = client_class(
        location=None,  # 缓存位置稍后设置
        auto_dump=llm_client_config["cache_config"]["auto_dump"],
        logger=self._client_logger,
        llm_config=self.llm_config,
        **llm_client_config.get("args", {}),
    )
```

**LLM 客户端特性：**
- **缓存机制：** 相同的输入不会重复调用 API，节省成本
- **统一接口：** 支持多种 LLM 提供商（Azure OpenAI、HuggingFace等）
- **日志记录：** 所有 LLM 请求和响应都会被记录

---

### **阶段 3：执行问答流程 - `QaWorkflow.run`**

初始化完成后，开始执行测试：

```python
def _single_thread_run(self) -> None:
    # 1. 打开输出文件（用于保存所有 QA 结果）
    fout = jsonlines.open(self._yaml_config["test_jsonl_path"], "w")
    
    # 2. 对每一轮测试执行（通常是1轮）
    for round_idx in range(self._yaml_config["test_rounds"]):
        round_id: str = f"Round{round_idx}"
        
        # 2.1: 更新 LLM 缓存位置
        self._update_llm_cache(round_idx)
        
        # 2.2: 通知评估器：新一轮开始
        self._evaluator.on_round_test_start(round_id)
        
        # 2.3: 遍历所有测试问题
        question_idx: int = 0
        pbar = tqdm(self._testing_suite, desc=f"...")
        for qa in pbar:
            # 【核心】调用 answer 方法回答问题
            output_dict: dict = self.answer(qa, question_idx)
            
            # 2.4: 更新 QA 数据对象
            answer = output_dict.pop("answer")
            qa.update_answer(answer)  # 设置生成的答案
            qa.answer_metadata.update(output_dict)  # 保存其他元数据
            
            # 2.5: 评估答案质量
            self._evaluator.update_round_metrics(qa)
            
            # 2.6: 保存结果
            fout.write(qa.as_dict())
            self._update_qas_metrics_table(qa)
            
            question_idx += 1
            
            # 2.7: 更新进度条（显示实时指标）
            self._update_pbar_desc(pbar, round_idx=round_idx, count=question_idx)
        
        # 2.8: 一轮结束
        self._evaluator.on_round_test_end(round_id)
    
    # 3. 所有测试结束
    self._evaluator.on_test_end()
    fout.close()
```

---

### **阶段 4：核心答题逻辑 - `QaWorkflow.answer`** ⭐⭐⭐

这是最核心的方法，处理单个问题的完整流程：

```python
def answer(self, qa: BaseQaData, question_idx: int) -> dict:
    """给定一个问题，执行决策过程生成答案
    
    这里实现的是：单次 LLM 调用 + 可选的检索增强
    """
    
    # ========== Step 1: 检索相关文档 ==========
    reference_chunks: List[str] = self._retriever.retrieve_contents(
        qa, 
        retrieve_id=f"Q{question_idx:03}"
    )
    
    # ========== Step 2: 构建提示消息 ==========
    messages = self._qa_protocol.process_input(
        content=qa.question,
        references=reference_chunks,
        **qa.as_dict()
    )
    
    # ========== Step 3: LLM 生成内容 ==========
    response = self._client.generate_content_with_messages(
        messages, 
        **self.llm_config
    )
    
    # ========== Step 4: 解析 LLM 输出 ==========
    output_dict: dict = self._qa_protocol.parse_output(
        response, 
        **qa.as_dict()
    )
    
    # ========== Step 5: 添加元数据 ==========
    if "response" not in output_dict:
        output_dict["response"] = response
    
    if "reference_chunks" not in output_dict:
        output_dict["reference_chunks"] = reference_chunks
    
    return output_dict
```

---

## 📖 详细步骤拆解

### **Step 1: 检索相关文档**

```python
reference_chunks: List[str] = self._retriever.retrieve_contents(
    qa, retrieve_id=f"Q{question_idx:03}"
)
```

**检索器执行流程：** `QaChunkRetriever.retrieve_contents`

```python
def retrieve_contents(self, qa: BaseQaData, retrieve_id: str="") -> List[str]:
    # A. 查询解析：将 QA 对象转换为查询字符串
    queries: List[str] = self._query_parser(qa)
    # 对于 question_as_query: 返回 [qa.question]
    
    # B. 计算每个查询的 top-k
    retrieve_k = math.ceil(self.retrieve_k / len(queries))
    # 如果配置 retrieve_k=16，只有1个查询，那么 retrieve_k=16
    
    # C. 对每个查询执行向量检索
    all_chunks: List[str] = []
    for query in queries:
        chunks = self.retrieve_contents_by_query(
            query, retrieve_id, retrieve_k=retrieve_k
        )
        all_chunks.extend(chunks)
    
    # D. 记录日志
    self.logger.debug(f"{retrieve_id}: {len(all_chunks)} strings returned.")
    
    return all_chunks
```

**向量检索细节：**

```python
def retrieve_contents_by_query(self, query: str, ...) -> List[str]:
    # 1. 执行向量相似度检索
    chunk_infos = self._get_doc_and_score_with_query(query, ...)
    # 返回: [(Document对象, 相似度分数), ...]
    
    # 2. 提取文档内容
    return self._get_relevant_strings(chunk_infos, retrieve_id)
    # 返回: ["chunk内容1", "chunk内容2", ...]
```

**检索结果示例：**
```python
reference_chunks = [
    "Arthur's Magazine was an American literary periodical published from 1844 to 1846...",
    "First for Women is a woman's magazine published by Bauer Media Group...",
    # ... 更多相关文档片段
]
```

---

### **Step 2: 构建提示消息**

```python
messages = self._qa_protocol.process_input(
    content=qa.question,
    references=reference_chunks,
    **qa.as_dict()
)
```

**协议处理输入：** `GenerationQaParser.encode`

```python
def encode(self, content: str, references: List[str]=[], 
           context_len_limit: int=80000, **kwargs) -> Tuple[str, dict]:
    # A. 构建 yes/no 限制指令
    answer_labels = kwargs.get("answer_labels", [])
    if len(answer_labels) == 1 and answer_labels[0] in ["yes", "no"]:
        yes_or_no_limit = """ Your answer shall be "Yes" or "No"."""
    else:
        yes_or_no_limit = ""
    
    # B. 构建参考上下文
    context_if_any = ""
    for context in list(set(references)):  # 去重
        context_if_any += f"\n{context}\n"
        if len(context_if_any) >= context_len_limit:  # 防止超长
            break
    
    # C. 返回问题和模板变量
    return content, {
        "yes_or_no_limit": yes_or_no_limit,
        "context_if_any": context_if_any,
    }
```

**生成的消息结构：**
```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant on question answering."
    },
    {
        "role": "user",
        "content": """
# Task
Your task is to answer a question referring to a given context, if any.
...

# Context, if any

Arthur's Magazine was an American literary periodical published from 1844 to 1846...

First for Women is a woman's magazine published by Bauer Media Group...

# Question
Which magazine was started first Arthur's Magazine or First for Women?

Let's think step by step.
"""
    }
]
```

---

### **Step 3: LLM 生成内容**

```python
response = self._client.generate_content_with_messages(
    messages, **self.llm_config
)
```

**LLM 客户端执行：**
1. 检查缓存：如果相同的 messages 之前调用过，直接返回缓存结果
2. 如果没有缓存，调用 Azure OpenAI API
3. 记录请求和响应到日志
4. 保存到缓存
5. 返回生成的文本

**LLM 响应示例：**
```json
{
    "answer": "Arthur's Magazine",
    "rationale": "Arthur's Magazine was published from 1844 to 1846, while First for Women was founded later. Therefore, Arthur's Magazine was started first."
}
```

---

### **Step 4: 解析 LLM 输出**

```python
output_dict: dict = self._qa_protocol.parse_output(
    response, **qa.as_dict()
)
```

**解析器执行：** `GenerationQaParser.decode`

```python
def decode(self, content: str, **kwargs) -> Dict[str, str]:
    try:
        # 尝试解析 JSON
        output = parse_json(content)
        # parse_json 会处理各种格式问题，如：
        #   - Markdown 代码块包裹的 JSON
        #   - 注释
        #   - 不标准的引号等
    except Exception as e:
        print(f"[GenerationQaParser] Exception: {e}")
        return {
            "answer": "parsing error",
            "rationale": "parsing error",
        }
    
    # 确保所有值都是字符串
    for key, value in output.items():
        output[key] = str(value)
    
    return output
```

**解析结果：**
```python
output_dict = {
    "answer": "Arthur's Magazine",
    "rationale": "Arthur's Magazine was published from 1844..."
}
```

---

### **Step 5: 添加元数据并返回**

```python
# 保存原始 LLM 响应
if "response" not in output_dict:
    output_dict["response"] = response

# 保存检索到的文档
if "reference_chunks" not in output_dict:
    output_dict["reference_chunks"] = reference_chunks

return output_dict
```

**最终返回的字典：**
```python
{
    "answer": "Arthur's Magazine",
    "rationale": "...",
    "response": "{\"answer\": \"Arthur's Magazine\", \"rationale\": \"...\"}",
    "reference_chunks": ["chunk1", "chunk2", ...]
}
```

---

### **阶段 5：评估答案**

回到 `run` 方法，answer 返回后：

```python
# 1. 提取答案并更新 QA 对象
answer = output_dict.pop("answer")
qa.update_answer(answer)  # 设置 qa.answer = "Arthur's Magazine"

# 2. 保存其他元数据
qa.answer_metadata.update(output_dict)

# 3. 评估答案
self._evaluator.update_round_metrics(qa)
# 这会计算：
#   - ExactMatch: qa.answer == qa.answer_labels[0] ? 1 : 0
#   - F1: 计算 token 级别的 F1 分数
#   - Precision, Recall 等
```

**评估结果保存在 `qa.answer_metric_scores` 中：**
```python
qa.answer_metric_scores = {
    "ExactMatch": 1.0,
    "F1": 1.0,
    "Precision": 1.0,
    "Recall": 1.0,
}
```

---

## 📊 完整数据流示意图

```
输入: "Which magazine was started first Arthur's Magazine or First for Women?"
    ↓
【检索器】→ 向量数据库查询
    ↓
检索结果: [
    "Arthur's Magazine was published from 1844...",
    "First for Women is a magazine...",
    ...
]
    ↓
【协议处理】→ 构建提示
    ↓
提示消息: {
    system: "You are a helpful AI assistant...",
    user: "# Context\n...\n# Question\n..."
}
    ↓
【LLM 客户端】→ 调用 GPT-4
    ↓
LLM 响应: {"answer": "Arthur's Magazine", "rationale": "..."}
    ↓
【协议解析】→ JSON 解析
    ↓
输出字典: {"answer": "Arthur's Magazine", "rationale": "...", ...}
    ↓
【评估器】→ 对比标准答案
    ↓
指标: {ExactMatch: 1.0, F1: 1.0, ...}
    ↓
保存到 JSONL 文件
```

---

## 🎯 关键设计模式和优势

### 1. **动态导入机制**
```python
# 所有组件都通过配置文件动态加载
workflow_class = getattr(importlib.import_module(path), name)
```
**优势：** 无需修改代码，只需修改 YAML 配置即可切换不同的检索器、LLM、工作流等。

### 2. **协议模式（Protocol Pattern）**
```python
# 提示模板 + 输入输出解析器 = 通信协议
protocol = CommunicationProtocol(template, parser)
```
**优势：** 提示工程模块化，易于测试和复用。

### 3. **缓存机制**
```python
# LLM 客户端自动缓存相同输入的响应
self._client.generate_content_with_messages(messages, ...)
```
**优势：** 节省成本，加速调试（重复运行不会重复调用 API）。

### 4. **日志系统**
```python
# 多层次的日志记录
- 主日志器：记录工作流事件
- 客户端日志器：记录所有 LLM 请求
- 检索器日志器：记录检索详情
```
**优势：** 完整的可追溯性，便于调试和分析。

---

## 📝 总结

基础 QA 工作流的核心就是：

```
问题 → 检索 → 提示构建 → LLM生成 → 解析 → 评估
```

虽然看起来简单，但 PIKE-RAG 的优势在于：
1. **模块化设计**：每个组件都可以独立替换
2. **配置驱动**：通过 YAML 灵活配置
3. **智能检索**：支持多种检索策略
4. **完善的日志**：所有过程可追溯
5. **缓存优化**：避免重复调用，节省成本

---

## 🚀 配置文件示例

完整的 YAML 配置文件结构：

```yaml
# 实验设置
experiment_name: naive_rag
log_root_dir: logs/hotpotqa
test_rounds: 1

# 工作流配置
workflow:
  module_path: pikerag.workflows.qa
  class_name: QaWorkflow

# 测试数据
test_loading:
  module: pikerag.utils.data_protocol_utils
  name: load_testing_suite
  args:
    filepath: data/hotpotqa/dev_500.jsonl

# LLM 配置
llm_client:
  module_path: pikerag.llm_client
  class_name: AzureOpenAIClient
  llm_config:
    model: gpt-4
    temperature: 0
  cache_config:
    auto_dump: True

# 检索器配置
retriever:
  module_path: pikerag.knowledge_retrievers
  class_name: QaChunkRetriever
  args:
    retrieve_k: 16
    retrieve_score_threshold: 0.2
    vector_store:
      collection_name: dev_500_chunks_ada
      persist_directory: data/vector_stores/hotpotqa

# 提示模板配置
qa_protocol:
  module_path: pikerag.prompts.qa
  attr_name: generation_qa_with_reference_protocol

# 评估指标
evaluator:
  metrics:
    - ExactMatch
    - F1
    - Precision
    - Recall
```

---

## 📚 相关文件索引

- **入口点**: `examples/qa.py`
- **工作流**: `pikerag/workflows/qa.py`
- **检索器**: `pikerag/knowledge_retrievers/chroma_qa_retriever.py`
- **LLM 客户端**: `pikerag/llm_client/azure_open_ai_client.py`
- **提示模板**: `pikerag/prompts/qa/generation.py`
- **数据加载**: `pikerag/utils/data_protocol_utils.py`
- **评估器**: `pikerag/workflows/evaluation/evaluator.py`


