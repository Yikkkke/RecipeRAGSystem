# 系统流程

## 整体流程

```
main() → RecipeRAGSystem.run_interactive()
    │
    ├─ initialize_system()
    │   ├─ DataPreparationModule (数据准备)
    │   ├─ IndexConstructionModule (索引构建)
    │   ├─ GenerationIntegrationModule (LLM生成)
    │   └─ Neo4jClient + GraphRetriever + HybridRetriever (图数据库)
    │
    ├─ build_knowledge_base()
    │   ├─ 加载文档
    │   ├─ 分块
    │   ├─ 构建/加载向量索引 (FAISS)
    │   └─ 初始化检索模块
    │
    └─ answer_query() ← 用户查询入口
        ├─ 1. query_router() → 判断查询类型 (detail/list/general)
        ├─ 2. query_rewrite() → 智能查询重写
        ├─ 3. retrieve_documents() → 检索 (核心)
        ├─ 4. get_parent_documents() → 获取完整文档
        └─ 5. 生成回答 (根据路由类型选择生成策略)
```

## 检索策略 (核心)

### 查询路由 → 决定检索方式

| 查询类型 | 判断标准 | 检索策略 |
|---------|---------|---------|
| detail | 具体做法、步骤、食材 | 向量混合检索 |
| list | 推荐、列表类查询 | 图谱精确匹配 |
| general | 知识性问题 | 向量检索 |

### 混合检索实现 (retrieval_optimization.py)

向量 + BM25 混合检索 (RRF重排):
```
hybrid_search(query):
  ├─ 向量检索 (dense) → similarity search
  ├─ BM25检索 (sparse) → keyword search  
  └─ RRF重排 → 1/(k+rank) 加权融合
```

### 检索策略分支 (main.py:466-544)

- enable_graph = True:
  - detail类型: 有filters → 向量检索+元数据过滤; 无filters → 向量混合检索
  - list/general类型: 有filters → 图谱精确匹配; 无filters → 图谱+向量混合检索
- enable_graph = False: 统一使用向量检索

### 图谱检索 (graph_modules/hybrid_retriever.py)

三种检索模式:
- 关键词匹配: 从query提取食材/分类 → 图谱Cypher查询
- 属性过滤: category、difficulty → Neo4j属性筛选
- 混合搜索: 图结果 + 向量结果加权组合

---


# 环境配置(CPU版本)
```bash
# CPU版本
# 将requirements.in编译为requirements.txt
uv pip compile requirements.in \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -o requirements.txt


# 安装虚拟环境
uv pip sync requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

```