# 菜谱知识图谱 - 使用说明

## 📖 概述

本项目已集成知识图谱功能，将菜谱数据转换为图数据库形式，支持更精确的检索和推荐功能。

## 🏗️ 项目结构

```
/mnt/d/CodeProjects/RecipeRAGSystem/
├── graph_modules/              # 图谱模块
│   ├── __init__.py            # 模块导出
│   ├── neo4j_client.py        # Neo4j客户端
│   ├── data_extractor.py      # 数据提取器
│   ├── graph_builder.py       # 图谱构建器
│   ├── graph_retriever.py     # 图检索器
│   ├── hybrid_retriever.py    # 混合检索器
│   └── utils.py              # 工具函数
├── cypher/                    # Cypher查询模块
│   ├── __init__.py
│   ├── schema.py              # Schema定义
│   └── queries.py             # 常用查询
├── config.py                  # 配置文件（已扩展）
├── main.py                    # 主程序（已集成）
└── test_graph_build.py       # 图谱构建测试脚本
```

## 🗂️ 图谱Schema

### 节点类型（4种）

| 节点 | 属性 | 说明 |
|-----|------|------|
| Recipe | name, difficulty_star | 菜谱 |
| Ingredient | name | 食材 |
| RecipeCategory | name | 分类 |
| DifficultyLevel | level, star_count, description | 难度等级 |

### 关系类型（4种）

| 关系类型 | 起点 | 终点 | 属性 | 说明 |
|---------|------|------|------|------|
| REQUIRES | Recipe | Ingredient | amount, unit | 菜谱-食材关系 |
| CONTAINS_STEP | Recipe | CookingStep | step_order | 菜谱-步骤关系 |
| HAS_DIFFICULTY_LEVEL | Recipe | DifficultyLevel | - | 菜谱-难度关系 |
| BELONGS_TO_CATEGORY | Recipe | RecipeCategory | - | 菜谱-分类关系 |

## 🚀 快速开始

### 1. 准备Neo4j环境

在 `docker-compose.yml` 中添加Neo4j服务：

```yaml
  neo4j:
    container_name: recipe-neo4j
    image: neo4j:5.26.0
    environment:
      - NEO4J_AUTH=neo4j/your_password_here
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=512m
      - NEO4J_dbms_memory_pagecache_size=1g
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/neo4j/data:/data
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/neo4j/logs:/logs
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "your_password_here", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - milvus
```

启动服务：
```bash
docker-compose up -d neo4j
```

### 2. 配置环境变量

在项目根目录的 `.env` 文件中添加：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

或直接修改 `config.py` 中的配置。

### 3. 运行图谱构建测试

```bash
python test_graph_build.py
```

### 4. 使用主程序

```bash
python main.py
```

在交互式界面中：
1. 选择是否构建知识图谱（输入 `y`）
2. 输入问题进行查询

## 🔍 使用示例

### 通过代码使用

```python
from main import RecipeRAGSystem
from config import DEFAULT_CONFIG

# 创建系统实例
system = RecipeRAGSystem(DEFAULT_CONFIG)
system.initialize_system()
system.build_knowledge_base()

# 构建知识图谱
system.build_knowledge_graph()

# 按食材搜索
results = system.search_by_ingredients(['土豆', '洋葱'])
print(results)

# 查找相似菜谱
similar = system.search_similar_recipes('清蒸鲈鱼')
print(similar)

# 获取菜谱详情
details = system.get_recipe_graph_details('红烧肉')
print(details)
```

### 查询语法

在交互式界面中，可以使用以下语法：

- **按食材查找**: `按食材查找 土豆,洋葱`
- **查找相似菜谱**: `相似菜谱 清蒸鲈鱼`
- **普通问题**: 直接输入问题，如"红烧肉怎么做？"

## 📊 数据规模

基于323个菜谱的预估规模：

| 节点类型 | 预估数量 |
|---------|---------|
| Recipe | 323 |
| Ingredient | ~200-300 |
| RecipeCategory | 9 |
| DifficultyLevel | 5 |
| CookingStep | ~1500-2000 |

| 关系类型 | 预估数量 |
|---------|---------|
| REQUIRES | ~1500-2000 |
| CONTAINS_STEP | ~1500-2000 |
| HAS_DIFFICULTY_LEVEL | 323 |
| BELONGS_TO_CATEGORY | 323 |

## 🛠️ 模块说明

### Neo4jClient
Neo4j数据库客户端，提供连接管理、查询执行、事务处理等功能。

### RecipeDataExtractor
从Markdown菜谱文件中提取结构化数据，包括：
- 菜名、分类、难度
- 食材列表及用量
- 操作步骤

### GraphBuilder
构建知识图谱，包括：
- 创建Schema（约束和索引）
- 批量创建节点
- 批量创建关系

### GraphRetriever
图检索器，提供：
- 按食材查找菜谱
- 按分类/难度筛选
- 查找相似菜谱
- 获取菜谱详情

### HybridRetriever
混合检索器，结合：
- 图检索（精确筛选）
- 向量检索（语义理解）
- 结果融合（加权平均）

## ⚙️ 配置说明

在 `config.py` 中可以配置：

```python
# 启用/禁用图谱
enable_graph: bool = True

# Neo4j连接配置
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "password"

# 混合检索权重
retrieval_weight: Dict[str, float] = {
    "vector": 0.7,  # 向量检索权重
    "graph": 0.3    # 图检索权重
}
```

## 🔧 常用Cypher查询

### 查找包含指定食材的菜谱
```cypher
MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
WHERE i.name IN ['土豆', '洋葱']
RETURN r.name, r.difficulty_star
```

### 查找相似菜谱
```cypher
MATCH (r1:Recipe {name: '红烧肉'})-[:REQUIRES]->(i:Ingredient)<-[:REQUIRES]-(r2:Recipe)
WITH r2, count(i) AS common_ingredients
RETURN r2.name, common_ingredients
ORDER BY common_ingredients DESC
```

### 按分类筛选
```cypher
MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory {name: '荤菜'})
RETURN r.name, r.difficulty_star
```

## 📝 注意事项

1. **首次运行**: 首次运行时需要构建知识图谱，可能需要几分钟
2. **数据更新**: 如果菜谱数据有更新，需要重新构建图谱
3. **密码安全**: 请将Neo4j密码保存在环境变量中，不要硬编码
4. **资源限制**: Neo4j内存配置可以根据实际情况调整

## 🐛 故障排查

### Neo4j连接失败
- 检查Docker容器是否运行: `docker ps | grep neo4j`
- 检查端口是否正确: `netstat -an | grep 7687`
- 检查用户名和密码是否正确

### 图谱构建失败
- 检查数据路径是否正确
- 查看日志文件: `logs/recipe_rag.log`
- 检查Markdown文件格式是否正确

### 检索结果为空
- 检查图谱是否已构建
- 使用Neo4j Browser查看图谱数据
- 检查查询的食材名称是否正确

## 📚 进一步学习

- [Neo4j官方文档](https://neo4j.com/docs/)
- [Cypher查询语言](https://neo4j.com/docs/cypher-manual/)
- [LangChain文档](https://python.langchain.com/)

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。
