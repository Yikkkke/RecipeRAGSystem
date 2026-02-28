# RecipeRAG 评估框架

## 1. 概述

RecipeRAG 系统的评估框架，用于评估检索质量和生成质量。

## 2. 功能特性

- **自动生成测试数据**：从菜谱数据中自动采样生成测试query
- **检索质量评估**：使用LLM判断文档相关性，计算Precision@K、MRR等指标
- **生成质量评估**：使用LLM评估答案的完整性、准确性、实用性
- **测试数据管理**：支持测试数据持久化，用户可编辑后重新评估
- **完整日志记录**：记录评估过程和详细结果

## 3. 目录结构

```
evaluation/
├── __init__.py                  # 模块导出
├── README.md                    # 使用说明文档
├── run_evaluation.py            # 评估运行脚本
├── logger.py                    # 日志配置模块
├── test_data_manager.py         # 测试数据管理器
├── retrieval_evaluator.py       # 检索质量评估
├── generation_evaluator.py      # 生成质量评估
└── evaluator.py                 # 主评估器
```

```
reports/
└── test_data/                   # 测试数据目录
    └── test_data_*.json         # 测试数据文件
```

```
logs/
└── evaluation/                  # 评估日志目录
    ├── eval_YYYYMMDD_HHMMSS.log # 评估日志
    └── eval_detail_*.json       # 详细评估结果
```

## 4. 快速开始

### 4.1 命令行运行（推荐）

```bash
# 进入项目目录
cd /path/to/RecipeRAGSystem

# 激活虚拟环境
source .venv/bin/activate

# 设置代理（如果需要）
export http_proxy=http://$(ip route | awk '/default/ {print $3}'):7890
export https_proxy=$http_proxy

# 运行评估 - 自动生成测试数据（默认10条）
python evaluation/run_evaluation.py

# 运行评估 - 指定测试数据文件
python evaluation/run_evaluation.py --test-file test_data_20260227_225939.json

# 运行评估 - 指定样本数量
python evaluation/run_evaluation.py --sample-size 20

# 运行评估 - 指定输出文件名
python evaluation/run_evaluation.py --test-file test_data_v1.json --output my_report

# 运行评估 - 指定检索文档数量
python evaluation/run_evaluation.py --top-k 5

# 查看帮助
python evaluation/run_evaluation.py --help
```

**命令行参数说明：**

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--test-file` | 测试数据文件名，不指定则自动生成 | 自动生成 |
| `--sample-size` | 生成的测试用例数量 | 10 |
| `--output` | 评估报告输出文件名 | 自动生成 |
| `--top-k` | 检索的文档数量 | 3 |

### 4.2 Python API 使用

```python
from main import RecipeRAGSystem
from evaluation import RecipeRAGEvaluator

# 1. 初始化RAG系统
rag = RecipeRAGSystem()
rag.initialize_system()
rag.build_knowledge_base()

# 2. 初始化评估器
evaluator = RecipeRAGEvaluator(rag)

# 3. 生成测试数据（首次使用）
# 从菜谱数据中自动生成20条测试query并保存
evaluator.generate_test_data(
    data_path="data/cook",
    sample_size=20,
    filename="test_data_v1.json"
)

# 4. 运行评估
results = evaluator.run_evaluation(
    test_file="test_data_v1.json",
    output_file="eval_report_v1.json"
)
```

### 4.3 编辑测试数据

生成的测试数据保存在 `reports/test_data/test_data_v1.json`，格式如下：

```json
[
  {
    "id": "q1",
    "text": "宫保鸡丁怎么做",
    "type": "detail",
    "expected_docs": ["宫保鸡丁"],
    "filters": null
  },
  {
    "id": "q2",
    "text": "素菜有哪些推荐",
    "type": "list",
    "expected_docs": null,
    "filters": null
  },
  {
    "id": "q3",
    "text": "简单的荤菜",
    "type": "with_filter",
    "expected_docs": null,
    "filters": {"difficulty": "简单", "category": "荤菜"}
  }
]
```

用户可编辑字段说明：
- `id`: 测试用例唯一标识
- `text`: 测试问题文本
- `type`: 问题类型
  - `detail`: 具体菜谱做法（如"宫保鸡丁怎么做"）
  - `list`: 推荐类查询（如"素菜有哪些"）
  - `with_filter`: 带过滤条件的查询（如"简单的荤菜"）
  - `meta`: 系统元问题
  - `general`: 一般性问题
- `expected_docs`: 期望检索到的文档（可选，用于人工验证）
- `filters`: 过滤条件（可选，如`{"category": "荤菜", "difficulty": "简单"}`）

### 4.4 使用编辑后的数据重新评估

```python
# 编辑测试数据后，重新运行评估
results = evaluator.run_evaluation(
    test_file="test_data_v1.json",  # 使用编辑后的文件
    output_file="eval_report_v2.json"
)
```

### 4.5 列出已有测试数据

```python
# 列出所有测试数据文件
files = evaluator.list_test_files()
print(files)  # ['test_data_20260227_143000.json', ...]
```

## 5. 评估指标

### 5.1 检索质量

| 指标 | 说明 |
|-----|------|
| Precision@K | 检索结果中相关文档的比例 |
| MRR | 第一个相关文档排名的倒数平均值 |

### 5.2 生成质量

| 维度 | 说明 | 分数 |
|-----|------|------|
| 完整性 | 是否完整回答了问题 | 1-5分 |
| 准确性 | 是否与上下文一致 | 1-5分 |
| 实用性 | 对用户是否有帮助 | 1-5分 |

## 6. 输出文件

### 6.1 评估报告 (JSON)

`reports/eval_report_*.json`:

```json
{
  "generated_at": "2026-02-27T14:30:00",
  "summary": {
    "test_file": "test_data_v1.json",
    "sample_size": 20,
    "retrieval": {
      "avg_precision": 0.85,
      "avg_mrr": 0.78
    },
    "generation": {
      "avg_completeness": 4.2,
      "avg_accuracy": 4.5,
      "avg_usefulness": 4.0,
      "avg_score": 4.23
    }
  },
  "by_type": {
    "detail": {
      "count": 10,
      "retrieval": {"avg_precision": 0.82, "avg_mrr": 0.75},
      "generation": {"avg_score": 4.4}
    },
    "list": {
      "count": 5,
      "retrieval": {"avg_precision": 0.90, "avg_mrr": 0.85},
      "generation": {"avg_score": 3.8}
    }
  },
  "details": [...]
}
```

### 6.2 评估日志

`logs/evaluation/eval_YYYYMMDD_HHMMSS.log`:

```
2026-02-27 14:30:00 - INFO - ==================================================
2026-02-27 14:30:00 - INFO - 开始生成测试数据
2026-02-27 14:30:00 - INFO - ==================================================
2026-02-27 14:30:00 - INFO - 测试用例 1: [detail] 宫保鸡丁怎么做
...
2026-02-27 14:30:15 - INFO - 
--- 检索评估: 宫保鸡丁怎么做
2026-02-27 14:30:15 - INFO -   检索到 3 个文档
2026-02-27 14:30:15 - INFO -     文档1: 宫保鸡丁 - 相关
2026-02-27 14:30:15 - INFO -   Precision@K: 0.67, MRR: 1.00
...
```

### 6.3 详细结果 (JSON)

`logs/evaluation/eval_detail_*.json`:

```json
[
  {
    "query_id": "q1",
    "query_text": "宫保鸡丁怎么做",
    "query_type": "detail",
    "retrieval_scores": {"precision": 0.67, "mrr": 1.0},
    "generation_scores": {"completeness": 4.5, "accuracy": 4.5, "usefulness": 4.0},
    "retrieved_docs": [
      {"name": "宫保鸡丁", "is_relevant": true},
      {"name": "辣子鸡", "is_relevant": true}
    ]
  }
]
```

## 7. 使用示例

### 示例1：首次评估

```python
from main import RecipeRAGSystem
from evaluation import RecipeRAGEvaluator

# 初始化
rag = RecipeRAGSystem()
rag.initialize_system()
rag.build_knowledge_base()

# 评估
evaluator = RecipeRAGEvaluator(rag)
results = evaluator.run_evaluation(sample_size=20)
print(results)
```

### 示例2：使用已有测试数据

```python
# 继续使用已生成的测试数据
results = evaluator.run_evaluation(
    test_file="test_data_v1.json",
    output_file="eval_report_v1.json"
)
```

### 示例3：仅生成测试数据

```python
# 只生成测试数据，不运行评估
queries = evaluator.generate_test_data(
    sample_size=30,
    filename="test_data_expanded.json"
)
```

### 示例4：查看测试数据列表

```python
files = evaluator.list_test_files()
for f in files:
    print(f)
```

## 8. 注意事项

1. **API Key**: 确保已设置 `MOONSHOT_API_KEY` 环境变量
2. **知识库**: 评估前需确保已构建知识库（`build_knowledge_base()`）
3. **测试数据**: 测试数据文件保存在 `reports/test_data/` 目录下
4. **评估次数**: 评估会调用LLM多次，请注意API使用量
