"""
RecipeRAG 评估框架主模块
整合检索和生成评估，提供完整的评估流程
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .test_data_manager import TestQuery, TestDataManager
from .retrieval_evaluator import RetrievalEvaluator
from .generation_evaluator import GenerationEvaluator
from .logger import EvaluationLogger

logger = logging.getLogger(__name__)


class RecipeRAGEvaluator:
    """
    RecipeRAG 评估器主类

    整合检索和生成评估，提供完整的评估流程：
    1. 生成或加载测试数据
    2. 对每个测试query执行检索和生成
    3. 评估检索质量和生成质量
    4. 输出评估报告和日志
    """

    def __init__(self, rag_system):
        """
        初始化评估器

        Args:
            rag_system: RecipeRAGSystem实例，需要包含:
                - config: 配置对象
                - retrieval_module: 检索模块
                - generation_module: 生成模块
                - data_module: 数据模块
        """
        self.rag = rag_system

        # 各子模块
        self.test_manager = TestDataManager()
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator()
        self.eval_logger = EvaluationLogger()

        # 报告输出目录
        self.report_dir = Path("reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_test_data(
        self,
        data_path: Optional[str] = None,
        sample_size: int = 20,
        output_file: Optional[str] = None,
    ) -> List[TestQuery]:
        """
        从菜谱数据生成测试query并保存到文件

        Args:
            data_path: 菜谱数据目录路径，默认使用RAG系统的配置
            sample_size: 生成的测试用例数量
            output_file: 输出文件名

        Returns:
            生成的测试查询列表
        """
        if data_path is None:
            data_path = self.rag.config.data_path

        # 生成测试数据
        queries = self.test_manager.generate_from_data(
            data_path=data_path, sample_size=sample_size, output_file=output_file
        )

        # 记录日志
        self.eval_logger.log_test_data_generation(
            [{"text": q.text, "type": q.type} for q in queries]
        )

        return queries

    def load_test_data(self, filename: str) -> List[TestQuery]:
        """
        从文件加载测试数据

        Args:
            filename: 测试数据文件名

        Returns:
            测试查询列表
        """
        queries = self.test_manager.load_test_queries(filename)

        # 记录日志
        self.eval_logger.log_test_data_generation(
            [{"text": q.text, "type": q.type} for q in queries]
        )

        return queries

    def list_test_files(self) -> List[str]:
        """列出所有测试数据文件"""
        return self.test_manager.list_test_files()

    def run_evaluation(
        self,
        test_file: Optional[str] = None,
        sample_size: int = 20,
        output_file: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        运行完整评估流程

        优先顺序：
        1. 如果指定了test_file，从文件加载测试数据
        2. 否则生成新的测试数据并保存

        Args:
            test_file: 测试数据文件名
            sample_size: 生成的测试用例数量（当test_file为None时生效）
            output_file: 评估报告输出文件名
            top_k: 检索和评估的文档数量

        Returns:
            评估结果字典
        """
        # ====== 1. 加载或生成测试数据 ======
        if test_file and Path(f"reports/test_data/{test_file}").exists():
            print(f"从文件加载测试数据: {test_file}")
            test_queries = self.load_test_data(test_file)
            used_test_file = test_file
        else:
            print("生成测试数据...")
            if test_file:
                print(f"  (文件 {test_file} 不存在，将创建新文件)")

            queries = self.generate_test_data(
                data_path=self.rag.config.data_path,
                sample_size=sample_size,
                output_file=test_file,
            )
            test_queries = queries

            # 获取生成的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            used_test_file = test_file or f"test_data_{timestamp}.json"

        # ====== 2. 执行评估 ======
        results = self._evaluate_queries(test_queries, top_k)

        # ====== 3. 生成汇总报告 ======
        summary = self._generate_summary(results, used_test_file)

        # ====== 4. 输出报告 ======
        self._output_report(summary, results, output_file)

        return summary

    def _evaluate_queries(
        self, test_queries: List[TestQuery], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        对每个测试query执行检索和评估 - 复用主系统的完整流程

        Args:
            test_queries: 测试查询列表
            top_k: 检索的文档数量

        Returns:
            评估结果列表
        """
        eval_results = []
        total = len(test_queries)

        for i, query in enumerate(test_queries):
            print(f"\n[{i + 1}/{total}] 评估: {query.text}")

            try:
                # ====== 调用主系统的 answer_query 方法 ======
                result = self.rag.answer_query(query.text, stream=False)

                # 解析返回值：(answer, retrieved_docs, route_type)
                if isinstance(result, tuple) and len(result) == 3:
                    generated_answer, retrieved_docs, route_type = result
                else:
                    # 兼容处理：流式输出等情况
                    generated_answer = str(result) if result else ""
                    retrieved_docs = []
                    route_type = "unknown"

                # ====== 评估检索质量 ======
                retrieval_scores = self.retrieval_eval.evaluate(
                    query, retrieved_docs, top_k, route_type
                )

                # 记录检索结果
                self.eval_logger.log_retrieval_result(
                    query.text, retrieved_docs, retrieval_scores
                )

                # ====== 评估生成质量 ======
                generation_scores = self.generation_eval.evaluate(
                    query, retrieved_docs, generated_answer
                )

                # 记录生成结果
                self.eval_logger.log_generation_result(
                    query.text, generated_answer, generation_scores
                )

                # ====== 保存详情 ======
                detail = {
                    "query_id": query.id,
                    "query_text": query.text,
                    "query_type": route_type,
                    "retrieval_scores": retrieval_scores,
                    "generation_scores": generation_scores,
                    "retrieved_docs": [
                        {
                            "name": doc.metadata.get("dish_name", "未知"),
                            "is_relevant": doc.metadata.get("is_relevant", False),
                        }
                        for doc in retrieved_docs[:top_k]
                    ],
                    "generated_answer": generated_answer,  # 完整保存
                }

                eval_results.append(detail)
                self.eval_logger.add_detail(detail)

            except Exception as e:
                logger.error(f"评估 query '{query.text}' 时出错: {e}")
                eval_results.append(
                    {
                        "query_id": query.id,
                        "query_text": query.text,
                        "query_type": "unknown",
                        "error": str(e),
                    }
                )

        return eval_results

    def _generate_summary(
        self, results: List[Dict[str, Any]], test_file: str
    ) -> Dict[str, Any]:
        """
        生成评估汇总

        Args:
            results: 评估结果列表
            test_file: 使用的测试数据文件名

        Returns:
            汇总信息字典
        """
        # 过滤掉出错的项
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"error": "所有评估都失败了"}

        # 计算检索指标
        precisions = [r["retrieval_scores"]["precision"] for r in valid_results]
        mrrs = [r["retrieval_scores"]["mrr"] for r in valid_results]

        # 计算生成指标
        completeness = [r["generation_scores"]["completeness"] for r in valid_results]
        accuracy = [r["generation_scores"]["accuracy"] for r in valid_results]
        usefulness = [r["generation_scores"]["usefulness"] for r in valid_results]

        # 按类型汇总
        type_stats = {}
        for r in valid_results:
            qtype = r["query_type"]
            if qtype not in type_stats:
                type_stats[qtype] = {
                    "retrieval_precision": [],
                    "retrieval_mrr": [],
                    "generation_completeness": [],
                    "generation_accuracy": [],
                    "generation_usefulness": [],
                }

            type_stats[qtype]["retrieval_precision"].append(
                r["retrieval_scores"]["precision"]
            )
            type_stats[qtype]["retrieval_mrr"].append(r["retrieval_scores"]["mrr"])
            type_stats[qtype]["generation_completeness"].append(
                r["generation_scores"]["completeness"]
            )
            type_stats[qtype]["generation_accuracy"].append(
                r["generation_scores"]["accuracy"]
            )
            type_stats[qtype]["generation_usefulness"].append(
                r["generation_scores"]["usefulness"]
            )

        # 构建汇总
        summary = {
            "summary": {
                "test_file": test_file,
                "sample_size": len(valid_results),
                "failed_count": len(results) - len(valid_results),
                "retrieval": {
                    "avg_precision": sum(precisions) / len(precisions)
                    if precisions
                    else 0,
                    "avg_mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
                },
                "generation": {
                    "avg_completeness": sum(completeness) / len(completeness)
                    if completeness
                    else 0,
                    "avg_accuracy": sum(accuracy) / len(accuracy) if accuracy else 0,
                    "avg_usefulness": sum(usefulness) / len(usefulness)
                    if usefulness
                    else 0,
                    "avg_score": (sum(completeness) + sum(accuracy) + sum(usefulness))
                    / (len(valid_results) * 3)
                    if valid_results
                    else 0,
                },
            },
            "by_type": {},
        }

        # 按类型详细汇总
        for qtype, stats in type_stats.items():
            count = len(stats["retrieval_precision"])
            summary["by_type"][qtype] = {
                "count": count,
                "retrieval": {
                    "avg_precision": sum(stats["retrieval_precision"]) / count,
                    "avg_mrr": sum(stats["retrieval_mrr"]) / count,
                },
                "generation": {
                    "avg_completeness": sum(stats["generation_completeness"]) / count,
                    "avg_accuracy": sum(stats["generation_accuracy"]) / count,
                    "avg_usefulness": sum(stats["generation_usefulness"]) / count,
                    "avg_score": (
                        sum(stats["generation_completeness"])
                        + sum(stats["generation_accuracy"])
                        + sum(stats["generation_usefulness"])
                    )
                    / (count * 3),
                },
            }

        # 记录日志
        self.eval_logger.log_summary(summary["summary"])

        return summary

    def _output_report(
        self,
        summary: Dict[str, Any],
        details: List[Dict[str, Any]],
        output_file: Optional[str],
    ):
        """
        输出评估报告

        Args:
            summary: 汇总信息
            details: 详细结果
            output_file: 输出文件名
        """
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file is None:
            output_file = f"eval_report_{timestamp}.json"

        # 确保是.json后缀
        if not output_file.endswith(".json"):
            output_file += ".json"

        filepath = self.report_dir / output_file

        # 构建完整报告
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "details": details,
        }

        # 保存JSON报告
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n评估报告已保存至: {filepath}")

        # 保存详细日志
        self.eval_logger.save_detail_results()

        print(f"详细日志已保存至: {self.eval_logger.get_detail_path()}")
