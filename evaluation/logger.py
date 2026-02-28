"""
评估日志模块
负责配置日志记录器和保存评估结果
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class EvaluationLogger:
    """
    评估日志记录器

    用于记录评估过程中的各类信息，包括：
    - 测试数据生成
    - 检索评估结果
    - 生成评估结果
    - 汇总报告
    """

    def __init__(self, log_dir: str = "logs/evaluation"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志文件存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的文件名，确保每次评估有独立的日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"eval_{timestamp}.log"
        self.detail_file = self.log_dir / f"eval_detail_{timestamp}.json"

        self._setup_logger()

        # 内存中存储详细评估结果，最后统一保存到JSON
        self.eval_details: List[Dict[str, Any]] = []

    def _setup_logger(self):
        """
        配置日志记录器

        同时输出到文件和控制台
        """
        self.logger = logging.getLogger("EvaluationLogger")
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 文件Handler - 记录完整日志
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)

        # 控制台Handler - 实时显示进度
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 统一的日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_test_data_generation(self, queries: List[Dict]):
        """
        记录测试数据生成信息

        Args:
            queries: 测试查询列表
        """
        self.logger.info("=" * 50)
        self.logger.info("开始生成测试数据")
        self.logger.info("=" * 50)

        for i, q in enumerate(queries):
            self.logger.info(
                f"测试用例 {i + 1}: [{q.get('type', 'unknown')}] {q.get('text', '')}"
            )

        self.logger.info(f"共生成 {len(queries)} 条测试用例")

    def log_retrieval_result(
        self, query: str, docs: List[Any], scores: Dict[str, float]
    ):
        """
        记录单次检索评估结果

        Args:
            query: 测试问题
            docs: 检索到的文档列表
            scores: 评估分数
        """
        self.logger.info(f"\n--- 检索评估: {query}")
        self.logger.info(f"  检索到 {len(docs)} 个文档")

        for i, doc in enumerate(docs):
            # 判断文档是否相关，默认为True（待LLM评估后更新）
            relevance = "相关" if doc.metadata.get("is_relevant", True) else "不相关"
            dish_name = doc.metadata.get("dish_name", "未知")
            self.logger.info(f"    文档{i + 1}: {dish_name} - {relevance}")

        self.logger.info(
            f"  Precision@K: {scores.get('precision', 0):.2f}, "
            f"MRR: {scores.get('mrr', 0):.2f}"
        )

    def log_generation_result(self, query: str, answer: str, scores: Dict[str, float]):
        """
        记录单次生成评估结果

        Args:
            query: 测试问题
            answer: 系统生成的答案
            scores: 评估分数
        """
        self.logger.info(f"\n--- 生成评估: {query}")

        # 完整打印答案，支持换行
        for line in answer.split('\n'):
            self.logger.info(f"  答案: {line}")

        self.logger.info(
            f"  完整性: {scores.get('completeness', 0):.1f}/5, "
            f"准确性: {scores.get('accuracy', 0):.1f}/5, "
            f"实用性: {scores.get('usefulness', 0):.1f}/5"
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        记录评估汇总结果

        Args:
            summary: 汇总信息字典
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("评估汇总报告")
        self.logger.info("=" * 50)

        self.logger.info(f"测试样本数: {summary.get('sample_size', 0)}")

        # 检索质量
        retrieval = summary.get("retrieval", {})
        self.logger.info("检索质量:")
        self.logger.info(f"  Precision@K: {retrieval.get('avg_precision', 0):.2f}")
        self.logger.info(f"  MRR: {retrieval.get('avg_mrr', 0):.2f}")

        # 生成质量
        generation = summary.get("generation", {})
        self.logger.info("生成质量:")
        self.logger.info(f"  完整性: {generation.get('avg_completeness', 0):.1f}/5")
        self.logger.info(f" 准确性: {generation.get('avg_accuracy', 0):.1f}/5")
        self.logger.info(f" 实用性: {generation.get('avg_usefulness', 0):.1f}/5")
        self.logger.info(f" 平均分: {generation.get('avg_score', 0):.1f}/5")

        # 按类型分析
        if "by_type" in summary:
            self.logger.info("按类型分析:")
            for qtype, scores in summary["by_type"].items():
                self.logger.info(
                    f"  {qtype}: 检索{scores.get('retrieval', 0):.2f}, "
                    f"生成{scores.get('generation', 0):.1f}"
                )

    def add_detail(self, detail: Dict[str, Any]):
        """
        添加单条详细评估记录

        Args:
            detail: 包含query、scores等信息的字典
        """
        self.eval_details.append(detail)

    def save_detail_results(self) -> Path:
        """
        保存详细评估结果到JSON文件

        Returns:
            保存的文件路径
        """
        with open(self.detail_file, "w", encoding="utf-8") as f:
            json.dump(self.eval_details, f, ensure_ascii=False, indent=2)

        self.logger.info(f"详细结果已保存至: {self.detail_file}")
        return self.detail_file

    def get_log_path(self) -> Path:
        """获取日志文件路径"""
        return self.log_file

    def get_detail_path(self) -> Path:
        """获取详细结果JSON文件路径"""
        return self.detail_file
