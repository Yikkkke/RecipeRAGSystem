"""
检索质量评估模块
使用LLM判断检索到的文档是否与问题相关，计算Precision@K、MRR等指标
"""

import logging
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .test_data_manager import TestQuery

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    检索质量评估器

    使用LLM判断每份检索到的文档是否"回答了问题"，
    据此计算检索质量指标：
    - Precision@K: 相关文档数 / 检索总数
    - MRR: 第一个相关文档排名的倒数平均值
    """

    def __init__(self, llm=None):
        """
        初始化检索评估器

        Args:
            llm: 语言模型实例，如果为None则使用默认的MoonshotChat
        """
        self.llm = llm
        self._init_llm()

    def _init_llm(self):
        """初始化LLM（如果未提供）"""
        if self.llm is None:
            import os
            from langchain_community.chat_models.moonshot import MoonshotChat

            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

            self.llm = MoonshotChat(
                model="kimi-k2-0711-preview", temperature=0.1, moonshot_api_key=api_key
            )

    def evaluate(
        self, test_query: TestQuery, retrieved_docs: List[Document], top_k: int = 3
    ) -> Dict[str, float]:
        """
        评估检索质量

        对每个检索到的文档，让LLM判断是否与问题相关，
        然后计算Precision@K和MRR

        Args:
            test_query: 测试问题
            retrieved_docs: 检索到的文档列表
            top_k: 评估时考虑的文档数量

        Returns:
            包含 precision 和 mrr 的字典
        """
        # 限制评估的文档数量
        docs_to_eval = retrieved_docs[:top_k]

        relevance_results = []

        for i, doc in enumerate(docs_to_eval):
            # 构建判断prompt
            is_relevant = self._judge_relevance(
                query=test_query.text,
                doc_content=doc.page_content,
                doc_name=doc.metadata.get("dish_name", "未知"),
            )

            # 保存判断结果到文档元数据
            doc.metadata["is_relevant"] = is_relevant
            relevance_results.append(is_relevant)

            logger.debug(
                f"文档{i + 1}({doc.metadata.get('dish_name')}): "
                f"{'相关' if is_relevant else '不相关'}"
            )

        # 计算指标
        precision = self._calculate_precision(relevance_results)
        mrr = self._calculate_mrr(relevance_results)

        return {
            "precision": precision,
            "mrr": mrr,
            "relevant_count": sum(relevance_results),
            "total_count": len(relevance_results),
        }

    def _judge_relevance(
        self, query: str, doc_content: str, doc_name: str, max_doc_length: int = 1000
    ) -> bool:
        """
        使用LLM判断文档是否与问题相关

        Args:
            query: 用户问题
            doc_content: 文档内容
            doc_name: 文档名称（用于日志）
            max_doc_length: 截取文档长度，避免超出LLM上下文限制

        Returns:
            True表示相关，False表示不相关
        """
        # 截取文档内容，避免过长
        truncated_content = doc_content[:max_doc_length]

        # 构建判断prompt
        prompt = ChatPromptTemplate.from_template("""
你是一个专业的相关性判断助手。

请判断给定的文档是否能够回答用户的问题。

用户问题：{query}

文档内容：
{doc_content}

请仅回答"相关"或"不相关"，不要添加其他内容。
""")

        chain = (
            {"query": lambda _: query, "doc_content": lambda _: truncated_content}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke({}).strip().lower()

            # 解析结果
            if "相关" in result and "不相关" not in result:
                return True
            elif "不相关" in result:
                return False
            else:
                # 如果无法明确判断，默认为相关（保守策略）
                logger.warning(f"无法明确判断文档'{doc_name}'的相关性，默认判定为相关")
                return True

        except Exception as e:
            logger.error(f"判断文档相关性时出错: {e}")
            return True  # 出错时默认返回相关

    def _calculate_precision(self, relevance_results: List[bool]) -> float:
        """
        计算Precision@K

        Precision@K = 相关文档数 / 检索总数

        Args:
            relevance_results: 每个位置的文档是否相关

        Returns:
            Precision分数
        """
        if not relevance_results:
            return 0.0

        relevant_count = sum(relevance_results)
        return relevant_count / len(relevance_results)

    def _calculate_mrr(self, relevance_results: List[bool]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)

        MRR = 第一个相关文档排名倒数的平均值
        即：如果第一个相关文档在第1位，得分为1；第2位，得分为0.5；依此类推

        Args:
            relevance_results: 每个位置的文档是否相关

        Returns:
            MRR分数
        """
        for i, is_relevant in enumerate(relevance_results):
            if is_relevant:
                return 1.0 / (i + 1)

        return 0.0

    def evaluate_batch(
        self,
        test_queries: List[TestQuery],
        retrieved_docs_list: List[List[Document]],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        批量评估多个查询的检索质量

        Args:
            test_queries: 测试问题列表
            retrieved_docs_list: 对应的检索结果列表
            top_k: 评估的文档数量

        Returns:
            汇总的评估指标
        """
        all_precisions = []
        all_mrrs = []

        # 按类型统计
        type_stats = {}

        for test_query, docs in zip(test_queries, retrieved_docs_list):
            scores = self.evaluate(test_query, docs, top_k)

            all_precisions.append(scores["precision"])
            all_mrrs.append(scores["mrr"])

            # 按类型统计
            qtype = test_query.type
            if qtype not in type_stats:
                type_stats[qtype] = {"precision": [], "mrr": []}
            type_stats[qtype]["precision"].append(scores["precision"])
            type_stats[qtype]["mrr"].append(scores["mrr"])

        # 计算总体指标
        result = {
            "avg_precision": sum(all_precisions) / len(all_precisions)
            if all_precisions
            else 0,
            "avg_mrr": sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0,
            "sample_count": len(test_queries),
        }

        # 按类型汇总
        by_type = {}
        for qtype, stats in type_stats.items():
            by_type[qtype] = {
                "avg_precision": sum(stats["precision"]) / len(stats["precision"])
                if stats["precision"]
                else 0,
                "avg_mrr": sum(stats["mrr"]) / len(stats["mrr"]) if stats["mrr"] else 0,
                "count": len(stats["precision"]),
            }

        result["by_type"] = by_type

        return result
