"""
生成质量评估模块
使用LLM评估生成答案的完整性、准确性、实用性
"""

import logging
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .test_data_manager import TestQuery

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """
    生成质量评估器

    使用LLM评估生成答案的质量，从三个维度打分：
    - 完整性：是否完整回答了问题
    - 准确性：是否与提供的上下文一致
    - 实用性：对用户是否有帮助

    每个维度1-5分
    """

    def __init__(self, llm=None):
        """
        初始化生成评估器

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
        self, test_query: TestQuery, context_docs: List[Document], generated_answer: str
    ) -> Dict[str, float]:
        """
        评估生成答案的质量

        让LLM从完整性、准确性、实用性三个维度打分

        Args:
            test_query: 测试问题
            context_docs: 检索到的上下文文档
            generated_answer: 系统生成的答案

        Returns:
            包含各维度分数的字典
        """
        # 构建上下文
        context = self._build_context(context_docs)

        # 调用LLM评估
        scores = self._llm_evaluate(
            query=test_query.text, context=context, answer=generated_answer
        )

        logger.debug(
            f"生成评估结果 - 完整性:{scores['completeness']}, "
            f"准确性:{scores['accuracy']}, 实用性:{scores['usefulness']}"
        )

        return scores

    def _build_context(self, docs: List[Document], max_length: int = 1500) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "无相关上下文"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            # 提取关键信息
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            content = doc.page_content[:500]  # 截取前500字符

            part = f"【文档{i + 1}】{dish_name}\n{content}\n"

            if current_length + len(part) > max_length:
                break

            context_parts.append(part)
            current_length += len(part)

        return "\n".join(context_parts)

    def _llm_evaluate(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """
        使用LLM进行评估打分

        Args:
            query: 用户问题
            context: 上下文文档
            answer: 生成的答案

        Returns:
            各维度分数
        """
        prompt = ChatPromptTemplate.from_template("""
你是一个专业的答案质量评估助手。

请根据提供的上下文和问题，对系统生成的答案进行质量评估。

用户问题：{query}

提供的上下文：
{context}

系统答案：
{answer}

请从以下三个维度对答案进行打分（每个维度1-5分）：

1. 完整性（completeness）：答案是否完整回答了问题？是否遗漏了重要信息？
2. 准确性（accuracy）：答案是否与提供的上下文一致？是否有错误信息？
3. 实用性（usefulness）：答案对用户是否有帮助？是否具有可操作性？

请严格按照以下JSON格式输出，不要添加其他内容：
{{
    "completeness": 分数(1-5),
    "accuracy": 分数(1-5),
    "usefulness": 分数(1-5)
}}
""")

        chain = (
            {
                "query": lambda _: query,
                "context": lambda _: context,
                "answer": lambda _: answer,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke({}).strip()

            # 解析JSON结果
            scores = self._parse_scores(result)
            return scores

        except Exception as e:
            logger.error(f"LLM评估出错: {e}")
            # 出错时返回默认分数
            return {"completeness": 3.0, "accuracy": 3.0, "usefulness": 3.0}

    def _parse_scores(self, result: str) -> Dict[str, float]:
        """
        解析LLM返回的评分结果

        Args:
            result: LLM返回的JSON字符串

        Returns:
            分数字典
        """
        import json
        import re

        # 尝试提取JSON
        json_match = re.search(r"\{[^}]+\}", result, re.DOTALL)

        if json_match:
            try:
                scores = json.loads(json_match.group())
                return {
                    "completeness": float(scores.get("completeness", 3)),
                    "accuracy": float(scores.get("accuracy", 3)),
                    "usefulness": float(scores.get("usefulness", 3)),
                }
            except json.JSONDecodeError:
                pass

        # 如果无法解析，尝试从文本中提取数字
        try:
            numbers = re.findall(r"(\d+(?:\.\d+)?)", result)
            if len(numbers) >= 3:
                return {
                    "completeness": min(5.0, max(1.0, float(numbers[0]))),
                    "accuracy": min(5.0, max(1.0, float(numbers[1]))),
                    "usefulness": min(5.0, max(1.0, float(numbers[2]))),
                }
        except Exception:
            pass

        # 无法解析时返回默认分数
        logger.warning("无法解析LLM评分结果，使用默认分数3.0")
        return {"completeness": 3.0, "accuracy": 3.0, "usefulness": 3.0}

    def evaluate_batch(
        self,
        test_queries: List[TestQuery],
        context_docs_list: List[List[Document]],
        answers: List[str],
    ) -> Dict[str, Any]:
        """
        批量评估多个生成结果

        Args:
            test_queries: 测试问题列表
            context_docs_list: 上下文文档列表
            answers: 生成的答案列表

        Returns:
            汇总的评估指标
        """
        all_scores = []

        # 按类型统计
        type_stats = {}

        for test_query, docs, answer in zip(test_queries, context_docs_list, answers):
            scores = self.evaluate(test_query, docs, answer)
            all_scores.append(scores)

            # 按类型统计
            qtype = test_query.type
            if qtype not in type_stats:
                type_stats[qtype] = {
                    "completeness": [],
                    "accuracy": [],
                    "usefulness": [],
                }
            type_stats[qtype]["completeness"].append(scores["completeness"])
            type_stats[qtype]["accuracy"].append(scores["accuracy"])
            type_stats[qtype]["usefulness"].append(scores["usefulness"])

        # 计算总体平均分
        avg_completeness = (
            sum(s["completeness"] for s in all_scores) / len(all_scores)
            if all_scores
            else 0
        )
        avg_accuracy = (
            sum(s["accuracy"] for s in all_scores) / len(all_scores)
            if all_scores
            else 0
        )
        avg_usefulness = (
            sum(s["usefulness"] for s in all_scores) / len(all_scores)
            if all_scores
            else 0
        )

        result = {
            "avg_completeness": avg_completeness,
            "avg_accuracy": avg_accuracy,
            "avg_usefulness": avg_usefulness,
            "avg_score": (avg_completeness + avg_accuracy + avg_usefulness) / 3,
            "sample_count": len(test_queries),
        }

        # 按类型汇总
        by_type = {}
        for qtype, stats in type_stats.items():
            count = len(stats["completeness"])
            by_type[qtype] = {
                "avg_completeness": sum(stats["completeness"]) / count,
                "avg_accuracy": sum(stats["accuracy"]) / count,
                "avg_usefulness": sum(stats["usefulness"]) / count,
                "avg_score": (
                    sum(stats["completeness"])
                    + sum(stats["accuracy"])
                    + sum(stats["usefulness"])
                )
                / (count * 3),
                "count": count,
            }

        result["by_type"] = by_type

        return result
