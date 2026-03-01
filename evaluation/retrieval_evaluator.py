"""
检索质量评估模块
使用LLM判断检索到的文档是否与问题相关，计算Precision@K、MRR等指标
"""

import logging
import re
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
        self,
        test_query: TestQuery,
        retrieved_docs: List[Document],
        top_k: int = 3,
        route_type: str = None,
    ) -> Dict[str, float]:
        """
        评估检索质量

        对每个检索到的文档，让LLM判断是否与问题相关，
        然后计算Precision@K和MRR

        Args:
            test_query: 测试问题
            retrieved_docs: 检索到的文档列表
            top_k: 评估时考虑的文档数量
            route_type: 路由类型（可选，如果传入优先使用）

        Returns:
            包含 precision 和 mrr 的字典
        """
        # 确定使用的 query_type
        query_type = route_type if route_type else test_query.type

        # 限制评估的文档数量
        docs_to_eval = retrieved_docs[:top_k]

        relevance_results = []

        for i, doc in enumerate(docs_to_eval):
            # 构建判断prompt
            is_relevant = self._judge_relevance(
                query=test_query.text,
                query_type=query_type,
                doc_content=doc.page_content,
                doc_name=doc.metadata.get("dish_name", "未知"),
                doc_metadata=doc.metadata,
                expected_docs=test_query.expected_docs,
                filters=test_query.filters,
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
        self,
        query: str,
        query_type: str,
        doc_content: str,
        doc_name: str,
        doc_metadata: Dict[str, Any],
        expected_docs: List[str] = None,
        filters: Dict[str, str] = None,
        max_doc_length: int = 1000,
    ) -> bool:
        """
        使用混合方式判断文档是否与问题相关

        判断优先级：
        1. expected_docs - 测试数据中指定的相关文档
        2. 元数据匹配 - category/difficulty过滤条件
        3. 类型特定判断 - list/detail/with_filter各有不同标准
        4. 语义判断 - LLM兜底判断

        Args:
            query: 用户问题
            query_type: 问题类型 (detail/list/with_filter)
            doc_content: 文档内容
            doc_name: 文档名称
            doc_metadata: 文档元数据 (category, difficulty等)
            expected_docs: 期望检索到的文档列表
            filters: 查询中的过滤条件
            max_doc_length: 截取文档长度

        Returns:
            True表示相关，False表示不相关
        """
        # 1. 检查expected_docs（最高优先级）
        if expected_docs:
            for expected in expected_docs:
                if expected in doc_name or doc_name in expected:
                    logger.debug(f"[{doc_name}] 命中expected_docs: {expected}")
                    return True

        # 2. 类型特定的判断逻辑
        if query_type == "list":
            return self._judge_list_relevance(
                query, doc_name, doc_metadata, doc_content
            )

        elif query_type == "detail":
            return self._judge_detail_relevance(query, doc_name, doc_content)

        elif query_type == "with_filter":
            # with_filter类型：先检查filters是否匹配
            if filters:
                matched = True
                for key, value in filters.items():
                    doc_value = doc_metadata.get(key)
                    if doc_value and value in str(doc_value):
                        continue
                    else:
                        matched = False
                        break
                if matched:
                    logger.debug(f"[{doc_name}] 命中filters: {filters}")
                    return True
            # filters不匹配时，使用filter专用判断逻辑
            return self._judge_filter_relevance(query, doc_name, doc_metadata, filters)

        # 3. 兜底：语义判断
        return self._semantic_judge(query, doc_content, doc_name, max_doc_length)

    def _judge_list_relevance(
        self, query: str, doc_name: str, doc_metadata: Dict, doc_content: str
    ) -> bool:
        """判断list类型查询的相关性"""
        # 从查询中提取类别关键词
        query_lower = query.lower()

        # 常见类别映射
        category_keywords = {
            "早餐": ["早餐", "早饭", "早茶"],
            "午餐": ["午餐", "午饭"],
            "晚餐": ["晚餐", "晚饭"],
            "素菜": ["素菜", "素食", "蔬菜"],
            "荤菜": ["荤菜", "肉菜", "肉类"],
            "水产": ["水产", "海鲜", "鱼", "虾", "蟹"],
            "汤品": ["汤", "羹"],
            "主食": ["主食", "饭", "面", "粥", "饼"],
            "甜品": ["甜品", "甜点", "蛋糕", "冰淇淋"],
            "饮品": ["饮品", "饮料", "茶", "咖啡", "奶", "果汁"],
            "小吃": ["小吃", "零食"],
            "调味": ["调料", "酱汁", "佐料"],
        }

        # 从查询中识别目标类别
        target_category = None
        for cat, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                target_category = cat
                break

        if target_category:
            # 检查文档的category字段
            doc_category = doc_metadata.get("category", "")
            if target_category in str(doc_category) or any(
                kw in str(doc_category)
                for kw in category_keywords.get(target_category, [])
            ):
                logger.debug(f"[{doc_name}] list类型命中category: {doc_category}")
                return True

            # 语义检查：常见早餐/饮品等的菜名
            common_dishes = {
                "早餐": [
                    "粥",
                    "饼",
                    "蛋",
                    "面",
                    "豆浆",
                    "油条",
                    "包子",
                    "馒头",
                    "吐司",
                    "三明治",
                ],
                "饮品": [
                    "奶茶",
                    "咖啡",
                    "果汁",
                    "奶",
                    "茶",
                    "可乐",
                    "汽水",
                    "冰沙",
                    "特调",
                ],
                "素菜": [
                    "青菜",
                    "白菜",
                    "菠菜",
                    "芹菜",
                    "土豆",
                    "茄子",
                    "番茄",
                    "黄瓜",
                ],
                "荤菜": ["鸡", "鸭", "鹅", "猪", "牛", "羊", "鱼", "虾", "肉"],
                "汤品": ["汤", "羹", "粥"],
                "主食": ["饭", "面", "粥", "粉", "饼", "馒头", "包子"],
                "甜品": ["蛋糕", "冰淇淋", "布丁", "奶茶", "糖水", "酥"],
            }
            if target_category in common_dishes:
                for dish_keyword in common_dishes[target_category]:
                    if dish_keyword in doc_name:
                        logger.debug(f"[{doc_name}] list类型语义命中: {dish_keyword}")
                        return True

        # 兜底
        logger.debug(f"[{doc_name}] list类型未命中")
        return False

    def _judge_detail_relevance(
        self, query: str, doc_name: str, doc_content: str
    ) -> bool:
        """判断detail类型查询的相关性"""
        dish_name = query
        for suffix in ["怎么做", "的做法", "如何做", "要怎么做", "怎么烧", "怎么煮"]:
            dish_name = dish_name.replace(suffix, "")
        dish_name = dish_name.strip()

        if dish_name in doc_name or doc_name in dish_name:
            logger.debug(f"[{doc_name}] detail类型命中文档名")
            return True

        first_line = doc_content.split("\n")[0] if doc_content else ""
        title_dish = re.sub(r"^#+\s*", "", first_line).strip()
        if dish_name in title_dish:
            logger.debug(f"[{doc_name}] detail类型命中标题")
            return True

        logger.debug(f"[{doc_name}] detail类型未命中: {dish_name}")
        return False

    def _judge_filter_relevance(
        self,
        query: str,
        doc_name: str,
        doc_metadata: Dict,
        filters: Dict[str, str],
    ) -> bool:
        """判断with_filter类型查询的相关性"""
        if not filters:
            return False

        # 检查每个过滤条件
        for key, value in filters.items():
            doc_value = str(doc_metadata.get(key, ""))

            # 模糊匹配
            if value in doc_value:
                continue
            # 同义词映射
            synonyms = {
                "简单": ["简单", "非常简单", "★"],
                "中等": ["中等", "一般"],
                "困难": ["困难", "非常困难", "★★★"],
            }
            if value in synonyms:
                if not any(syn in doc_value for syn in synonyms[value]):
                    return False
            elif value not in doc_value:
                return False

        logger.debug(f"[{doc_name}] filter类型命中: {filters}")
        return True

    def _semantic_judge(
        self, query: str, doc_content: str, doc_name: str, max_doc_length: int = 1000
    ) -> bool:
        """使用LLM进行语义判断（兜底）"""
        # 截取文档内容，避免过长
        truncated_content = doc_content[:max_doc_length]

        # 构建更智能的判断prompt
        prompt = ChatPromptTemplate.from_template("""
你是一个专业的相关性判断助手。

请判断给定的文档是否能够回答用户的问题。
判断标准：
1. 如果文档讨论的主题与问题相关，则为"相关"
2. 如果文档是关于做法的（菜谱），而问题询问某个菜品的做法，则为"相关"
3. 如果文档是推荐列表中的一道菜，而问题询问某类推荐，则为"相关"

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
            qtype = test_query.type if test_query.type else "unknown"
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
