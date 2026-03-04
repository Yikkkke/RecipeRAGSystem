import logging
from typing import List, Dict, Any, Optional

from .graph_retriever import GraphRetriever
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphVectorHybridRetriever:
    """图谱+向量混合检索器 - 结合图检索和向量检索"""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever=None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化混合检索器

        Args:
            graph_retriever: 图检索器
            vector_retriever: 向量检索器（可选）
            weights: 检索权重配置 {'graph': 0.3, 'vector': 0.7}
        """
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.weights = weights or {"graph": 0.3, "vector": 0.7}

    def search(
        self,
        query: str,
        ingredients: Optional[List[str]] = None,
        category: Optional[str] = None,
        max_difficulty: Optional[int] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        混合搜索 - 图+向量混合检索

        Args:
            query: 查询文本
            ingredients: 食材列表（用于图检索）
            category: 分类（用于图检索）
            max_difficulty: 最大难度（用于图检索）
            top_k: 返回结果数量

        Returns:
            混合检索结果
        """
        # 如果有明确的过滤条件或向量检索器可用，使用混合检索
        if ingredients or category or max_difficulty or self.vector_retriever:
            # 尝试从query中提取关键词（如果没有传入）
            if not ingredients and not category:
                keywords = self._extract_keywords(query)
                all_ingredients = self.graph_retriever.get_all_ingredients()
                matched_ingredients = [
                    ing for ing in keywords if ing in all_ingredients
                ]

                all_categories = self.graph_retriever.get_all_categories()
                matched_category = None
                for cat in all_categories:
                    if cat in query:
                        matched_category = cat
                        break

                ingredients = matched_ingredients
                category = matched_category

            results = self._hybrid_search(
                query=query,
                ingredients=ingredients,
                category=category,
                max_difficulty=max_difficulty,
                top_k=top_k,
            )
        else:
            # 没有向量检索器时，使用纯图检索
            results = self._graph_search(query, top_k)

        return results

    def _hybrid_search(
        self,
        query: str,
        ingredients: Optional[List[str]],
        category: Optional[str],
        max_difficulty: Optional[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（图+向量）- 真正的混合检索实现

        Args:
            query: 查询文本
            ingredients: 食材列表
            category: 分类
            max_difficulty: 最大难度
            top_k: 返回结果数量

        Returns:
            混合检索结果
        """
        all_results = {}  # doc_id -> result
        doc_objects = {}  # doc_id -> result object
        doc_scores = {}  # doc_id -> rrf_score

        # 1. 图谱检索
        graph_results = self.graph_retriever.search_recipes(
            ingredients=ingredients,
            category=category,
            max_difficulty=max_difficulty,
            limit=top_k * 2,
        )
        for rank, result in enumerate(graph_results):
            doc_id = result["name"]
            doc_objects[doc_id] = result
            score = 1.0 / (rank + 60)  # RRF k=60
            doc_scores[doc_id] = (
                doc_scores.get(doc_id, 0) + score * self.weights["graph"]
            )

        # 2. 向量检索 (如果可用)
        if self.vector_retriever is not None:
            try:
                vector_docs = self.vector_retriever.get_relevant_documents(query)
                for rank, doc in enumerate(vector_docs):
                    dish_name = doc.metadata.get("dish_name", "")
                    if not dish_name:
                        continue
                    if dish_name not in doc_objects:
                        doc_objects[dish_name] = {"name": dish_name}
                    score = 1.0 / (rank + 60)  # RRF k=60
                    doc_scores[dish_name] = (
                        doc_scores.get(dish_name, 0) + score * self.weights["vector"]
                    )
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 3. 按得分排序
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        combined_results = []
        for doc_id, score in ranked[:top_k]:
            result = doc_objects[doc_id]
            combined_results.append({**result, "hybrid_score": score})

        return combined_results

    def _graph_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        纯图检索（用于没有明确过滤条件的查询）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            图检索结果
        """
        all_categories = self.graph_retriever.get_all_categories()

        keywords = self._extract_keywords(query)
        all_ingredients = self.graph_retriever.get_all_ingredients()
        matched_ingredients = [ing for ing in keywords if ing in all_ingredients]

        matched_category = None
        for cat in all_categories:
            if cat in query:
                matched_category = cat
                break

        if matched_ingredients or matched_category:
            return self.graph_retriever.search_recipes(
                ingredients=matched_ingredients if matched_ingredients else None,
                category=matched_category,
                limit=top_k,
            )
        else:
            all_recipes = self.graph_retriever.search_recipes(limit=top_k)
            return all_recipes

    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        import re

        keywords = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        return keywords[:10]

    def find_recipes_with_ingredients(
        self, ingredient_names: List[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找包含指定食材的菜谱

        Args:
            ingredient_names: 食材名称列表
            top_k: 返回结果数量

        Returns:
            匹配的菜谱列表
        """
        return self.graph_retriever.find_recipes_by_ingredients(
            ingredient_names, limit=top_k
        )

    def find_similar_recipes(
        self, recipe_name: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        查找相似菜谱

        Args:
            recipe_name: 菜谱名称
            top_k: 返回结果数量

        Returns:
            相似菜谱列表
        """
        return self.graph_retriever.find_similar_recipes(recipe_name, limit=top_k)

    def get_recipe_details(self, recipe_name: str) -> Optional[Dict[str, Any]]:
        """
        获取菜谱详情

        Args:
            recipe_name: 菜谱名称

        Returns:
            菜谱详情
        """
        return self.graph_retriever.get_recipe_details(recipe_name)
