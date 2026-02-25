import logging
from typing import List, Dict, Any, Optional

from .graph_retriever import GraphRetriever
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器 - 结合图检索和向量检索"""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever=None,
        weights: Optional[Dict[str, float]] = None
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
        self.weights = weights or {'graph': 0.3, 'vector': 0.7}

    def search(
        self,
        query: str,
        ingredients: Optional[List[str]] = None,
        category: Optional[str] = None,
        max_difficulty: Optional[int] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合搜索

        Args:
            query: 查询文本
            ingredients: 食材列表（用于图检索）
            category: 分类（用于图检索）
            max_difficulty: 最大难度（用于图检索）
            top_k: 返回结果数量

        Returns:
            混合检索结果
        """
        if ingredients or category or max_difficulty:
            results = self._hybrid_search(
                query=query,
                ingredients=ingredients,
                category=category,
                max_difficulty=max_difficulty,
                top_k=top_k
            )
        else:
            results = self._graph_search(query, top_k)

        return results

    def _hybrid_search(
        self,
        query: str,
        ingredients: Optional[List[str]],
        category: Optional[str],
        max_difficulty: Optional[int],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（图+向量）

        Args:
            query: 查询文本
            ingredients: 食材列表
            category: 分类
            max_difficulty: 最大难度
            top_k: 返回结果数量

        Returns:
            混合检索结果
        """
        graph_results = self.graph_retriever.search_recipes(
            ingredients=ingredients,
            category=category,
            max_difficulty=max_difficulty,
            limit=top_k * 2
        )

        graph_scores = {r['name']: self.weights['graph'] for r in graph_results}

        combined_results = []
        for result in graph_results:
            name = result['name']
            score = graph_scores.get(name, 0)
            combined_results.append({
                **result,
                'hybrid_score': score,
                'source': 'graph'
            })

        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return combined_results[:top_k]

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
                limit=top_k
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
        keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        return keywords[:10]

    def find_recipes_with_ingredients(
        self,
        ingredient_names: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找包含指定食材的菜谱

        Args:
            ingredient_names: 食材名称列表
            top_k: 返回结果数量

        Returns:
            匹配的菜谱列表
        """
        return self.graph_retriever.find_recipes_by_ingredients(ingredient_names, limit=top_k)

    def find_similar_recipes(
        self,
        recipe_name: str,
        top_k: int = 5
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
