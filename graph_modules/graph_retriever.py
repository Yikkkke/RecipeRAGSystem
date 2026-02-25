import logging
from typing import List, Dict, Any, Optional

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphRetriever:
    """图检索器"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化图检索器

        Args:
            neo4j_client: Neo4j客户端
        """
        self.client = neo4j_client

    def find_recipes_by_ingredients(
        self, ingredient_names: List[str], limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按食材查找菜谱（使用REQUIRES关系）

        Args:
            ingredient_names: 食材名称列表
            limit: 返回结果数量限制

        Returns:
            匹配的菜谱列表
        """
        query = """
        MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
        WHERE i.name IN $ingredient_names
        WITH r, count(i) AS ingredient_count, collect(i.name) AS ingredients
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               ingredient_count, ingredients,
               d.description AS difficulty_description,
               c.name AS category
        ORDER BY ingredient_count DESC, r.difficulty_star ASC
        LIMIT $limit
        """

        results = self.client.execute_query(
            query, {"ingredient_names": ingredient_names, "limit": limit}
        )

        return results

    def find_recipes_by_category(
        self, category_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        按分类查找菜谱（使用BELONGS_TO_CATEGORY关系）

        Args:
            category_name: 分类名称
            limit: 返回结果数量限制

        Returns:
            匹配的菜谱列表
        """
        query = """
        MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory {name: $category_name})
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               d.description AS difficulty_description, c.name AS category
        ORDER BY r.difficulty_star ASC
        LIMIT $limit
        """

        results = self.client.execute_query(
            query, {"category_name": category_name, "limit": limit}
        )

        return results

    def find_recipes_by_difficulty(
        self, max_star: int, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        按难度等级查找菜谱（使用HAS_DIFFICULTY_LEVEL关系）

        Args:
            max_star: 最大星级（包含）
            limit: 返回结果数量限制

        Returns:
            匹配的菜谱列表
        """
        query = """
        MATCH (r:Recipe)
        WHERE r.difficulty_star <= $max_star
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               d.description AS difficulty_description, c.name AS category
        ORDER BY r.difficulty_star ASC
        LIMIT $limit
        """

        results = self.client.execute_query(
            query, {"max_star": max_star, "limit": limit}
        )

        return results

    def _find_recipe_by_name(self, recipe_name: str) -> Optional[str]:
        """
        根据菜谱名称查找匹配的菜谱名（支持模糊匹配）

        Args:
            recipe_name: 用户输入的菜谱名称

        Returns:
            匹配到的菜谱名称，如果没有匹配返回None
        """
        exact_query = """
        MATCH (r:Recipe {name: $recipe_name})
        RETURN r.name AS name
        LIMIT 1
        """
        exact_result = self.client.execute_query(
            exact_query, {"recipe_name": recipe_name}
        )
        if exact_result:
            return exact_result[0]["name"]

        fuzzy_query = """
        MATCH (r:Recipe)
        WHERE r.name CONTAINS $recipe_name OR $recipe_name CONTAINS r.name
        RETURN r.name AS name
        ORDER BY 
            CASE 
                WHEN r.name = $recipe_name THEN 0
                WHEN r.name STARTS WITH $recipe_name THEN 1
                WHEN r.name CONTAINS $recipe_name THEN 2
                WHEN $recipe_name CONTAINS r.name THEN 3
                ELSE 4
            END,
            size(r.name) ASC
        LIMIT 1
        """
        fuzzy_result = self.client.execute_query(
            fuzzy_query, {"recipe_name": recipe_name}
        )
        if fuzzy_result:
            return fuzzy_result[0]["name"]

        return None

    def find_similar_recipes(
        self, recipe_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找相似菜谱（基于共同食材）

        Args:
            recipe_name: 菜谱名称
            limit: 返回结果数量限制

        Returns:
            相似菜谱列表
        """
        matched_name = self._find_recipe_by_name(recipe_name)
        if not matched_name:
            logger.warning(f"未找到菜谱: {recipe_name}")
            return []

        query = """
        MATCH (r1:Recipe {name: $recipe_name})-[:REQUIRES]->(i:Ingredient)<-[:REQUIRES]-(r2:Recipe)
        WHERE r1 <> r2
        WITH r2, count(i) AS common_ingredients, collect(i.name) AS common_ingredient_names
        OPTIONAL MATCH (r2)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r2)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        RETURN r2.name AS name, r2.difficulty_star AS difficulty_star,
               common_ingredients, common_ingredient_names,
               d.description AS difficulty_description, c.name AS category
        ORDER BY common_ingredients DESC
        LIMIT $limit
        """

        results = self.client.execute_query(
            query, {"recipe_name": matched_name, "limit": limit}
        )

        return results

    def get_recipe_details(self, recipe_name: str) -> Optional[Dict[str, Any]]:
        """
        获取菜谱详情

        Args:
            recipe_name: 菜谱名称

        Returns:
            菜谱详情字典
        """
        matched_name = self._find_recipe_by_name(recipe_name)
        if not matched_name:
            logger.warning(f"未找到菜谱: {recipe_name}")
            return None

        query = """
        MATCH (r:Recipe {name: $recipe_name})
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
        OPTIONAL MATCH (r)-[:CONTAINS_STEP]->(s:CookingStep)
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               c.name AS category, d.description AS difficulty_description,
               collect(DISTINCT {name: i.name, amount: i.amount, unit: i.unit}) AS ingredients,
               collect(DISTINCT {step_id: s.step_id, order: s.step_order, description: s.description}) AS steps
        """

        results = self.client.execute_query(query, {"recipe_name": matched_name})

        if results:
            return results[0]
        return None

    def get_all_categories(self) -> List[str]:
        """
        获取所有分类

        Returns:
            分类列表
        """
        query = """
        MATCH (c:RecipeCategory)
        RETURN c.name AS name
        ORDER BY c.name
        """

        results = self.client.execute_query(query)
        return [r["name"] for r in results]

    def get_all_ingredients(self) -> List[str]:
        """
        获取所有食材

        Returns:
            食材列表
        """
        query = """
        MATCH (i:Ingredient)
        RETURN i.name AS name
        ORDER BY i.name
        """

        results = self.client.execute_query(query)
        return [r["name"] for r in results]

    def search_recipes(
        self,
        ingredients: Optional[List[str]] = None,
        category: Optional[str] = None,
        max_difficulty: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        综合搜索菜谱

        Args:
            ingredients: 食材列表（必须包含的食材）
            category: 分类名称
            max_difficulty: 最大难度星级
            limit: 返回结果数量限制

        Returns:
            匹配的菜谱列表
        """
        conditions = []
        params = {"limit": limit}

        if ingredients:
            conditions.append(
                "EXISTS {(r)-[:REQUIRES]->(i:Ingredient) WHERE i.name IN $ingredients}"
            )
            params["ingredients"] = ingredients

        if category:
            conditions.append(
                "EXISTS {(r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory) WHERE c.name = $category}"
            )
            params["category"] = category

        if max_difficulty is not None:
            conditions.append("r.difficulty_star <= $max_difficulty")
            params["max_difficulty"] = max_difficulty

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (r:Recipe)
        WHERE {where_clause}
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
        WITH r, d, c, collect(DISTINCT i.name) AS ingredients
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               d.description AS difficulty_description, c.name AS category,
               ingredients
        ORDER BY r.difficulty_star ASC
        LIMIT $limit
        """

        return self.client.execute_query(query, params)
