import logging
from typing import Dict, List, Any, Optional

from graph_modules.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class CypherQueries:
    """常用Cypher查询集合"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化Cypher查询集合

        Args:
            neo4j_client: Neo4j客户端
        """
        self.client = neo4j_client

    def count_all_nodes(self) -> Dict[str, int]:
        """
        统计所有节点数量

        Returns:
            各类型节点的数量
        """
        query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """

        results = self.client.execute_query(query)
        return {r['label']: r['count'] for r in results}

    def count_all_relationships(self) -> Dict[str, int]:
        """
        统计所有关系数量

        Returns:
            各类型关系的数量
        """
        query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """

        results = self.client.execute_query(query)
        return {r['rel_type']: r['count'] for r in results}

    def get_recipe_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        根据菜谱名称获取菜谱

        Args:
            name: 菜谱名称

        Returns:
            菜谱信息
        """
        query = """
        MATCH (r:Recipe {name: $name})
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        RETURN r.name AS name, r.difficulty_star AS difficulty_star,
               c.name AS category, d.description AS difficulty_description
        """

        results = self.client.execute_query(query, {'name': name})
        return results[0] if results else None

    def find_ingredient_substitutes(self, ingredient_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        查找食材替代品（基于共同出现在菜谱中的频率）

        Args:
            ingredient_name: 食材名称
            limit: 返回数量限制

        Returns:
            替代品列表
        """
        query = """
        MATCH (i1:Ingredient {name: $ingredient_name})<-[:REQUIRES]-(r:Recipe)-[:REQUIRES]->(i2:Ingredient)
        WHERE i1 <> i2
        WITH i2, count(r) AS co_occurrence
        RETURN i2.name AS name, co_occurrence
        ORDER BY co_occurrence DESC
        LIMIT $limit
        """

        return self.client.execute_query(query, {
            'ingredient_name': ingredient_name,
            'limit': limit
        })

    def get_most_used_ingredients(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取使用最多的食材

        Args:
            limit: 返回数量限制

        Returns:
            食材列表及其使用次数
        """
        query = """
        MATCH (i:Ingredient)<-[:REQUIRES]-(r:Recipe)
        RETURN i.name AS name, count(r) AS usage_count
        ORDER BY usage_count DESC
        LIMIT $limit
        """

        return self.client.execute_query(query, {'limit': limit})

    def get_category_statistics(self) -> List[Dict[str, Any]]:
        """
        获取各分类的统计信息

        Returns:
            分类统计列表
        """
        query = """
        MATCH (c:RecipeCategory)<-[:BELONGS_TO_CATEGORY]-(r:Recipe)
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        RETURN c.name AS category, count(r) AS recipe_count,
               avg(r.difficulty_star) AS avg_difficulty,
               collect(d.description)[0] AS most_common_difficulty
        ORDER BY recipe_count DESC
        """

        return self.client.execute_query(query)

    def find_recipes_with_most_ingredients(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        查找食材最多的菜谱

        Args:
            limit: 返回数量限制

        Returns:
            菜谱列表
        """
        query = """
        MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
        WITH r, count(i) AS ingredient_count
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        RETURN r.name AS name, ingredient_count, c.name AS category
        ORDER BY ingredient_count DESC
        LIMIT $limit
        """

        return self.client.execute_query(query, {'limit': limit})

    def find_recipes_with_fewest_ingredients(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        查找食材最少的菜谱

        Args:
            limit: 返回数量限制

        Returns:
            菜谱列表
        """
        query = """
        MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
        WITH r, count(i) AS ingredient_count
        WHERE ingredient_count > 0
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:RecipeCategory)
        RETURN r.name AS name, ingredient_count, c.name AS category
        ORDER BY ingredient_count ASC
        LIMIT $limit
        """

        return self.client.execute_query(query, {'limit': limit})

    def get_difficulty_distribution(self) -> List[Dict[str, Any]]:
        """
        获取难度分布统计

        Returns:
            难度分布列表
        """
        query = """
        MATCH (r:Recipe)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        RETURN d.description AS difficulty, d.star_count AS star_count, count(r) AS recipe_count
        ORDER BY d.star_count
        """

        return self.client.execute_query(query)

    def find_ingredient_combinations(self, min_count: int = 3) -> List[Dict[str, Any]]:
        """
        查找常见的食材组合

        Args:
            min_count: 最少共同出现次数

        Returns:
            食材组合列表
        """
        query = """
        MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
        WITH r, collect(i.name) AS ingredients, count(i) AS ing_count
        WHERE ing_count >= $min_count
        UNWIND range(0, size(ingredients)-2) AS i
        UNWIND range(i+1, size(ingredients)-1) AS j
        WITH ingredients[i] AS ing1, ingredients[j] AS ing2, count(r) AS pair_count
        RETURN ing1, ing2, pair_count
        ORDER BY pair_count DESC
        LIMIT 20
        """

        return self.client.execute_query(query, {'min_count': min_count})

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱整体统计信息

        Returns:
            统计信息字典
        """
        node_stats = self.count_all_nodes()
        rel_stats = self.count_all_relationships()

        return {
            'nodes': node_stats,
            'relationships': rel_stats,
            'total_nodes': sum(node_stats.values()),
            'total_relationships': sum(rel_stats.values())
        }
