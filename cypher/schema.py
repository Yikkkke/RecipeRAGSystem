import logging
from typing import List

from graph_modules.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SchemaBuilder:
    """Schema构建器 - 创建约束和索引"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化Schema构建器

        Args:
            neo4j_client: Neo4j客户端
        """
        self.client = neo4j_client

    def create_all_constraints(self):
        """创建所有约束"""
        constraints = [
            ("recipe_name_unique", "CREATE CONSTRAINT recipe_name_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.name IS UNIQUE"),
            ("ingredient_name_unique", "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE"),
            ("category_name_unique", "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:RecipeCategory) REQUIRE c.name IS UNIQUE"),
            ("difficulty_level_unique", "CREATE CONSTRAINT difficulty_level_unique IF NOT EXISTS FOR (d:DifficultyLevel) REQUIRE d.level IS UNIQUE"),
        ]

        for name, query in constraints:
            try:
                self.client.execute_write_query(query)
                logger.info(f"✓ 创建约束: {name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "EquivalentSchemaRule" in str(e):
                    logger.info(f"⊘ 约束已存在: {name}")
                else:
                    logger.warning(f"创建约束失败 {name}: {e}")

    def create_all_indexes(self):
        """创建所有索引"""
        indexes = [
            ("recipe_difficulty_index", "CREATE INDEX recipe_difficulty_index IF NOT EXISTS FOR (r:Recipe) ON (r.difficulty_star)"),
            ("step_order_index", "CREATE INDEX step_order_index IF NOT EXISTS FOR (s:CookingStep) ON (s.step_order)"),
        ]

        for name, query in indexes:
            try:
                self.client.execute_write_query(query)
                logger.info(f"✓ 创建索引: {name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "EquivalentSchemaRule" in str(e):
                    logger.info(f"⊘ 索引已存在: {name}")
                else:
                    logger.warning(f"创建索引失败 {name}: {e}")

    def drop_all_constraints(self):
        """删除所有约束（慎用！）"""
        constraints = [
            "DROP CONSTRAINT recipe_name_unique IF EXISTS",
            "DROP CONSTRAINT ingredient_name_unique IF EXISTS",
            "DROP CONSTRAINT category_name_unique IF EXISTS",
            "DROP CONSTRAINT difficulty_level_unique IF EXISTS",
        ]

        for query in constraints:
            try:
                self.client.execute_write_query(query)
                logger.info(f"✓ 删除约束: {query}")
            except Exception as e:
                logger.warning(f"删除约束失败: {e}")

    def drop_all_indexes(self):
        """删除所有索引（慎用！）"""
        indexes = [
            "DROP INDEX recipe_difficulty_index IF EXISTS",
            "DROP INDEX step_order_index IF EXISTS",
        ]

        for query in indexes:
            try:
                self.client.execute_write_query(query)
                logger.info(f"✓ 删除索引: {query}")
            except Exception as e:
                logger.warning(f"删除索引失败: {e}")

    def verify_schema(self) -> bool:
        """
        验证Schema是否正确

        Returns:
            Schema是否有效
        """
        try:
            result = self.client.execute_query("SHOW CONSTRAINTS")
            if len(result) < 4:
                logger.warning(f"约束数量不足，预期4个，实际{len(result)}个")
                return False

            result = self.client.execute_query("SHOW INDEXES")
            logger.info(f"Schema验证通过: {len(result)}个索引")

            return True
        except Exception as e:
            logger.error(f"Schema验证失败: {e}")
            return False
