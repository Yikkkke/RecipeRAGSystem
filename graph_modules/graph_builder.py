import logging
from typing import List, Dict, Any
from tqdm import tqdm

from .neo4j_client import Neo4jClient
from cypher.schema import SchemaBuilder

logger = logging.getLogger(__name__)


class GraphBuilder:
    """知识图谱构建器"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化图谱构建器

        Args:
            neo4j_client: Neo4j客户端
        """
        self.client = neo4j_client
        self.schema_builder = SchemaBuilder(neo4j_client)

    def build_schema(self):
        """创建Schema（约束和索引）"""
        logger.info("开始创建Schema...")
        self.schema_builder.create_all_constraints()
        self.schema_builder.create_all_indexes()
        logger.info("Schema创建完成")

    def build_graph(self, graph_data: Dict[str, Any]):
        """
        构建完整知识图谱

        Args:
            graph_data: 从数据提取器获取的结构化数据
        """
        recipes = graph_data.get('recipes', [])
        if not recipes:
            logger.warning("没有菜谱数据，跳过图谱构建")
            return

        logger.info(f"开始构建知识图谱，共 {len(recipes)} 个菜谱...")

        self._create_recipe_categories(graph_data.get('categories', []))
        self._create_difficulty_levels()
        self._create_recipes(recipes)
        self._create_ingredients(recipes)
        self._create_cooking_steps(recipes)
        self._create_belongs_to_category_relationships(recipes)
        self._create_has_difficulty_level_relationships(recipes)
        self._create_requires_relationships(recipes)
        self._create_contains_step_relationships(recipes)

        self._print_statistics()

        logger.info("知识图谱构建完成")

    def _create_recipe_categories(self, categories: List[str]):
        """创建分类节点"""
        if not categories:
            return

        data = [{'name': cat} for cat in categories]
        query = """
        UNWIND $batch AS data
        MERGE (c:RecipeCategory {name: data.name})
        """
        self.client.batch_execute(query, data)
        logger.info(f"创建了 {len(categories)} 个分类节点")

    def _create_difficulty_levels(self):
        """创建难度等级节点"""
        levels = [
            {'level': '1-星', 'star_count': 1, 'description': '非常简单'},
            {'level': '2-星', 'star_count': 2, 'description': '简单'},
            {'level': '3-星', 'star_count': 3, 'description': '中等'},
            {'level': '4-星', 'star_count': 4, 'description': '困难'},
            {'level': '5-星', 'star_count': 5, 'description': '非常困难'}
        ]

        query = """
        UNWIND $batch AS data
        MERGE (d:DifficultyLevel {level: data.level})
        SET d.star_count = data.star_count, d.description = data.description
        """
        self.client.batch_execute(query, levels)
        logger.info("创建了 5 个难度等级节点")

    def _create_recipes(self, recipes: List[Dict[str, Any]]):
        """创建菜谱节点"""
        data = []
        for recipe in recipes:
            data.append({
                'name': recipe['name'],
                'difficulty_star': recipe['difficulty_star']
            })

        query = """
        UNWIND $batch AS data
        MERGE (r:Recipe {name: data.name})
        SET r.difficulty_star = data.difficulty_star
        """
        self.client.batch_execute(query, data)
        logger.info(f"创建了 {len(recipes)} 个菜谱节点")

    def _create_ingredients(self, recipes: List[Dict[str, Any]]):
        """创建食材节点"""
        ingredients = set()
        for recipe in recipes:
            for ing in recipe.get('ingredients', []):
                if ing.get('name'):
                    ingredients.add(ing['name'])

        data = [{'name': ing} for ing in ingredients]
        query = """
        UNWIND $batch AS data
        MERGE (i:Ingredient {name: data.name})
        """
        self.client.batch_execute(query, data)
        logger.info(f"创建了 {len(ingredients)} 个食材节点")

    def _create_cooking_steps(self, recipes: List[Dict[str, Any]]):
        """创建烹饪步骤节点"""
        steps_data = []
        for recipe in recipes:
            recipe_name = recipe['name']
            for step in recipe.get('steps', []):
                steps_data.append({
                    'step_id': f"{recipe_name}_{step['step_id']}",
                    'recipe_name': recipe_name,
                    'description': step['description'],
                    'step_order': step.get('step_order', 0)
                })

        query = """
        UNWIND $batch AS data
        MERGE (s:CookingStep {step_id: data.step_id})
        SET s.description = data.description, s.step_order = data.step_order
        """
        self.client.batch_execute(query, steps_data)
        logger.info(f"创建了 {len(steps_data)} 个烹饪步骤节点")

    def _create_belongs_to_category_relationships(self, recipes: List[Dict[str, Any]]):
        """创建 BELONGS_TO_CATEGORY 关系：菜谱-分类"""
        data = []
        for recipe in recipes:
            data.append({
                'recipe_name': recipe['name'],
                'category_name': recipe['category']
            })

        query = """
        UNWIND $batch AS data
        MATCH (r:Recipe {name: data.recipe_name})
        MATCH (c:RecipeCategory {name: data.category_name})
        MERGE (r)-[:BELONGS_TO_CATEGORY]->(c)
        """
        self.client.batch_execute(query, data)
        logger.info(f"创建了 {len(data)} 个 BELONGS_TO_CATEGORY 关系")

    def _create_has_difficulty_level_relationships(self, recipes: List[Dict[str, Any]]):
        """创建 HAS_DIFFICULTY_LEVEL 关系：菜谱-难度"""
        data = []
        for recipe in recipes:
            data.append({
                'recipe_name': recipe['name'],
                'difficulty_level': recipe['difficulty_level']
            })

        query = """
        UNWIND $batch AS data
        MATCH (r:Recipe {name: data.recipe_name})
        MATCH (d:DifficultyLevel {level: data.difficulty_level})
        MERGE (r)-[:HAS_DIFFICULTY_LEVEL]->(d)
        """
        self.client.batch_execute(query, data)
        logger.info(f"创建了 {len(data)} 个 HAS_DIFFICULTY_LEVEL 关系")

    def _create_requires_relationships(self, recipes: List[Dict[str, Any]]):
        """创建 REQUIRES 关系：菜谱-食材"""
        data = []
        for recipe in recipes:
            recipe_name = recipe['name']
            for ing in recipe.get('ingredients', []):
                if ing.get('name'):
                    data.append({
                        'recipe_name': recipe_name,
                        'ingredient_name': ing['name'],
                        'amount': ing.get('amount', ''),
                        'unit': ing.get('unit', '')
                    })

        query = """
        UNWIND $batch AS data
        MATCH (r:Recipe {name: data.recipe_name})
        MATCH (i:Ingredient {name: data.ingredient_name})
        MERGE (r)-[rel:REQUIRES]->(i)
        SET rel.amount = data.amount, rel.unit = data.unit
        """
        self.client.batch_execute(query, data, batch_size=500)
        logger.info(f"创建了 {len(data)} 个 REQUIRES 关系")

    def _create_contains_step_relationships(self, recipes: List[Dict[str, Any]]):
        """创建 CONTAINS_STEP 关系：菜谱-步骤"""
        data = []
        for recipe in recipes:
            recipe_name = recipe['name']
            for step in recipe.get('steps', []):
                data.append({
                    'recipe_name': recipe_name,
                    'step_id': f"{recipe_name}_{step['step_id']}",
                    'step_order': step['step_order']
                })

        query = """
        UNWIND $batch AS data
        MATCH (r:Recipe {name: data.recipe_name})
        MATCH (s:CookingStep {step_id: data.step_id})
        MERGE (r)-[rel:CONTAINS_STEP]->(s)
        SET rel.step_order = data.step_order
        """
        self.client.batch_execute(query, data, batch_size=500)
        logger.info(f"创建了 {len(data)} 个 CONTAINS_STEP 关系")

    def _print_statistics(self):
        """打印图谱统计信息"""
        print("\n" + "="*50)
        print("知识图谱统计信息")
        print("="*50)

        stats = [
            ("Recipe", "菜谱节点"),
            ("Ingredient", "食材节点"),
            ("RecipeCategory", "分类节点"),
            ("DifficultyLevel", "难度等级节点"),
            ("CookingStep", "烹饪步骤节点"),
        ]

        for label, name in stats:
            count = self.client.get_node_count(label)
            print(f"  {name:20s}: {count:>6}")

        print("\n关系统计:")
        relations = [
            ("REQUIRES", "菜谱-食材"),
            ("CONTAINS_STEP", "菜谱-步骤"),
            ("HAS_DIFFICULTY_LEVEL", "菜谱-难度"),
            ("BELONGS_TO_CATEGORY", "菜谱-分类"),
        ]

        for rel_type, name in relations:
            count = self.client.get_relationship_count(rel_type)
            print(f"  {name:20s}: {count:>6}")

        print("="*50 + "\n")
