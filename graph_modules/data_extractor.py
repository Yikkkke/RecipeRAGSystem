import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from rag_modules.data_preparation import DataPreparationModule

logger = logging.getLogger(__name__)


class RecipeDataExtractor:
    """从菜谱Markdown文档中提取结构化数据"""

    CATEGORY_MAPPING = DataPreparationModule.CATEGORY_MAPPING

    def __init__(self, documents: List[Document]):
        """
        初始化数据提取器

        Args:
            documents: 已加载的文档列表
        """
        self.documents = documents

    def extract_all(self) -> Dict[str, Any]:
        """
        提取所有菜谱数据

        Returns:
            包含所有提取数据的字典
        """
        recipes = []
        categories = set()
        difficulties = []

        logger.info(f"开始提取 {len(self.documents)} 个菜谱的数据...")

        for doc in self.documents:
            recipe_data = self._extract_recipe(doc)
            if recipe_data:
                recipes.append(recipe_data)
                categories.add(recipe_data['category'])
                difficulties.append(recipe_data['difficulty_star'])

        logger.info(f"数据提取完成: {len(recipes)} 个菜谱, {len(categories)} 个分类")

        return {
            'recipes': recipes,
            'categories': list(categories),
            'difficulty_levels': sorted(list(set(difficulties)))
        }

    def _extract_recipe(self, doc: Document) -> Optional[Dict[str, Any]]:
        """
        提取单个菜谱的数据

        Args:
            doc: 文档对象

        Returns:
            菜谱数据字典
        """
        content = doc.page_content
        metadata = doc.metadata

        recipe_data = {
            'name': metadata.get('dish_name', ''),
            'category': metadata.get('category', '其他'),
            'difficulty_star': self._extract_difficulty_star(content),
            'difficulty_level': self._star_to_level(metadata.get('difficulty', '未知')),
            'ingredients': self._extract_ingredients(content),
            'steps': self._extract_steps(content)
        }

        if not recipe_data['name']:
            logger.warning(f"跳过无效菜谱: {metadata.get('source', '未知')}")
            return None

        return recipe_data

    def _extract_difficulty_star(self, content: str) -> int:
        """
        从内容中提取难度星级

        Args:
            content: 文档内容

        Returns:
            星级数量 (1-5)
        """
        match = re.search(r'预估烹饪难度[：:]\s*(★+)', content)
        if match:
            stars = match.group(1)
            return min(len(stars), 5)
        return 3

    def _star_to_level(self, difficulty_text: str) -> str:
        """
        将难度文本转换为等级描述

        Args:
            difficulty_text: 难度文本

        Returns:
            等级描述
        """
        mapping = {
            '非常简单': '1-星',
            '简单': '2-星',
            '中等': '3-星',
            '困难': '4-星',
            '非常困难': '5-星'
        }
        return mapping.get(difficulty_text, '3-星')

    def _extract_ingredients(self, content: str) -> List[Dict[str, str]]:
        """
        提取食材列表

        Args:
            content: 文档内容

        Returns:
            食材列表，包含名称、数量、单位
        """
        ingredients = []

        section_match = re.search(r'##\s*必备原料和工具\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if not section_match:
            return ingredients

        section = section_match.group(1)
        lines = [line.strip() for line in section.split('\n') 
                 if line.strip() and (line.startswith('-') or line.startswith('*'))]

        for line in lines:
            line = line.lstrip('-*').strip()

            if re.search(r'^[【\[（(].*[）)\]】]', line):
                continue

            ingredient_info = self._parse_ingredient_line(line)
            if ingredient_info['name']:
                ingredients.append(ingredient_info)

        return ingredients

    def _parse_ingredient_line(self, line: str) -> Dict[str, str]:
        """
        解析食材行

        Args:
            line: 食材行

        Returns:
            食材信息字典
        """
        ingredient_info = {'name': '', 'amount': '', 'unit': ''}

        match = re.search(r'^([^-]+)(?:（([^）]+)）|\(([^)]+)\))?\s*(\d+(?:\.\d+)?)?\s*([a-zA-Z\u4e00-\u9fa5]*[克ml个只条片块根根把勺杯碗盘斤两升毫升]*[a-zA-Z\u4e00-\u9fa5]*[克ml个只条片块根根把勺杯碗盘斤两升毫升]*)?\s*[-：:]*', line)

        if match:
            name = match.group(1).strip()
            amount_match = match.group(4)
            unit_match = match.group(5)

            if name and len(name) >= 2 and name not in ['饮用水', '食用油', '盐', '糖', '食用油（推荐品牌', '燃气灶', '锅', '碗与盘子', '筷子', '炒勺', '洗涤剂', '抹布', '钢丝球', '菜刀']:
                ingredient_info['name'] = name

                if amount_match:
                    ingredient_info['amount'] = amount_match

                if unit_match and unit_match not in ['推荐', '推荐品牌', '别称', '选用', '为佳']:
                    ingredient_info['unit'] = unit_match
                elif amount_match and not unit_match:
                    unit_match = re.search(r'([a-zA-Z\u4e00-\u9fa5]+[克ml个只条片块根根把勺杯碗盘斤两升毫升]*)', line)
                    if unit_match:
                        ingredient_info['unit'] = unit_match.group(1)

        return ingredient_info

    def _extract_steps(self, content: str) -> List[Dict[str, Any]]:
        """
        提取操作步骤

        Args:
            content: 文档内容

        Returns:
            步骤列表
        """
        steps = []

        section_match = re.search(r'##\s*操作\s*\n(.*?)(?=\n##|\n##\s*附加内容|\Z)', content, re.DOTALL)
        if not section_match:
            return steps

        section = section_match.group(1)

        lines = []
        for line in section.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or re.match(r'^\d+[\.\、]', line)):
                lines.append(line)

        for idx, line in enumerate(lines, start=1):
            step_text = re.sub(r'^[-\d\s\.\、]+\s*', '', line)
            step_text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', step_text)
            step_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', step_text)

            if step_text:
                steps.append({
                    'step_id': f"step_{idx}",
                    'step_order': idx,
                    'description': step_text.strip()
                })

        return steps

    def get_statistics(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取提取数据的统计信息

        Args:
            graph_data: 提取的数据

        Returns:
            统计信息
        """
        recipes = graph_data.get('recipes', [])
        total_ingredients = sum(len(r.get('ingredients', [])) for r in recipes)
        total_steps = sum(len(r.get('steps', [])) for r in recipes)

        unique_ingredients = set()
        for recipe in recipes:
            for ing in recipe.get('ingredients', []):
                if ing.get('name'):
                    unique_ingredients.add(ing['name'])

        return {
            'total_recipes': len(recipes),
            'total_ingredients': total_ingredients,
            'unique_ingredients': len(unique_ingredients),
            'total_steps': total_steps,
            'categories': graph_data.get('categories', []),
            'difficulty_levels': graph_data.get('difficulty_levels', [])
        }
