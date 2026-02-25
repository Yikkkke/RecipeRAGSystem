import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def format_recipe_result(recipe: Dict[str, Any]) -> str:
    """
    格式化菜谱结果为可读字符串

    Args:
        recipe: 菜谱字典

    Returns:
        格式化后的字符串
    """
    name = recipe.get('name', '未知菜品')
    difficulty = recipe.get('difficulty_description', recipe.get('difficulty_star', '未知'))
    category = recipe.get('category', '未知分类')

    result = f"🍽️  {name}\n"
    result += f"   分类: {category} | 难度: {difficulty}"

    if 'ingredients' in recipe and recipe['ingredients']:
        result += f"\n   食材: {', '.join([ing.get('name', '') for ing in recipe['ingredients'] if ing.get('name')])[:50]}..."

    if 'common_ingredient_names' in recipe:
        result += f"\n   共同食材: {', '.join(recipe['common_ingredient_names'][:5])}"

    return result


def format_search_results(results: List[Dict[str, Any]], max_items: int = 10) -> str:
    """
    格式化搜索结果列表

    Args:
        results: 结果列表
        max_items: 最多显示的项目数

    Returns:
        格式化后的字符串
    """
    if not results:
        return "没有找到相关菜谱。"

    formatted = f"找到 {len(results)} 个相关菜谱:\n\n"
    for i, recipe in enumerate(results[:max_items], 1):
        formatted += f"{i}. {format_recipe_result(recipe)}\n"

    if len(results) > max_items:
        formatted += f"\n...还有 {len(results) - max_items} 个结果"

    return formatted


def extract_ingredient_names(text: str, known_ingredients: List[str]) -> List[str]:
    """
    从文本中提取食材名称

    Args:
        text: 输入文本
        known_ingredients: 已知食材列表

    Returns:
        匹配的食材列表
    """
    matched = []
    text_lower = text.lower()

    for ingredient in known_ingredients:
        if ingredient.lower() in text_lower:
            matched.append(ingredient)

    return matched


def parse_difficulty_query(text: str) -> int:
    """
    解析查询中的难度要求

    Args:
        text: 查询文本

    Returns:
        最大难度星级（1-5），如果没有要求返回5
    """
    difficulty_keywords = {
        '非常简单': 1,
        '简单': 2,
        '中等': 3,
        '困难': 4,
        '非常困难': 5
    }

    for keyword, level in difficulty_keywords.items():
        if keyword in text:
            return level

    return 5


def parse_category_query(text: str, known_categories: List[str]) -> str:
    """
    解析查询中的分类要求

    Args:
        text: 查询文本
        known_categories: 已知分类列表

    Returns:
        分类名称，如果没有要求返回None
    """
    for category in known_categories:
        if category in text:
            return category

    return None
