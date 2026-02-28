"""
测试数据管理模块
负责测试数据的持久化、加载和生成
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class TestQuery:
    """
    测试查询数据类
    
    Attributes:
        id: 测试用例唯一标识
        text: 测试问题文本
        type: 问题类型 (detail/list/with_filter/meta/general)
        expected_docs: 期望检索到的文档列表（可选，用于人工验证）
        filters: 过滤条件字典（可选，如 {"category": "荤菜", "difficulty": "简单"}）
    """
    id: str
    text: str
    type: str
    expected_docs: Optional[List[str]] = None
    filters: Optional[Dict[str, str]] = None


class TestDataManager:
    """
    测试数据管理器
    
    负责测试数据的持久化：
    - 从菜谱数据生成测试query并保存到文件
    - 从文件加载测试query
    - 列出已有测试数据文件
    
    数据文件保存在 reports/test_data/ 目录下
    """
    
    DEFAULT_DATA_DIR = "reports/test_data"
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        """
        初始化测试数据管理器
        
        Args:
            data_dir: 测试数据文件存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_test_queries(
        self, 
        queries: List[TestQuery], 
        filename: str
    ) -> Path:
        """
        保存测试数据到JSON文件
        
        Args:
            queries: 测试查询列表
            filename: 文件名
        
        Returns:
            保存的文件路径
        """
        filepath = self.data_dir / filename
        
        # 将dataclass转换为字典
        data = [asdict(q) for q in queries]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"测试数据已保存至: {filepath}")
        return filepath
    
    def load_test_queries(self, filename: str) -> List[TestQuery]:
        """
        从JSON文件加载测试数据
        
        Args:
            filename: 文件名
        
        Returns:
            测试查询列表
        
        Raises:
            FileNotFoundError: 文件不存在
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [TestQuery(**item) for item in data]
    
    def list_test_files(self) -> List[str]:
        """
        列出所有测试数据文件
        
        Returns:
            测试数据文件名列表
        """
        return sorted([f.name for f in self.data_dir.glob("*.json")])
    
    def generate_from_data(
        self,
        data_path: str,
        sample_size: int = 20,
        output_file: Optional[str] = None
    ) -> List[TestQuery]:
        """
        从菜谱数据生成测试query，并保存到文件
        
        Args:
            data_path: 菜谱数据目录路径
            sample_size: 生成的测试用例数量
            output_file: 输出文件名，如果为None则自动生成
        
        Returns:
            生成的测试查询列表
        """
        # 1. 加载菜谱数据
        recipes = self._load_recipes(data_path)
        
        if not recipes:
            raise ValueError(f"无法从 {data_path} 加载菜谱数据")
        
        # 2. 生成测试query
        queries = self._generate_queries(recipes, sample_size)
        
        # 3. 如果没有指定文件名，自动生成
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_data_{timestamp}.json"
        
        # 4. 保存到文件
        self.save_test_queries(queries, output_file)
        
        return queries
    
    def _load_recipes(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载菜谱数据
        
        从指定目录加载所有菜谱的元数据（名称、分类、难度）
        
        Args:
            data_path: 菜谱数据目录路径
        
        Returns:
            菜谱列表，每项包含 name, category, difficulty
        """
        # 复用现有的DataPreparationModule加载数据
        from rag_modules import DataPreparationModule
        
        data_module = DataPreparationModule(data_path)
        data_module.load_documents()
        
        recipes = []
        for doc in data_module.documents:
            recipes.append({
                "name": doc.metadata.get("dish_name", "未知"),
                "category": doc.metadata.get("category", ""),
                "difficulty": doc.metadata.get("difficulty", "")
            })
        
        return recipes
    
    def _generate_queries(
        self, 
        recipes: List[Dict[str, Any]], 
        sample_size: int
    ) -> List[TestQuery]:
        """
        从菜谱列表生成测试query
        
        按比例生成不同类型的query：
        - 50% detail: "XXX怎么做"
        - 25% list: "有哪些XXX推荐"
        - 25% with_filter: "简单的XXX"
        
        Args:
            recipes: 菜谱列表
            sample_size: 生成数量
        
        Returns:
            测试查询列表
        """
        # 设置随机种子确保可复现
        random.seed(42)
        
        queries = []
        query_id = 1
        
        # 计算各类型数量
        detail_count = int(sample_size * 0.5)    # 50% detail
        list_count = int(sample_size * 0.25)    # 25% list
        filter_count = sample_size - detail_count - list_count  # 25% filter
        
        # ===== detail类型: 从菜谱名生成 =====
        # 随机采样不重复的菜谱
        sampled_recipes = random.sample(recipes, min(detail_count, len(recipes)))
        for recipe in sampled_recipes:
            queries.append(TestQuery(
                id=f"q{query_id}",
                text=f"{recipe['name']}怎么做",
                type="detail",
                expected_docs=[recipe['name']]
            ))
            query_id += 1
        
        # 如果菜谱数量不足detail_count，补充一些
        while len([q for q in queries if q.type == "detail"]) < detail_count:
            recipe = random.choice(recipes)
            if recipe['name'] not in [q.expected_docs[0] for q in queries if q.expected_docs]:
                queries.append(TestQuery(
                    id=f"q{query_id}",
                    text=f"{recipe['name']}怎么做",
                    type="detail",
                    expected_docs=[recipe['name']]
                ))
                query_id += 1
        
        # ===== list类型: 分类查询 =====
        # 获取所有分类
        categories = list(set(
            r.get("category") for r in recipes 
            if r.get("category")
        ))
        
        for _ in range(list_count):
            cat = random.choice(categories)
            queries.append(TestQuery(
                id=f"q{query_id}",
                text=f"有哪些{cat}推荐",
                type="list"
            ))
            query_id += 1
        
        # ===== filter类型: 难度+分类 =====
        difficulties = ["简单", "中等", "困难"]
        
        for _ in range(filter_count):
            diff = random.choice(difficulties)
            cat = random.choice(categories)
            queries.append(TestQuery(
                id=f"q{query_id}",
                text=f"{diff}的{cat}",
                type="with_filter",
                filters={"difficulty": diff, "category": cat}
            ))
            query_id += 1
        
        return queries
    
    def delete_test_file(self, filename: str) -> bool:
        """
        删除指定的测试数据文件
        
        Args:
            filename: 文件名
        
        Returns:
            是否删除成功
        """
        filepath = self.data_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            print(f"已删除: {filepath}")
            return True
        
        return False
