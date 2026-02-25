#!/usr/bin/env python3
"""
知识图谱构建测试脚本
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from config import RAGconfig
from rag_modules import DataPreparationModule
from graph_modules import Neo4jClient, RecipeDataExtractor, GraphBuilder

load_dotenv()

def test_graph_build():
    """测试图谱构建"""
    print("="*60)
    print("知识图谱构建测试")
    print("="*60)

    # 配置
    config = RAGconfig(
        data_path="data/cook",
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
    )

    print(f"\n配置信息:")
    print(f"  数据路径: {config.data_path}")
    print(f"  Neo4j URI: {config.neo4j_uri}")
    print(f"  Neo4j User: {config.neo4j_user}")

    # 初始化
    print("\n1. 初始化数据模块...")
    data_module = DataPreparationModule(config.data_path)

    print("2. 加载文档...")
    data_module.load_documents()
    print(f"   加载了 {len(data_module.documents)} 个文档")

    print("3. 提取结构化数据...")
    extractor = RecipeDataExtractor(data_module.documents)
    graph_data = extractor.extract_all()

    stats = extractor.get_statistics(graph_data)
    print(f"   菜谱: {stats['total_recipes']}")
    print(f"   食材: {stats['unique_ingredients']}")
    print(f"   步骤: {stats['total_steps']}")

    print("\n4. 连接Neo4j...")
    client = Neo4jClient(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password
    )
    client.connect()

    if client.health_check():
        print("   ✅ Neo4j连接成功")
    else:
        print("   ❌ Neo4j健康检查失败")
        return

    # 清空现有图谱数据
    print("\n5. 清空现有图谱数据...")
    client.clear_database()
    print("   ✅ 已清空现有图谱数据")

    print("\n6. 构建知识图谱...")
    builder = GraphBuilder(client)
    builder.build_schema()
    builder.build_graph(graph_data)

    print("\n6. 验证图谱...")
    from cypher.queries import CypherQueries
    queries = CypherQueries(client)
    graph_stats = queries.get_graph_statistics()

    print("\n图谱统计:")
    for label, count in graph_stats['nodes'].items():
        print(f"  {label}: {count}")

    print("\n关系统计:")
    for rel_type, count in graph_stats['relationships'].items():
        print(f"  {rel_type}: {count}")

    print(f"\n总计: {graph_stats['total_nodes']} 个节点, {graph_stats['total_relationships']} 个关系")

    client.close()

    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)

if __name__ == "__main__":
    try:
        test_graph_build()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
