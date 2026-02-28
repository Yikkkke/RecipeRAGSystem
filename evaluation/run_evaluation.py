"""
评估脚本
用于运行 RecipeRAG 系统的评估
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import RecipeRAGSystem
from evaluation import RecipeRAGEvaluator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 RecipeRAG 评估")
    parser.add_argument(
        "--test-file", type=str, default=None, help="测试数据文件名，如不指定则自动生成"
    )
    parser.add_argument(
        "--sample-size", type=int, default=10, help="生成的测试用例数量（默认10）"
    )
    parser.add_argument("--output", type=str, default=None, help="评估报告输出文件名")
    parser.add_argument("--top-k", type=int, default=3, help="检索的文档数量（默认3）")

    args = parser.parse_args()

    print("=" * 50)
    print("RecipeRAG 评估")
    print("=" * 50)

    # 初始化RAG系统
    print("\n[1/3] 初始化RAG系统...")
    rag = RecipeRAGSystem()
    rag.initialize_system()
    rag.build_knowledge_base()
    print("✅ RAG系统初始化完成")

    # 初始化评估器
    print("\n[2/3] 初始化评估器...")
    evaluator = RecipeRAGEvaluator(rag)
    print("✅ 评估器初始化完成")

    # 运行评估
    print("\n[3/3] 运行评估...")
    results = evaluator.run_evaluation(
        test_file=args.test_file,
        sample_size=args.sample_size,
        output_file=args.output,
        top_k=args.top_k,
    )

    print("\n" + "=" * 50)
    print("评估完成!")
    print("=" * 50)
    print(f"\n检索质量:")
    print(f"  Precision@K: {results['summary']['retrieval']['avg_precision']:.2f}")
    print(f"  MRR: {results['summary']['retrieval']['avg_mrr']:.2f}")
    print(f"\n生成质量:")
    print(f"  完整性: {results['summary']['generation']['avg_completeness']:.1f}/5")
    print(f"  准确性: {results['summary']['generation']['avg_accuracy']:.1f}/5")
    print(f"  实用性: {results['summary']['generation']['avg_usefulness']:.1f}/5")
    print(f"  平均分: {results['summary']['generation']['avg_score']:.1f}/5")


if __name__ == "__main__":
    main()
