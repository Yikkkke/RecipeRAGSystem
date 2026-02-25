"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGconfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)
from graph_modules import (
    Neo4jClient,
    RecipeDataExtractor,
    GraphBuilder,
    GraphRetriever,
    HybridRetriever,
)
from graph_modules.utils import (
    format_search_results,
    extract_ingredient_names,
    parse_difficulty_query,
    parse_category_query,
)

load_dotenv()

# 配置日志记录（可通过 `DEFAULT_CONFIG` 控制是否输出到文件/终端）
log_level = getattr(logging, DEFAULT_CONFIG.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
root_logger = logging.getLogger()

# 处理终端输出（StreamHandler）
if not DEFAULT_CONFIG.enable_console_log:
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler):
            root_logger.removeHandler(h)
else:
    # 确保至少有一个 StreamHandler
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(sh)

# 处理文件输出（FileHandler）
log_file_path = Path(DEFAULT_CONFIG.log_file)
if DEFAULT_CONFIG.enable_file_log:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    found = False
    for h in root_logger.handlers:
        if getattr(h, "baseFilename", None) == str(log_file_path):
            found = True
            break
    if not found:
        fh = logging.FileHandler(filename=str(log_file_path), encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(fh)
else:
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(
            h, "baseFilename", None
        ) == str(log_file_path):
            root_logger.removeHandler(h)

logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """RAG系统主程序"""

    def __init__(self, config: RAGconfig = DEFAULT_CONFIG):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 图谱相关
        self.graph_client = None
        self.graph_retriever = None
        self.hybrid_retriever = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在：{self.config.data_path}")

        # 检查api密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """初始化RAGModules的所有模块"""
        print("🚀 正在初始化RAG系统...")

        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            self.config.embedding_model_name, self.config.index_save_path
        )
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # 初始化图数据库
        if self.config.enable_graph:
            print("🕸️  初始化图数据库客户端...")
            try:
                self.graph_client = Neo4jClient(
                    uri=self.config.neo4j_uri,
                    user=self.config.neo4j_user,
                    password=self.config.neo4j_password,
                )
                self.graph_client.connect()

                # 初始化图检索器
                self.graph_retriever = GraphRetriever(self.graph_client)

                # 初始化混合检索器
                self.hybrid_retriever = HybridRetriever(
                    graph_retriever=self.graph_retriever,
                    vector_retriever=None,
                    weights=self.config.retrieval_weight,
                )

                print("✅ 图数据库连接成功")
            except Exception as e:
                logger.warning(f"图数据库初始化失败: {e}")
                self.config.enable_graph = False
                print("⚠️  图数据库初始化失败，将仅使用向量检索")
                print("⚠️  ", e)

        print("✅ 系统初始化完成")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1.加载文档和分块用于检索模块
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()

        # 2. 尝试加载已保存的索引
        vectorstore = self.index_module.load_vector_index()
        if vectorstore is None:
            print("没有找到已保存的索引，开始构建新索引并保存......")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_vector_index()

        # 3. 初始化检索模块
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore=vectorstore, chunks=chunks
        )

        # 4. 显示知识库的统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def build_knowledge_graph(self):
        """构建知识图谱"""
        if not self.config.enable_graph:
            print("⚠️  图数据库未启用，跳过图谱构建")
            return

        if not self.graph_client:
            print("⚠️  图数据库未连接，跳过图谱构建")
            return

        print("\n🕸️  开始构建知识图谱...")

        try:
            # 确保数据已加载
            if not self.data_module.documents:
                self.data_module.load_documents()

            # 提取数据
            print("正在提取菜谱结构化数据...")
            extractor = RecipeDataExtractor(self.data_module.documents)
            graph_data = extractor.extract_all()

            # 显示提取统计
            stats = extractor.get_statistics(graph_data)
            print(f"  提取统计:")
            print(f"    菜谱总数: {stats['total_recipes']}")
            print(f"    食材总数: {stats['total_ingredients']}")
            print(f"    唯一食材: {stats['unique_ingredients']}")
            print(f"    步骤总数: {stats['total_steps']}")

            # 构建图谱
            print("\n正在构建图谱...")
            builder = GraphBuilder(self.graph_client)
            builder.build_schema()
            builder.build_graph(graph_data)

            print("✅ 知识图谱构建完成！")

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            print(f"❌ 构建知识图谱失败: {e}")

    def answer_query(self, query: str, stream: bool = False) -> str:
        """
        回答用户问题
        Args:
            query: 用户问题
            stream: 是否使用流式输出，即一边想一边回答

        Returns:
            生成的回答或者生成器
        """
        if self.retrieval_module is None or self.generation_module is None:
            raise ValueError("请先初始化RAG系统并构建知识库")

        print(f"\n❓ 用户问题: {query}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(query=query)

        # 2. 智能查询重写（根据路由类型判断是否需要重写）
        if route_type == "list":
            rewritten_query = query
            print(f"📝 列表查询保持原样: {query}")
        else:
            # 采用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(query)

        # 3. 检索相关子块（+自动应用元数据过滤）
        filters = self._extract_filters_from_query(query)  # 采用原始query提取元数据
        if filters:
            print(f"🔍 应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                query=rewritten_query,  # 采用重写后的query进行检索
                metadata_filters=filters,
                top_k=self.config.top_k,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                query=rewritten_query, top_k=self.config.top_k
            )
        ## 显示检索到的子块信息
        print(f"找到 {len(relevant_chunks)} 个相关文档块")
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:50].replace("\n", " ").strip()
                if content_preview.startswith("#"):
                    # 如果是标题开头，提取标题
                    title_end = (
                        content_preview.find("\n")
                        if "\n" in chunk.page_content[:100]
                        else len(content_preview)
                    )
                    section_title = chunk.page_content[:title_end].strip("#").strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")
            print(f"找到的文档块：{', '.join(chunk_info)}")
        else:
            # 4. 没有检索到相关文档块，停止继续查找生成答案
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 获取父文档（所有相关的完整菜谱文档）
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        ### 显示找到的文档名称
        doc_names = []
        for doc in relevant_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            doc_names.append(dish_name)
        if doc_names:
            logger.info(f"找到文档: {', '.join(doc_names)}")

        # 6. 根据路由类型选择回答方式
        if route_type == "list":
            # 列表查询：直接返回菜品名称列表
            logger.info("📝 生成列表式回答...")
            answer = self.generation_module.generate_list_answer(
                query=rewritten_query, context_docs=relevant_docs
            )
            return answer
        elif route_type == "detail":
            print("🤖 生成菜谱详情回答...")
            if stream:
                answer_generator = (
                    self.generation_module.generate_step_by_step_answer_stream(
                        query=rewritten_query, context_docs=relevant_chunks
                    )
                )
                return answer_generator
            else:
                return self.generation_module.generate_step_by_step_answer(
                    query=rewritten_query, context_docs=relevant_docs
                )
        elif route_type == "meta":
            # 元问题：跳过检索，直接生成回答
            print("🤖 生成元问题回答...")
            return self.generation_module.generate_meta_answer(query)
        else:
            print("🤖 生成基础回答...")
            if stream:
                answer_generator = self.generation_module.generate_basic_answer_stream(
                    query=rewritten_query, context_docs=relevant_chunks
                )
                return answer_generator
            else:
                return self.generation_module.generate_basic_answer(
                    query=rewritten_query, context_docs=relevant_docs
                )

    def _extract_filters_from_query(self, query: str):
        """
        从查询中提取元数据过滤条件（如菜系、难度等）

        Args:
            query: 用户查询

        Returns:
            过滤条件字典
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters["category"] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters["difficulty"] = diff
                break

        return filters

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}

        docs = self.retrieval_module.metadata_filtered_search(
            search_query, filters, top_k=10
        )

        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(
            f"{dish_name}需要什么食材？", docs
        )

        return answer

    def search_by_ingredients(self, ingredient_names: list, limit: int = 10) -> str:
        """
        根据食材搜索菜谱（使用图谱）

        Args:
            ingredient_names: 食材名称列表
            limit: 返回结果数量

        Returns:
            格式化的菜谱列表
        """
        if not self.graph_retriever:
            return "图检索未启用，请先启用图数据库"

        results = self.graph_retriever.find_recipes_by_ingredients(
            ingredient_names, limit
        )
        return format_search_results(results, limit)

    def search_similar_recipes(self, recipe_name: str, limit: int = 5) -> str:
        """
        查找相似菜谱（使用图谱）

        Args:
            recipe_name: 菜谱名称
            limit: 返回结果数量

        Returns:
            格式化的相似菜谱列表
        """
        if not self.graph_retriever:
            return "图检索未启用，请先启用图数据库"

        results = self.graph_retriever.find_similar_recipes(recipe_name, limit)
        return format_search_results(results, limit)

    def get_recipe_graph_details(self, recipe_name: str) -> str:
        """
        获取菜谱的图谱详情

        Args:
            recipe_name: 菜谱名称

        Returns:
            格式化的菜谱详情
        """
        if not self.graph_retriever:
            return "图检索未启用，请先启用图数据库"

        details = self.graph_retriever.get_recipe_details(recipe_name)

        if not details:
            return f"未找到菜谱: {recipe_name}"

        result = f"🍽️  {details['name']}\n"
        result += f"   分类: {details['category']} | 难度: {details['difficulty_description']} ({details['difficulty_star']}★)\n\n"
        result += f"🥘 食材:\n"
        for ing in details.get("ingredients", []):
            if ing.get("name"):
                amount = f" {ing.get('amount', '')}" if ing.get("amount") else ""
                unit = f" {ing.get('unit', '')}" if ing.get("unit") else ""
                result += f"   - {ing['name']}{amount}{unit}\n"

        if details.get("steps") and len(details["steps"]) > 0:
            result += f"\n📝 制作步骤:\n"
            for step in details["steps"]:
                result += f"   {step['order']}. {step['description'][:60]}...\n"

        return result

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        # 询问是否构建知识图谱
        if self.config.enable_graph:
            build_graph = input("\n是否构建知识图谱? (y/n, 默认n): ").strip().lower()
            if build_graph == "y":
                self.build_knowledge_graph()

        print("\n交互式问答 (输入'退出'结束):")

        # 询问是否使用流式输出
        stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
        while True:
            try:
                user_input = input("\n您的问题：   ").strip()
                if user_input.lower() in ["退出", "exit", "quit", ""]:
                    break
                use_stream = stream_choice != "n"

                # 检查是否是图谱查询
                if user_input.startswith("按食材查找") or user_input.startswith(
                    "相似菜谱"
                ):
                    print("\n回答：")
                    if "按食材查找" in user_input:
                        ingredients_str = user_input.replace("按食材查找", "").strip()
                        if ingredients_str:
                            ingredients = [
                                ing.strip() for ing in ingredients_str.split(",")
                            ]
                            answer = self.search_by_ingredients(ingredients)
                        else:
                            answer = "请输入食材名称，例如：按食材查找 土豆,洋葱"
                    elif "相似菜谱" in user_input:
                        recipe_name = user_input.replace("相似菜谱", "").strip()
                        if recipe_name:
                            answer = self.search_similar_recipes(recipe_name)
                        else:
                            answer = "请输入菜谱名称，例如：相似菜谱 清蒸鲈鱼"
                    print(answer)
                else:
                    print("\n回答：")
                    if use_stream:
                        # 流式输出
                        for chunk in self.answer_query(user_input, stream=True):
                            print(chunk, end="", flush=True)
                        print("\n")
                    else:
                        # 普通输出
                        answer = self.answer_query(user_input, stream=False)
                        print(answer)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
