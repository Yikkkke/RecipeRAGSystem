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
    GraphVectorHybridRetriever,
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

                # 初始化图谱+向量混合检索器
                self.hybrid_retriever = GraphVectorHybridRetriever(
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

        # 5. 更新混合检索器的向量检索器
        if self.hybrid_retriever and self.retrieval_module:
            self.hybrid_retriever.vector_retriever = (
                self.retrieval_module.vector_retriever
            )

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

    def answer_query(self, query: str, stream: bool = False):
        """
        回答用户问题
        Args:
            query: 用户问题
            stream: 是否使用流式输出，即一边想一边回答

        Returns:
            非流式: (生成的回答, 检索到的文档列表, 路由类型) 元组
            流式: 生成器
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

        # 3. 检索相关子块（统一的检索接口）
        filters = self._extract_filters_from_query(query)  # 采用原始query提取元数据

        relevant_chunks = self.retrieve_documents(
            query=query,
            query_type=route_type,
            filters=filters,
            rewritten_query=rewritten_query,
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
                    section_title = content_preview[:title_end].strip("#").strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")
            print(f"找到的文档块：{', '.join(chunk_info)}")
        else:
            # 4. 没有检索到相关文档块，停止继续查找生成答案
            if stream:
                return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            else:
                return (
                    "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。",
                    [],
                    route_type,
                )

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
            if stream:
                return answer
            return answer, relevant_chunks, route_type
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
                answer = self.generation_module.generate_step_by_step_answer(
                    query=rewritten_query, context_docs=relevant_docs
                )
                return answer, relevant_chunks, route_type
        else:
            # general 类型：使用基础回答生成
            print("🤖 生成基础回答...")
            if stream:
                answer_generator = self.generation_module.generate_basic_answer_stream(
                    query=rewritten_query, context_docs=relevant_chunks
                )
                return answer_generator
            else:
                answer = self.generation_module.generate_basic_answer(
                    query=rewritten_query, context_docs=relevant_docs
                )
                return answer, relevant_chunks, route_type

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
        else:
            # general 类型：使用基础回答生成
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

    def _difficulty_to_star(self, difficulty: str):
        """难度文本转星级"""
        mapping = {"简单": 2, "中等": 3, "困难": 4, "非常困难": 5}
        return mapping.get(difficulty)

    def _convert_graph_to_docs(self, graph_results):
        """将图谱结果转换为 Document 对象"""
        from langchain_core.documents import Document

        docs = []
        for result in graph_results:
            dish_name = result.get("name")
            if not dish_name:
                continue
            if self.data_module and self.data_module.documents:
                for doc in self.data_module.documents:
                    if doc.metadata.get("dish_name") == dish_name:
                        docs.append(doc)
                        break
        return docs

    def retrieve_documents(
        self,
        query: str,
        query_type: str = None,
        filters: dict = None,
        rewritten_query: str = None,
    ) -> List:
        """
        统一的文档检索接口

        Args:
            query: 原始查询文本
            query_type: 查询类型 (detail/list/general)
            filters: 过滤条件字典
            rewritten_query: 重写后的查询（用于向量检索）

        Returns:
            检索到的文档列表
        """
        if filters is None:
            filters = {}
        if not query_type:
            query_type = "detail"

        # 将 with_filter 视为 list 类型
        if query_type == "with_filter":
            query_type = "list"

        # 根据 enable_graph + query_type + filters 选择检索方式
        if self.config.enable_graph and self.graph_retriever:
            # 开启图谱
            if query_type == "detail":
                # detail 类型使用向量检索（需要具体内容）
                if filters:
                    print(f"🔍 使用向量检索+过滤: {filters}")
                    return self.retrieval_module.metadata_filtered_search(
                        query=rewritten_query or query,
                        metadata_filters=filters,
                        top_k=self.config.top_k,
                    )
                else:
                    print("🔍 使用向量混合检索")
                    return self.retrieval_module.hybrid_search(
                        query=rewritten_query or query, top_k=self.config.top_k
                    )
            else:
                # list/general 类型
                if filters:
                    # 有过滤条件 → 图谱精确匹配
                    print(f"🔍 使用图谱检索 (精确匹配): {filters}")
                    graph_results = self.graph_retriever.search_recipes(
                        category=filters.get("category"),
                        max_difficulty=self._difficulty_to_star(
                            filters.get("difficulty")
                        ),
                        limit=self.config.top_k,
                    )
                    return self._convert_graph_to_docs(graph_results)
                else:
                    # 无过滤条件 → 图谱+向量混合检索
                    print("🔍 使用图谱+向量混合检索")
                    graph_results = self.hybrid_retriever.search(
                        query=rewritten_query or query, top_k=self.config.top_k
                    )
                    return self._convert_graph_to_docs(graph_results)
        else:
            # 未开启图谱 → 统一使用向量检索
            if filters:
                print(f"🔍 使用向量检索+过滤: {filters}")
                return self.retrieval_module.metadata_filtered_search(
                    query=rewritten_query or query,
                    metadata_filters=filters,
                    top_k=self.config.top_k,
                )
            else:
                print("🔍 使用向量混合检索")
                return self.retrieval_module.hybrid_search(
                    query=rewritten_query or query, top_k=self.config.top_k
                )

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品 [已废弃]

        此方法已废弃，请使用 query() 方法统一检索入口。
        查询"简单的甜品"等带过滤条件的查询会自动使用图谱检索。

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        raise NotImplementedError(
            "search_by_category() 已废弃，请使用 query() 方法进行检索。"
            "带过滤条件的查询（如'简单的甜品'）会自动使用图谱检索。"
        )

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
        根据食材搜索菜谱 [已废弃]

        此方法已废弃，请使用 query() 方法统一检索入口。
        """
        raise NotImplementedError(
            "search_by_ingredients() 已废弃，请使用 query() 方法进行检索。"
        )

    def search_similar_recipes(self, recipe_name: str, limit: int = 5) -> str:
        """
        查找相似菜谱 [已废弃]

        此方法已废弃，请使用 query() 方法统一检索入口。
        """
        raise NotImplementedError(
            "search_similar_recipes() 已废弃，请使用 query() 方法进行检索。"
        )

    def get_recipe_graph_details(self, recipe_name: str) -> str:
        """
        获取菜谱的图谱详情 [已废弃]

        此方法已废弃，请使用 query() 方法统一检索入口。
        """
        raise NotImplementedError(
            "get_recipe_graph_details() 已废弃，请使用 query() 方法进行检索。"
        )

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

                # 统一使用 answer_query 方法处理所有查询
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
