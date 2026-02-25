"""
RAG系统配置文件
"""
from dataclasses import dataclass, field
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGconfig:
    """配置类"""
    # 数据路径配置
    data_path: str = "data/cook"
    index_save_path: str = "./vector_index"
    # 模型配置
    embedding_model_name:str = "BAAI/bge-small-zh-v1.5"
    llm_model_name:str = "kimi-k2-0711-preview"
    # 检索配置
    top_k:int = 3
    # 生成配置
    temperature: float = 0.1
    max_tokens:int = 2048

    # 日志配置
    enable_file_log: bool = True
    enable_console_log: bool = False
    log_file: str = "logs/recipe_rag.log"
    log_level: str = "INFO"

    # Neo4j图数据库配置
    enable_graph: bool = True
    neo4j_uri: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user: str = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', 'password')

    # 混合检索配置
    retrieval_weight: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.7,
        "graph": 0.3
    })


    def __post_init__(self):
        """初始化后的处理"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGconfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model_name': self.embedding_model_name,
            'llm_model_name': self.llm_model_name,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'enable_file_log': self.enable_file_log,
            'enable_console_log': self.enable_console_log,
            'log_file': self.log_file,
            'log_level': self.log_level,
            'enable_graph': self.enable_graph,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'retrieval_weight': self.retrieval_weight
        }

# 默认配置实例
DEFAULT_CONFIG = RAGconfig()