import logging
from neo4j import GraphDatabase, Driver
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j数据库客户端"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j客户端

        Args:
            uri: Neo4j数据库URI，如 "bolt://localhost:7687"
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[Driver] = None

    def connect(self) -> bool:
        """连接到Neo4j数据库"""
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            logger.info(f"成功连接到Neo4j数据库: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()
            logger.info("已关闭Neo4j数据库连接")

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行Cypher查询

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if not self._driver:
            raise ConnectionError("数据库未连接，请先调用connect()方法")

        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            raise

    def execute_write_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """
        执行写操作查询

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果
        """
        if not self._driver:
            raise ConnectionError("数据库未连接，请先调用connect()方法")

        try:
            with self._driver.session() as session:
                return session.execute_write(lambda tx: tx.run(query, parameters or {}).consume().counters)
        except Exception as e:
            logger.error(f"执行写操作失败: {e}")
            raise

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            数据库是否可用
        """
        try:
            result = self.execute_query("RETURN 1 AS test")
            return len(result) > 0 and result[0].get('test') == 1
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def batch_execute(self, query: str, data_list: List[Dict], batch_size: int = 1000) -> int:
        """
        批量执行查询

        Args:
            query: Cypher查询语句（使用UNWIND处理批量数据）
            data_list: 数据列表
            batch_size: 每批处理的数据量

        Returns:
            处理的总记录数
        """
        total_processed = 0
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            self.execute_write_query(query, {"batch": batch})
            total_processed += len(batch)
            logger.debug(f"已处理 {total_processed}/{len(data_list)} 条记录")

        logger.info(f"批量执行完成，共处理 {total_processed} 条记录")
        return total_processed

    def get_node_count(self, label: str) -> int:
        """
        获取指定标签的节点数量

        Args:
            label: 节点标签

        Returns:
            节点数量
        """
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        result = self.execute_query(query)
        return result[0].get('count', 0) if result else 0

    def get_relationship_count(self, relationship_type: str) -> int:
        """
        获取指定类型的关系数量

        Args:
            relationship_type: 关系类型

        Returns:
            关系数量
        """
        query = f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) AS count"
        result = self.execute_query(query)
        return result[0].get('count', 0) if result else 0

    def clear_database(self):
        """清空数据库（慎用！）"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write_query(query)
        logger.warning("数据库已清空")

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
