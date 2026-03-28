"""
CEIEF Knowledge Graph Retriever Module
文化教育交互增强框架 - 知识图谱检索模块

负责从 Neo4j 知识图谱中检索文化知识三元组，构建子图摘要，
支持关键词检索与语义向量检索两种模式。
"""

import json
import csv
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class Triple:
    """知识图谱三元组"""
    subject: str
    relation: str
    obj: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """转换为文本表示"""
        return f"({self.subject}, {self.relation}, {self.obj})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "weight": self.weight,
            "metadata": self.metadata
        }


@dataclass
class SubgraphSummary:
    """子图摘要结构"""
    query: str
    triples: List[Triple]
    entities: List[str]
    relations: List[str]
    summary_text: str
    retrieval_method: str  # 'keyword' / 'semantic' / 'hybrid'

    def to_evidence_list(self) -> List[str]:
        """转换为证据文本列表（用于LLM提示词注入）"""
        return [t.to_text() for t in self.triples]


class KGRetriever:
    """
    CEIEF 知识图谱检索器
    支持两种后端：
    1. Neo4j（生产环境，bolt协议）
    2. CSV文件（轻量/离线模式，直接读取 cultural_triples.csv）
    """

    # 预定义的文化领域关键词映射
    CULTURAL_KEYWORDS = {
        "端午节": ["端午", "龙舟", "粽子", "屈原", "节庆"],
        "春节": ["春节", "新年", "除夕", "年夜饭", "红包", "拜年"],
        "中秋节": ["中秋", "月饼", "嫦娥", "赏月"],
        "清明节": ["清明", "祭祖", "扫墓", "踏青"],
        "孝道": ["孝道", "孝顺", "尊老", "养老", "百善孝为先"],
        "礼仪": ["礼仪", "礼节", "礼貌", "问候", "仪式"],
        "集体记忆": ["集体记忆", "社区记忆", "文化记忆", "共同记忆"],
        "成人礼": ["成人礼", "成年礼", "成丁", "冠礼"],
        "祭祖": ["祭祖", "祭祀", "祭奠", "扫墓"],
        "社区认同": ["社区认同", "文化认同", "群体认同", "身份认同"],
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识图谱检索器

        Args:
            config: 来自 config.yaml 的完整配置
        """
        self.kg_config = config.get("kg_retrieval", {})
        self.neo4j_config = config.get("neo4j", {})
        self.paths = config.get("paths", {})

        self.max_triples = self.kg_config.get("max_triples_per_query", 10)
        self.similarity_threshold = self.kg_config.get("similarity_threshold", 0.6)
        self.max_hop_depth = self.kg_config.get("max_hop_depth", 2)
        self.use_semantic = self.kg_config.get("use_semantic_search", False)

        # 本地三元组存储（CSV备用模式）
        self._triples: List[Triple] = []
        self._neo4j_driver = None
        self._embedding_model = None
        self._embeddings = None
        self._entity_index: Dict[str, List[Triple]] = {}

        # 尝试加载本地三元组数据
        self._load_local_triples()
        logger.info(f"KGRetriever initialized with {len(self._triples)} triples")

    def _load_local_triples(self):
        """从本地 CSV 文件加载知识图谱三元组"""
        triples_path = self.paths.get("kg_triples", "data/kg/cultural_triples.csv")
        path = Path(triples_path)
        if not path.exists():
            logger.warning(f"KG triples file not found: {triples_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    triple = Triple(
                        subject=row.get("subject", "").strip(),
                        relation=row.get("relation", "").strip(),
                        obj=row.get("object", "").strip(),
                        weight=float(row.get("weight", 1.0)),
                        metadata={
                            "domain": row.get("domain", ""),
                            "source": row.get("source", ""),
                            "entity_type_s": row.get("entity_type_subject", ""),
                            "entity_type_o": row.get("entity_type_object", "")
                        }
                    )
                    if triple.subject and triple.relation and triple.obj:
                        self._triples.append(triple)

            # 建立实体索引
            self._build_entity_index()
            logger.info(f"Loaded {len(self._triples)} triples from {triples_path}")

        except Exception as e:
            logger.error(f"Failed to load triples from {triples_path}: {e}")

    def _build_entity_index(self):
        """构建实体到三元组的倒排索引，加速检索"""
        self._entity_index = {}
        for triple in self._triples:
            for entity in [triple.subject, triple.obj]:
                if entity not in self._entity_index:
                    self._entity_index[entity] = []
                self._entity_index[entity].append(triple)

    def _connect_neo4j(self):
        """连接 Neo4j 数据库（按需连接）"""
        if self._neo4j_driver is not None:
            return True
        try:
            from neo4j import GraphDatabase
            uri = self.neo4j_config.get("uri", "bolt://localhost:7687")
            user = self.neo4j_config.get("user", "neo4j")
            password = self.neo4j_config.get("password", "")
            self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            self._neo4j_driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection failed, falling back to CSV mode: {e}")
            return False

    def retrieve_by_keyword(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Triple]:
        """
        关键词检索：基于字符串匹配从本地三元组中检索

        Args:
            query: 查询字符串
            max_results: 返回最大数量

        Returns:
            匹配的三元组列表
        """
        if max_results is None:
            max_results = self.max_triples

        query_lower = query.lower()
        scored_triples: List[Tuple[float, Triple]] = []

        # 扩展查询关键词
        expanded_keywords = self._expand_keywords(query)

        for triple in self._triples:
            score = self._compute_match_score(triple, query_lower, expanded_keywords)
            if score > 0:
                scored_triples.append((score, triple))

        # 按分数降序排列
        scored_triples.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triples[:max_results]]

    def _expand_keywords(self, query: str) -> List[str]:
        """扩展查询关键词（利用文化领域词典）"""
        keywords = [query]
        for concept, synonyms in self.CULTURAL_KEYWORDS.items():
            if concept in query or any(s in query for s in synonyms):
                keywords.extend(synonyms)
                keywords.append(concept)
        return list(set(keywords))

    def _compute_match_score(
        self,
        triple: Triple,
        query: str,
        keywords: List[str]
    ) -> float:
        """计算三元组与查询的匹配分数"""
        score = 0.0
        triple_text = f"{triple.subject} {triple.relation} {triple.obj}".lower()

        # 直接字符串匹配
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in triple.subject.lower():
                score += 2.0
            if kw_lower in triple.obj.lower():
                score += 1.5
            if kw_lower in triple.relation.lower():
                score += 0.5
            if kw_lower in triple_text:
                score += 0.3

        return score * triple.weight

    def retrieve_from_neo4j(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Triple]:
        """
        从 Neo4j 数据库检索三元组

        Args:
            query: 查询字符串（实体名称或关键词）
            max_results: 最大返回数量

        Returns:
            检索到的三元组列表
        """
        if max_results is None:
            max_results = self.max_triples

        if not self._connect_neo4j():
            logger.warning("Neo4j unavailable, falling back to keyword search")
            return self.retrieve_by_keyword(query, max_results)

        try:
            cypher = """
            MATCH (s)-[r]->(o)
            WHERE s.name CONTAINS $query OR o.name CONTAINS $query
               OR r.type CONTAINS $query
            RETURN s.name AS subject, type(r) AS relation, o.name AS object,
                   r.weight AS weight, s.entity_type AS entity_type_s,
                   o.entity_type AS entity_type_o
            LIMIT $limit
            """
            with self._neo4j_driver.session() as session:
                result = session.run(cypher, query=query, limit=max_results)
                triples = []
                for record in result:
                    triple = Triple(
                        subject=record["subject"] or "",
                        relation=record["relation"] or "",
                        obj=record["object"] or "",
                        weight=record["weight"] or 1.0,
                        metadata={
                            "entity_type_s": record["entity_type_s"] or "",
                            "entity_type_o": record["entity_type_o"] or ""
                        }
                    )
                    triples.append(triple)
                logger.info(f"Neo4j retrieved {len(triples)} triples for query: '{query}'")
                return triples

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}, falling back to keyword search")
            return self.retrieve_by_keyword(query, max_results)

    def retrieve_subgraph(
        self,
        query: str,
        use_neo4j: bool = True
    ) -> SubgraphSummary:
        """
        检索并构建子图摘要

        Args:
            query: 学习者查询文本
            use_neo4j: 是否优先使用 Neo4j（否则直接用 CSV）

        Returns:
            SubgraphSummary 对象
        """
        if use_neo4j and self._connect_neo4j():
            triples = self.retrieve_from_neo4j(query)
            method = "neo4j"
        else:
            triples = self.retrieve_by_keyword(query)
            method = "keyword"

        if not triples:
            logger.warning(f"No triples found for query: '{query}'")
            triples = []

        # 提取实体和关系
        entities = list(set(
            [t.subject for t in triples] + [t.obj for t in triples]
        ))
        relations = list(set([t.relation for t in triples]))

        # 生成子图摘要文本
        summary_text = self._build_summary_text(query, triples)

        return SubgraphSummary(
            query=query,
            triples=triples,
            entities=entities,
            relations=relations,
            summary_text=summary_text,
            retrieval_method=method
        )

    def _build_summary_text(self, query: str, triples: List[Triple]) -> str:
        """将检索到的三元组构建为自然语言摘要"""
        if not triples:
            return f"No cultural knowledge found for: {query}"

        lines = [f"Cultural knowledge retrieved for '{query}':"]
        for t in triples:
            lines.append(f"  {t.to_text()}")

        return "\n".join(lines)

    def get_neighbors(
        self,
        entity: str,
        relation_filter: Optional[List[str]] = None
    ) -> List[Triple]:
        """
        获取指定实体的邻居三元组（1-hop）

        Args:
            entity: 实体名称
            relation_filter: 关系类型过滤器

        Returns:
            邻居三元组列表
        """
        neighbors = self._entity_index.get(entity, [])
        if relation_filter:
            neighbors = [t for t in neighbors if t.relation in relation_filter]
        return neighbors[:self.max_triples]

    def close(self):
        """关闭数据库连接"""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            logger.info("Neo4j connection closed")


class KGLoader:
    """
    知识图谱数据加载器
    用于将 CSV 数据批量导入 Neo4j
    """

    def __init__(self, neo4j_config: Dict[str, Any]):
        self.config = neo4j_config

    def load_triples_to_neo4j(self, csv_path: str) -> int:
        """
        将三元组 CSV 批量导入 Neo4j

        Args:
            csv_path: CSV 文件路径

        Returns:
            成功导入的三元组数量
        """
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"])
            )

            count = 0
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            with driver.session() as session:
                # 创建约束（仅首次运行）
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CulturalEntity) "
                    "REQUIRE n.name IS UNIQUE"
                )

                for row in rows:
                    subject = row.get("subject", "").strip()
                    relation = row.get("relation", "").strip()
                    obj = row.get("object", "").strip()
                    domain = row.get("domain", "general")
                    weight = float(row.get("weight", 1.0))
                    etype_s = row.get("entity_type_subject", "concept")
                    etype_o = row.get("entity_type_object", "concept")

                    if not (subject and relation and obj):
                        continue

                    cypher = """
                    MERGE (s:CulturalEntity {name: $subject})
                    SET s.entity_type = $etype_s, s.domain = $domain
                    MERGE (o:CulturalEntity {name: $obj})
                    SET o.entity_type = $etype_o, o.domain = $domain
                    MERGE (s)-[r:RELATION {type: $relation}]->(o)
                    SET r.weight = $weight, r.domain = $domain
                    """
                    session.run(cypher, subject=subject, obj=obj,
                                relation=relation, domain=domain,
                                weight=weight, etype_s=etype_s, etype_o=etype_o)
                    count += 1

            driver.close()
            logger.info(f"Loaded {count} triples into Neo4j")
            return count

        except Exception as e:
            logger.error(f"Failed to load triples to Neo4j: {e}")
            return 0


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    retriever = KGRetriever(config)
    subgraph = retriever.retrieve_subgraph("端午节", use_neo4j=False)
    print(f"Retrieved {len(subgraph.triples)} triples:")
    for t in subgraph.triples:
        print(f"  {t.to_text()}")
    print(f"\nSummary:\n{subgraph.summary_text}")
    retriever.close()
