"""
CEIEF Neo4j Data Loader
将 cultural_triples.csv 导入 Neo4j 图数据库

用法：python scripts/load_kg_to_neo4j.py
前提：Neo4j 服务已启动，config.yaml 中配置正确
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from src.kg_retriever import KGLoader


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    neo4j_config = config["neo4j"]
    triples_path = config["paths"]["kg_triples"]

    print(f"Loading triples from: {triples_path}")
    print(f"Target Neo4j: {neo4j_config['uri']}")

    loader = KGLoader(neo4j_config)
    count = loader.load_triples_to_neo4j(triples_path)

    print(f"Successfully loaded {count} triples into Neo4j.")
    print("You can verify in Neo4j Browser: MATCH (n)-[r]->(m) RETURN count(r)")


if __name__ == "__main__":
    main()
