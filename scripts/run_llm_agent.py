"""
CEIEF LLM+Agent Experiment Runner
运行 LLM+Agent 条件实验（无KG，有Agent）

用法：python scripts/run_llm_agent.py --task_id T001 --turns 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from src.dialogue_engine import DialogueEngine, CONDITION_LLM_AGENT


def run_session(task_id: str, max_turns: int = 5, config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open("data/tasks/task_bank.json", "r", encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in json.load(f)["tasks"]}
    task = tasks[task_id]

    engine = DialogueEngine(config)
    session_id = engine.start_session(
        condition=CONDITION_LLM_AGENT,
        task_family=task["family"],
        role_profile=task["role_profile"]
    )
    logger.info(f"LLM+Agent session: {session_id} | task={task_id}")

    queries = [task["prompt"],
               "请进一步解释一下这个概念与文化认同的关系。",
               "为什么这个文化实践在今天还有意义？",
               "能比较一下传统与现代的理解差异吗？",
               "这对当代学校教育有什么启示？"
               ][:max_turns]

    for i, query in enumerate(queries, 1):
        print(f"\n[Turn {i}] Q: {query}")
        result = engine.process_turn(query)
        print(f"[Turn {i}] A: {result['response'][:400]}")
        print(f"   Actions: {result['actions']} | "
              f"Intent: {result['state']['intent']} | "
              f"Confusion: {result['state']['confusion']}")

    session = engine.end_session()
    engine.save_session_log(session, "logs/raw_dialogues/")
    engine.close()
    print(f"\nLLM+Agent session done. Turns: {session.total_turns}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", default="T001")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_session(args.task_id, args.turns, args.config)


if __name__ == "__main__":
    main()
