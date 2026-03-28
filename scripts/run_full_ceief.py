"""
CEIEF Full CEIEF Experiment Runner
运行 Full CEIEF 条件实验（KG + LLM + Agent）

用法：python scripts/run_full_ceief.py --task_id T001 --turns 5
"""

import sys
import json
import argparse
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.dialogue_engine import DialogueEngine, CONDITION_FULL_CEIEF


def load_task(task_id: str, task_bank_path: str = "data/tasks/task_bank.json") -> dict:
    """从任务库加载指定任务"""
    with open(task_bank_path, "r", encoding="utf-8") as f:
        task_bank = json.load(f)
    for task in task_bank["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Task {task_id} not found in task bank")


def run_session(
    task_id: str,
    max_turns: int = 5,
    config_path: str = "config.yaml",
    output_dir: str = "logs/raw_dialogues/"
):
    """运行单个 Full CEIEF 会话"""
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载任务
    task = load_task(task_id)
    logger.info(f"Running Full CEIEF | task={task_id} | family={task['family']}")

    # 初始化对话引擎
    engine = DialogueEngine(config)

    # 开始会话
    session_id = engine.start_session(
        condition=CONDITION_FULL_CEIEF,
        task_family=task["family"],
        role_profile=task["role_profile"]
    )
    logger.info(f"Session started: {session_id}")

    # 第一轮：使用任务提示词
    queries = [task["prompt"]]

    # 模拟后续轮次（实验中由真实学习者输入）
    followup_queries = [
        "请进一步解释一下这个概念与文化认同的关系。",
        "为什么这个文化实践在今天还有意义？",
        "能比较一下传统与现代的理解差异吗？",
        "这对当代学校教育有什么启示？"
    ]
    queries.extend(followup_queries[:max_turns - 1])

    # 执行对话
    for i, query in enumerate(queries[:max_turns], 1):
        print(f"\n{'='*60}")
        print(f"[Turn {i}] Learner: {query}")
        try:
            result = engine.process_turn(query)
            print(f"[Turn {i}] System Response:")
            print(result["response"])
            print(f"\n[Metadata] KG triples: {len(result['kg_evidence'])} | "
                  f"Actions: {result['actions']} | "
                  f"Intent: {result['state']['intent']} | "
                  f"Confusion: {result['state']['confusion']}")
        except Exception as e:
            logger.error(f"Turn {i} failed: {e}")
            break

    # 结束会话
    session = engine.end_session()
    log_path = engine.save_session_log(session, output_dir)
    print(f"\n{'='*60}")
    print(f"Session completed. Log saved to: {log_path}")
    print(f"Total turns: {session.total_turns}")

    engine.close()
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Run Full CEIEF experiment session")
    parser.add_argument("--task_id", type=str, default="T001", help="Task ID from task bank")
    parser.add_argument("--turns", type=int, default=5, help="Maximum number of dialogue turns")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default="logs/raw_dialogues/", help="Output directory")
    args = parser.parse_args()

    run_session(
        task_id=args.task_id,
        max_turns=args.turns,
        config_path=args.config,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
