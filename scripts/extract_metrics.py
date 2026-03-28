"""
CEIEF Metrics Extraction Script
从对话日志中提取自动评估指标

用法：python scripts/extract_metrics.py --log_dir logs/raw_dialogues/ --output logs/scoring_outputs/auto_metrics.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.evaluator import AutoEvaluator


def main():
    parser = argparse.ArgumentParser(description="Extract automatic metrics from dialogue logs")
    parser.add_argument("--log_dir", type=str, default="logs/raw_dialogues/",
                        help="Directory containing dialogue JSON logs")
    parser.add_argument("--output", type=str, default="logs/scoring_outputs/auto_metrics.json",
                        help="Output path for metrics JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-session details")
    args = parser.parse_args()

    evaluator = AutoEvaluator()
    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        return

    log_files = list(log_dir.glob("*.json"))
    if not log_files:
        print(f"No JSON log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files. Evaluating...")
    all_metrics = []

    for log_file in log_files:
        try:
            sm = evaluator.evaluate_from_file(str(log_file))
            all_metrics.append(sm)

            if args.verbose:
                print(f"\n{log_file.name}:")
                print(f"  Condition: {sm.condition}")
                print(f"  Turns: {sm.total_turns}")
                print(f"  CSC: {sm.avg_cultural_semantic_coherence:.3f}")
                print(f"  KG Coverage: {sm.avg_kg_coverage_ratio:.3f}")
                print(f"  Depth: {sm.avg_response_depth_score:.3f}")
                print(f"  Action Diversity: {sm.avg_action_diversity_index:.3f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {log_file}: {e}")

    # 按条件汇总
    condition_groups: dict = {}
    for sm in all_metrics:
        cond = sm.condition
        if cond not in condition_groups:
            condition_groups[cond] = []
        condition_groups[cond].append(sm)

    print(f"\n{'='*50}")
    print("SUMMARY BY CONDITION")
    print(f"{'='*50}")

    summary = {}
    for cond, sessions in condition_groups.items():
        n = len(sessions)
        avg_csc = sum(s.avg_cultural_semantic_coherence for s in sessions) / n
        avg_kg = sum(s.avg_kg_coverage_ratio for s in sessions) / n
        avg_depth = sum(s.avg_response_depth_score for s in sessions) / n
        avg_adi = sum(s.avg_action_diversity_index for s in sessions) / n

        summary[cond] = {
            "n_sessions": n,
            "avg_cultural_semantic_coherence": round(avg_csc, 4),
            "avg_kg_coverage_ratio": round(avg_kg, 4),
            "avg_response_depth_score": round(avg_depth, 4),
            "avg_action_diversity_index": round(avg_adi, 4)
        }

        print(f"\nCondition: {cond} (n={n})")
        print(f"  Cultural Semantic Coherence: {avg_csc:.3f}")
        print(f"  KG Coverage Ratio: {avg_kg:.3f}")
        print(f"  Response Depth Score: {avg_depth:.3f}")
        print(f"  Action Diversity Index: {avg_adi:.3f}")

    # 保存结果
    evaluator.save_metrics(all_metrics, args.output)

    summary_path = Path(args.output).parent / "condition_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nCondition summary saved to: {summary_path}")
    print(f"Detailed metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
