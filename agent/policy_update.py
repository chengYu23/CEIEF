"""
CEIEF Agent Policy Update Module
文化教育交互增强框架 - Agent 策略更新模块

支持基于会话日志的规则权重调整（离线分析用）。
在当前版本中实现规则权重统计分析，为后续ML策略提供基础。
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from collections import Counter, defaultdict

from loguru import logger


class PolicyAnalyzer:
    """
    策略分析器：分析动作选择历史，计算动作效能指标，
    为规则权重调整提供数据依据。
    """

    def __init__(self, log_dir: str = "logs/controller_actions/"):
        self.log_dir = Path(log_dir)
        self.action_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "count": 0,
            "avg_depth_at_use": 0.0,
            "confusion_distribution": {"low": 0, "medium": 0, "high": 0},
            "intent_distribution": {}
        })

    def load_action_logs(self) -> List[Dict[str, Any]]:
        """加载所有动作日志文件"""
        logs = []
        for log_file in self.log_dir.glob("*.json"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logs.extend(data)
                    else:
                        logs.append(data)
            except Exception as e:
                logger.warning(f"Failed to load log {log_file}: {e}")
        return logs

    def analyze_action_distribution(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析动作使用分布"""
        action_counts = Counter()
        condition_action_map = defaultdict(Counter)

        for log in logs:
            action = log.get("action_name", "unknown")
            action_counts[action] += 1

            # 按条件分析
            state = log.get("metadata", {})
            confusion = state.get("confusion_level", "unknown")
            condition_action_map[confusion][action] += 1

        total = sum(action_counts.values())
        distribution = {
            action: {
                "count": count,
                "proportion": count / total if total > 0 else 0
            }
            for action, count in action_counts.most_common()
        }

        return {
            "total_actions": total,
            "action_distribution": distribution,
            "condition_action_map": {
                k: dict(v) for k, v in condition_action_map.items()
            }
        }

    def suggest_weight_updates(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        基于动作分布分析，建议权重更新方向。
        当某动作使用过度（>40%）时建议降低权重；
        当某动作使用不足（<5%）时建议提高权重。
        """
        suggestions = {}
        dist = analysis.get("action_distribution", {})
        total = analysis.get("total_actions", 1)

        for action, stats in dist.items():
            proportion = stats["proportion"]
            if proportion > 0.4:
                suggestions[action] = -0.05  # 降低权重
                logger.info(f"Action '{action}' overused ({proportion:.1%}), suggest weight -0.05")
            elif proportion < 0.05:
                suggestions[action] = +0.05  # 提高权重
                logger.info(f"Action '{action}' underused ({proportion:.1%}), suggest weight +0.05")
            else:
                suggestions[action] = 0.0  # 保持不变

        return suggestions

    def generate_policy_report(
        self, output_path: str = "logs/policy_analysis_report.json"
    ) -> Dict[str, Any]:
        """生成完整策略分析报告"""
        logs = self.load_action_logs()
        if not logs:
            logger.warning("No action logs found for analysis")
            return {"status": "no_data"}

        analysis = self.analyze_action_distribution(logs)
        suggestions = self.suggest_weight_updates(analysis)

        report = {
            "analysis": analysis,
            "weight_update_suggestions": suggestions,
            "total_logs_analyzed": len(logs)
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Policy report saved to {output_path}")
        return report


if __name__ == "__main__":
    analyzer = PolicyAnalyzer()
    report = analyzer.generate_policy_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
