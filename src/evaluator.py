"""
CEIEF Evaluator Module
文化教育交互增强框架 - 自动评估器

实现三个核心评估维度的自动化指标计算：
  1. 文化语义连贯性（Cultural Semantic Coherence）
  2. KG 覆盖率（KG Coverage Ratio）
  3. 响应深度分数（Response Depth Score）
  4. 动作多样性指数（Action Diversity Index）
"""

import re
import math
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path

from loguru import logger


@dataclass
class TurnMetrics:
    """单轮评估指标"""
    turn_id: int
    cultural_semantic_coherence: float
    kg_coverage_ratio: float
    response_depth_score: float
    action_diversity_index: float
    response_length: int
    entity_overlap: int
    kg_evidence_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "cultural_semantic_coherence": round(self.cultural_semantic_coherence, 4),
            "kg_coverage_ratio": round(self.kg_coverage_ratio, 4),
            "response_depth_score": round(self.response_depth_score, 4),
            "action_diversity_index": round(self.action_diversity_index, 4),
            "response_length": self.response_length,
            "entity_overlap": self.entity_overlap,
            "kg_evidence_count": self.kg_evidence_count
        }


@dataclass
class SessionMetrics:
    """会话级别评估指标"""
    session_id: str
    condition: str
    task_family: str
    avg_cultural_semantic_coherence: float
    avg_kg_coverage_ratio: float
    avg_response_depth_score: float
    avg_action_diversity_index: float
    total_turns: int
    turn_metrics: List[TurnMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "condition": self.condition,
            "task_family": self.task_family,
            "avg_cultural_semantic_coherence": round(self.avg_cultural_semantic_coherence, 4),
            "avg_kg_coverage_ratio": round(self.avg_kg_coverage_ratio, 4),
            "avg_response_depth_score": round(self.avg_response_depth_score, 4),
            "avg_action_diversity_index": round(self.avg_action_diversity_index, 4),
            "total_turns": self.total_turns,
            "turn_metrics": [t.to_dict() for t in self.turn_metrics]
        }


class AutoEvaluator:
    """
    CEIEF 自动评估器
    对对话会话日志进行自动化指标计算。
    """

    CULTURAL_KEYWORDS = [
        "端午节", "春节", "中秋节", "清明节", "重阳节", "七夕", "元宵节",
        "节庆", "节日", "仪式", "习俗",
        "孝道", "礼仪", "和谐", "仁义", "集体", "社区", "认同", "尊重",
        "价值观", "道德", "伦理",
        "屈原", "孔子", "孟子",
        "集体记忆", "文化认同", "社区认同", "文化传承", "非物质文化遗产",
        "传统", "文化", "历史", "语境", "象征", "意义",
        "学习", "理解", "思考", "反思", "比较", "分析"
    ]

    DEPTH_INDICATORS = [
        "因此", "由此", "可见", "这说明", "这意味着", "从这个角度",
        "值得注意的是", "更深层次", "本质上", "从历史角度", "文化背景下",
        "相比之下", "另一方面", "不仅如此", "此外",
        "therefore", "thus", "this suggests", "fundamentally",
        "from a cultural perspective", "historically"
    ]

    def __init__(self):
        logger.info("AutoEvaluator initialized")

    def compute_cultural_semantic_coherence(
        self,
        response: str,
        learner_query: str,
        kg_evidence: List[str]
    ) -> float:
        """
        计算文化语义连贯性分数
        基于：(1) 文化关键词密度 (2) 主题重叠 (3) KG证据引用
        """
        if not response:
            return 0.0

        score = 0.0
        char_len = len(response)

        # (1) 文化关键词密度（权重 0.4）
        kw_count = sum(1 for kw in self.CULTURAL_KEYWORDS if kw in response)
        kw_density = min(kw_count / max(char_len / 100, 1), 1.0)
        score += kw_density * 0.4

        # (2) 与查询的主题重叠（权重 0.3）
        query_words = set(self._tokenize(learner_query))
        response_words = set(self._tokenize(response))
        if query_words:
            overlap = len(query_words & response_words) / len(query_words)
            score += min(overlap, 1.0) * 0.3

        # (3) KG证据引用（权重 0.3）
        if kg_evidence:
            evidence_hit = 0
            for ev in kg_evidence:
                entities = re.findall(r'[（(]?([^,，()（）]+)', ev)
                for ent in entities:
                    ent = ent.strip()
                    if len(ent) >= 2 and ent in response:
                        evidence_hit += 1
                        break
            coverage = evidence_hit / len(kg_evidence)
            score += coverage * 0.3

        return min(score, 1.0)

    def compute_kg_coverage_ratio(
        self,
        response: str,
        kg_evidence: List[str]
    ) -> float:
        """计算 KG 覆盖率"""
        if not kg_evidence:
            return 0.0

        hit_count = 0
        for ev in kg_evidence:
            # 提取三元组中的实体名
            parts = ev.strip("()").split(",")
            for part in parts:
                part = part.strip()
                if len(part) >= 2 and part in response:
                    hit_count += 1
                    break

        return min(hit_count / len(kg_evidence), 1.0)

    def compute_response_depth_score(self, response: str) -> float:
        """计算响应深度分数"""
        if not response:
            return 0.0

        score = 0.0
        char_len = len(response)

        # (1) 响应长度分数（权重 0.3）
        if char_len < 50:
            length_score = 0.1
        elif char_len < 200:
            length_score = 0.4
        elif char_len <= 800:
            length_score = 1.0
        elif char_len <= 1200:
            length_score = 0.8
        else:
            length_score = 0.6
        score += length_score * 0.3

        # (2) 深度词汇密度（权重 0.4）
        depth_count = sum(1 for d in self.DEPTH_INDICATORS if d in response)
        depth_density = min(depth_count / 3, 1.0)
        score += depth_density * 0.4

        # (3) 句子数量（权重 0.3）
        sentences = re.split(r'[。！？.!?]', response)
        sentences = [s for s in sentences if len(s.strip()) > 5]
        sent_score = min(len(sentences) / 5, 1.0)
        score += sent_score * 0.3

        return min(score, 1.0)

    def compute_action_diversity_index(self, action_history: List[str]) -> float:
        """计算动作多样性指数（基于信息熵）"""
        if not action_history:
            return 0.0

        action_count = Counter(action_history)
        total = len(action_history)
        unique = len(action_count)

        if total == 1:
            return 0.0

        entropy = 0.0
        for count in action_count.values():
            p = count / total
            entropy -= p * math.log2(p)

        max_entropy = math.log2(unique) if unique > 1 else 1.0
        return min(entropy / max_entropy if max_entropy > 0 else 0.0, 1.0)

    def evaluate_turn(
        self,
        turn_id: int,
        response: str,
        learner_query: str,
        kg_evidence: List[str],
        action_history: List[str]
    ) -> TurnMetrics:
        """评估单轮对话"""
        csc = self.compute_cultural_semantic_coherence(response, learner_query, kg_evidence)
        kgcr = self.compute_kg_coverage_ratio(response, kg_evidence)
        rds = self.compute_response_depth_score(response)
        adi = self.compute_action_diversity_index(action_history)

        query_words = set(self._tokenize(learner_query))
        resp_words = set(self._tokenize(response))
        entity_overlap = len(query_words & resp_words)

        return TurnMetrics(
            turn_id=turn_id,
            cultural_semantic_coherence=csc,
            kg_coverage_ratio=kgcr,
            response_depth_score=rds,
            action_diversity_index=adi,
            response_length=len(response),
            entity_overlap=entity_overlap,
            kg_evidence_count=len(kg_evidence)
        )

    def evaluate_session(self, session_log: Dict[str, Any]) -> SessionMetrics:
        """评估整个会话日志"""
        session_id = session_log.get("session_id", "unknown")
        condition = session_log.get("condition", "unknown")
        task_family = session_log.get("task_family", "unknown")
        turns = session_log.get("turns", [])

        turn_metrics_list = []
        action_history: List[str] = []

        for turn in turns:
            turn_id = turn.get("turn_id", 0)
            response = turn.get("llm_response", "")
            query = turn.get("learner_query", "")
            kg_evidence = turn.get("kg_evidence", [])
            actions = turn.get("selected_actions", [])
            action_history.extend(actions)

            tm = self.evaluate_turn(
                turn_id=turn_id,
                response=response,
                learner_query=query,
                kg_evidence=kg_evidence,
                action_history=action_history[:]
            )
            turn_metrics_list.append(tm)

        def avg(lst: List[TurnMetrics], key: str) -> float:
            vals = [getattr(m, key) for m in lst]
            return sum(vals) / len(vals) if vals else 0.0

        return SessionMetrics(
            session_id=session_id,
            condition=condition,
            task_family=task_family,
            avg_cultural_semantic_coherence=avg(turn_metrics_list, "cultural_semantic_coherence"),
            avg_kg_coverage_ratio=avg(turn_metrics_list, "kg_coverage_ratio"),
            avg_response_depth_score=avg(turn_metrics_list, "response_depth_score"),
            avg_action_diversity_index=avg(turn_metrics_list, "action_diversity_index"),
            total_turns=len(turn_metrics_list),
            turn_metrics=turn_metrics_list
        )

    def evaluate_from_file(self, log_path: str) -> SessionMetrics:
        """从 JSON 日志文件读取并评估"""
        with open(log_path, "r", encoding="utf-8") as f:
            session_log = json.load(f)
        return self.evaluate_session(session_log)

    def batch_evaluate(
        self, log_dir: str
    ) -> List[SessionMetrics]:
        """批量评估目录中的所有日志文件"""
        results = []
        for log_file in Path(log_dir).glob("*.json"):
            try:
                sm = self.evaluate_from_file(str(log_file))
                results.append(sm)
                logger.info(f"Evaluated: {log_file.name} | CSC={sm.avg_cultural_semantic_coherence:.3f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {log_file}: {e}")
        return results

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（中英文混合）"""
        tokens = re.split(r'[\s，。！？、；：""''《》【】()（）]+', text)
        return [t for t in tokens if len(t) >= 2]

    def save_metrics(
        self,
        metrics_list: List[SessionMetrics],
        output_path: str = "logs/scoring_outputs/auto_metrics.json"
    ):
        """将评估结果保存为 JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in metrics_list]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to {output_path}")
