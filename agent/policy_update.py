"""
CEIEF Agent Policy Update Module
文化教育交互增强框架 - Agent 策略更新模块

包含两个核心组件：
1. PolicyAnalyzer          — 基于会话日志的规则权重统计分析（离线分析用）
2. REINFORCEPolicyUpdater  — 基于 episode 级 REINFORCE 的在线策略梯度更新
"""

import json
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from loguru import logger


# ============================================================
# 全局常量
# ============================================================

# 奖励权重（对应论文 reward_design 配置；合计 = 1.0）
REWARD_WEIGHTS: Dict[str, float] = {
    "semantic_grounding":          0.35,
    "contextual_coherence":        0.25,
    "pedagogical_appropriateness": 0.25,
    "role_style_fidelity":         0.15,
}

# 7 个控制器动作（对应 softmax 策略的输出维度）
ACTION_LABELS: List[str] = [
    "evidence_expansion",
    "role_anchoring",
    "perspective_contrast",
    "reflective_prompting",
    "clarification",
    "simplification",
    "summary_closure",
]

# 15 维状态特征名称（对应 agent/state_feature_schema.json）
STATE_FEATURE_NAMES: List[str] = [
    "confusion_level_low",
    "confusion_level_medium",
    "confusion_level_high",
    "intent_explanatory_question",
    "intent_comparison_request",
    "intent_reflection",
    "intent_transfer_request",
    "intent_clarification_request",
    "dialogue_depth_norm",           # 归一化轮次计数
    "kg_evidence_count_norm",        # 归一化检索三元组数
    "last_action_diversity",         # 窗口内不同动作占比
    "response_depth_score",
    "cultural_semantic_coherence",
    "kg_coverage_ratio",
    "role_consistency",
]
assert len(STATE_FEATURE_NAMES) == 15, "State vector must be exactly 15-dimensional"


# ============================================================
# PolicyAnalyzer
# ============================================================

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

            state = log.get("metadata", {})
            confusion = state.get("confusion_level", "unknown")
            condition_action_map[confusion][action] += 1

        total = sum(action_counts.values())
        distribution = {
            action: {
                "count": count,
                "proportion": count / total if total > 0 else 0.0
            }
            for action, count in action_counts.items()
        }

        return {
            "total_actions": total,
            "action_distribution": distribution,
            "confusion_breakdown": {
                level: dict(counts)
                for level, counts in condition_action_map.items()
            }
        }

    def suggest_weight_updates(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """根据动作使用频率建议权重调整量"""
        distribution = analysis.get("action_distribution", {})
        total_actions = len(ACTION_LABELS)
        target_proportion = 1.0 / total_actions if total_actions > 0 else 0.0

        suggestions: Dict[str, float] = {}
        for action in ACTION_LABELS:
            info = distribution.get(action, {"proportion": 0.0})
            proportion = info.get("proportion", 0.0)

            if proportion > target_proportion * 2:
                suggestions[action] = -0.05
                logger.info(f"Action '{action}' overused ({proportion:.1%}), suggest weight -0.05")
            elif proportion < 0.05:
                suggestions[action] = +0.05
                logger.info(f"Action '{action}' underused ({proportion:.1%}), suggest weight +0.05")
            else:
                suggestions[action] = 0.0

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


# ============================================================
# REINFORCEPolicyUpdater
# ============================================================

class REINFORCEPolicyUpdater:
    """
    基于 episode 级 REINFORCE 的策略梯度更新器。

    规格（论文 policy_module 配置节）：
      - 状态维度  : state_dim = 15
      - 动作空间  : n_actions = 7（softmax 分布）
      - 更新规则  : episode-level REINFORCE（Monte-Carlo 回报，无自举）
      - 学习率    : lr = 1e-3
      - 方差缩减  : 运行均值基线（running_mean baseline）

    参数化：
      θ ∈ R^{state_dim × n_actions}（线性 softmax 策略）
      π(a|s) = softmax(s @ θ)[a]

    更新公式（每 episode 结束后执行一次）：
      ∇J(θ) ≈ Σ_t (G_t − b) · ∇ log π(a_t|s_t)
      θ ← θ + lr · ∇J(θ)
    其中 G_t 为折现回报，b 为运行均值基线。
    """

    STATE_DIM: int   = 15
    N_ACTIONS: int   = 7
    LR:        float = 1e-3
    GAMMA:     float = 0.99

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "logs/policy_checkpoints/",
    ):
        """
        初始化策略参数。

        Args:
            config: 来自 agent/controller_config.yaml 的 ml_policy 节（可选）。
                    若为 None 则使用论文默认值。
            checkpoint_dir: 策略参数检查点保存目录。
        """
        cfg = config or {}
        self.state_dim:         int   = int(cfg.get("state_dim",       self.STATE_DIM))
        self.n_actions:         int   = int(cfg.get("n_actions",       self.N_ACTIONS))
        self.lr:                float = float(cfg.get("learning_rate",  self.LR))
        self.gamma:             float = float(cfg.get("discount_factor", self.GAMMA))
        self.baseline:          str   = str(cfg.get("baseline",        "running_mean"))
        self.normalize_returns: bool  = bool(cfg.get("normalize_returns", True))
        self.checkpoint_dir           = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 线性策略参数矩阵  θ ∈ R^{state_dim × n_actions}
        self.theta: np.ndarray = np.zeros(
            (self.state_dim, self.n_actions), dtype=np.float64
        )

        # 运行均值基线
        self._baseline_value: float = 0.0
        self._baseline_count: int   = 0

        # episode 缓冲区：[(state, action_idx, reward), ...]
        self._episode_buffer: List[Tuple[np.ndarray, int, float]] = []

        logger.info(
            f"REINFORCEPolicyUpdater initialised | "
            f"state_dim={self.state_dim}, n_actions={self.n_actions}, "
            f"lr={self.lr}, gamma={self.gamma}, baseline={self.baseline}"
        )

    # ----------------------------------------------------------
    # 奖励计算
    # ----------------------------------------------------------

    @staticmethod
    def compute_composite_reward(
        semantic_grounding:          float,
        contextual_coherence:        float,
        pedagogical_appropriateness: float,
        role_style_fidelity:         float,
    ) -> float:
        """
        计算加权复合奖励（对应论文 reward_design 配置）。

        权重：
          semantic_grounding          = 0.35
          contextual_coherence        = 0.25
          pedagogical_appropriateness = 0.25
          role_style_fidelity         = 0.15

        Args:
            semantic_grounding:          语义接地度分数 ∈ [0, 1]
            contextual_coherence:        上下文连贯性分数 ∈ [0, 1]
            pedagogical_appropriateness: 教学适切性分数 ∈ [0, 1]
            role_style_fidelity:         角色/风格忠实度分数 ∈ [0, 1]

        Returns:
            r ∈ [0, 1]：加权复合奖励
        """
        w = REWARD_WEIGHTS
        return (
            w["semantic_grounding"]          * semantic_grounding
            + w["contextual_coherence"]      * contextual_coherence
            + w["pedagogical_appropriateness"] * pedagogical_appropriateness
            + w["role_style_fidelity"]       * role_style_fidelity
        )

    # ----------------------------------------------------------
    # 策略推断
    # ----------------------------------------------------------

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """数值稳定 softmax。"""
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        给定 15 维状态向量，返回 7 个动作的 softmax 概率分布。

        Args:
            state: shape (15,) 的状态特征向量。

        Returns:
            probs: shape (7,) 的动作概率分布。
        """
        assert state.shape == (self.state_dim,), (
            f"Expected state shape ({self.state_dim},), got {state.shape}"
        )
        logits = state @ self.theta   # (7,)
        return self._softmax(logits)

    def select_action(
        self, state: np.ndarray, greedy: bool = False
    ) -> Tuple[int, str, float]:
        """
        依据当前策略选择动作。

        Args:
            state:  15 维状态向量。
            greedy: True → argmax（评估模式）；False → 采样（训练模式）。

        Returns:
            (action_idx, action_label, action_prob)
        """
        probs = self.get_action_probs(state)
        if greedy:
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(self.n_actions, p=probs))
        return idx, ACTION_LABELS[idx], float(probs[idx])

    # ----------------------------------------------------------
    # Episode 缓冲区管理
    # ----------------------------------------------------------

    def record_step(
        self, state: np.ndarray, action_idx: int, reward: float
    ) -> None:
        """
        记录单步 (s_t, a_t, r_t) 到 episode 缓冲区。

        Args:
            state:      15 维状态向量。
            action_idx: 执行的动作索引（0–6）。
            reward:     该步获得的即时奖励。
        """
        self._episode_buffer.append((state.copy(), action_idx, reward))

    # ----------------------------------------------------------
    # Episode 级更新（REINFORCE）
    # ----------------------------------------------------------

    def update_episode(self) -> Dict[str, Any]:
        """
        在 episode 结束后执行 REINFORCE 策略梯度更新。

        算法步骤：
          1. 计算每步折现回报  G_t = Σ_{k≥t} γ^{k-t} r_k
          2. 用运行均值基线 b 中心化回报（方差缩减）
          3. 可选：对回报序列做 z-score 归一化
          4. 对每步计算策略梯度并累加
          5. 用梯度上升更新 θ
          6. 清空 episode 缓冲区

        Returns:
            update_info: 包含 total_return、policy_loss、episode_length 的字典。
        """
        if not self._episode_buffer:
            logger.warning("update_episode called with empty buffer — skipped")
            return {"status": "empty_buffer"}

        T = len(self._episode_buffer)
        rewards = np.array([r for _, _, r in self._episode_buffer], dtype=np.float64)

        # 1. 折现回报
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        total_return = float(returns[0])

        # 2. 基线中心化
        if self.baseline == "running_mean":
            self._baseline_count += 1
            self._baseline_value += (
                total_return - self._baseline_value
            ) / self._baseline_count
            b = self._baseline_value
        else:
            b = 0.0
        advantages = returns - b

        # 3. 可选归一化
        if self.normalize_returns and T > 1:
            std = advantages.std() + 1e-8
            advantages = advantages / std

        # 4. 累计策略梯度
        grad_theta = np.zeros_like(self.theta)
        policy_loss = 0.0
        for t, (state, action_idx, _) in enumerate(self._episode_buffer):
            probs   = self.get_action_probs(state)       # (7,)
            adv_t   = advantages[t]
            one_hot = np.zeros(self.n_actions, dtype=np.float64)
            one_hot[action_idx] = 1.0
            # ∇_θ log π(a_t|s_t) = s_t ⊗ (e_{a_t} − π(·|s_t))
            score       = np.outer(state, one_hot - probs)  # (15, 7)
            grad_theta  += adv_t * score
            policy_loss -= adv_t * math.log(probs[action_idx] + 1e-10)

        # 5. 梯度上升
        self.theta += self.lr * grad_theta

        # 6. 清空缓冲区
        self._episode_buffer.clear()

        update_info = {
            "total_return":   total_return,
            "policy_loss":    float(policy_loss),
            "episode_length": T,
            "baseline_value": float(b),
        }
        logger.debug(f"REINFORCE update | {update_info}")
        return update_info

    # ----------------------------------------------------------
    # 检查点 I/O
    # ----------------------------------------------------------

    def save_checkpoint(self, name: str = "policy_theta.npy") -> str:
        """将策略参数 θ 保存为 .npy 文件。"""
        path = self.checkpoint_dir / name
        np.save(str(path), self.theta)
        meta = {
            "state_dim":      self.state_dim,
            "n_actions":      self.n_actions,
            "lr":             self.lr,
            "gamma":          self.gamma,
            "baseline":       self.baseline,
            "action_labels":  ACTION_LABELS,
            "state_features": STATE_FEATURE_NAMES,
        }
        meta_path = self.checkpoint_dir / name.replace(".npy", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Policy checkpoint saved → {path}")
        return str(path)

    def load_checkpoint(self, name: str = "policy_theta.npy") -> None:
        """从 .npy 文件加载策略参数 θ。"""
        path = self.checkpoint_dir / name
        self.theta = np.load(str(path))
        logger.info(f"Policy checkpoint loaded ← {path} | θ.shape={self.theta.shape}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # --- PolicyAnalyzer demo ---
    analyzer = PolicyAnalyzer()
    report = analyzer.generate_policy_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # --- REINFORCEPolicyUpdater demo ---
    print("\n=== REINFORCEPolicyUpdater demo ===")
    updater = REINFORCEPolicyUpdater()
    np_rng = np.random.default_rng(42)

    # 模拟一个 episode（5 步）
    for step in range(5):
        state = np_rng.random(15).astype(np.float64)
        idx, label, prob = updater.select_action(state)
        reward = REINFORCEPolicyUpdater.compute_composite_reward(
            semantic_grounding=np_rng.random(),
            contextual_coherence=np_rng.random(),
            pedagogical_appropriateness=np_rng.random(),
            role_style_fidelity=np_rng.random(),
        )
        updater.record_step(state, idx, reward)
        print(f"  step {step+1}: action={label}, reward={reward:.4f}")

    info = updater.update_episode()
    print(f"\nEpisode update: {info}")
