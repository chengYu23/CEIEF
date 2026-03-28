"""
CEIEF Agent Controller Module
文化教育交互增强框架 - Agent 控制层

基于当前交互状态选择最优教学动作，
实现自适应交互调节（Adaptive Interaction Regulation）。
动作空间：evidence_expansion, role_anchoring, perspective_contrast,
          reflective_prompting, clarification, simplification, summary_closure
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.state_tracker import (
    InteractionState, LearnerIntentType, ConfusionLevel, BehaviorFeatures
)


@dataclass
class ActionResult:
    """Agent 动作选择结果"""
    action_name: str
    prompt_instruction: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentController:
    """
    CEIEF Agent 控制器
    根据当前状态 s_t 选择最优教学动作 a_t，
    并返回对应的提示词指令片段。

    策略：基于规则的确定性策略（rule_based）
    可扩展为 ML-based 或 hybrid 策略。
    """

    # 动作到提示词指令的映射（来自实现方案第8章）
    ACTION_PROMPT_MAP: Dict[str, str] = {
        "evidence_expansion": (
            "Use at least two specific cultural facts or historical references "
            "from the retrieved knowledge graph evidence. Expand the explanation "
            "by linking the evidence to broader cultural patterns. "
            "Show the chain of cultural reasoning explicitly."
        ),
        "role_anchoring": (
            "Re-establish your role perspective clearly before answering. "
            "Remind the learner of your cultural position and the historical "
            "context you are speaking from. Ensure your response reflects "
            "the specific cultural lens of your assigned role."
        ),
        "perspective_contrast": (
            "Present at least two contrasting cultural perspectives or "
            "interpretations of this topic. Use comparative language "
            "(e.g., 'In contrast...', 'While some traditions...', 'However...'). "
            "Acknowledge tensions or value differences respectfully."
        ),
        "reflective_prompting": (
            "After your main response, include one open-ended reflective question "
            "that invites the learner to think more deeply about the cultural "
            "implications. The question should be grounded in the retrieved evidence "
            "and culturally grounded."
        ),
        "clarification": (
            "Simplify your explanation by breaking it into 2-3 clear points. "
            "Define any technical cultural terms before using them. "
            "Use concrete, relatable examples to anchor abstract concepts. "
            "Check for understanding at the end."
        ),
        "simplification": (
            "Provide a concise summary (3-4 sentences) focusing on the most "
            "essential cultural insight. Avoid complex terminology. "
            "Use plain language while preserving cultural accuracy."
        ),
        "summary_closure": (
            "Provide a brief summary of the key cultural insights discussed "
            "so far in this dialogue. Highlight the main cultural concepts "
            "encountered and the connections made. Invite the learner to "
            "reflect on what they have learned."
        )
    }

    # 状态到优先动作的规则映射
    RULE_TABLE: List[Dict[str, Any]] = [
        # 规则格式: {conditions, actions, priority, description}
        {
            "description": "高困惑 + 澄清意图 -> 澄清",
            "conditions": {
                "confusion_level": ConfusionLevel.HIGH,
                "intent_type": LearnerIntentType.CLARIFICATION_REQUEST
            },
            "actions": ["clarification", "simplification"],
            "priority": 10
        },
        {
            "description": "高困惑 -> 简化",
            "conditions": {
                "confusion_level": ConfusionLevel.HIGH
            },
            "actions": ["simplification", "clarification"],
            "priority": 9
        },
        {
            "description": "解释性问题 -> 证据扩展",
            "conditions": {
                "intent_type": LearnerIntentType.EXPLANATORY_QUESTION
            },
            "actions": ["evidence_expansion", "role_anchoring"],
            "priority": 8
        },
        {
            "description": "比较请求 -> 视角对比",
            "conditions": {
                "intent_type": LearnerIntentType.COMPARISON_REQUEST
            },
            "actions": ["perspective_contrast", "evidence_expansion"],
            "priority": 8
        },
        {
            "description": "反思 -> 反思性追问",
            "conditions": {
                "intent_type": LearnerIntentType.REFLECTION
            },
            "actions": ["reflective_prompting", "perspective_contrast"],
            "priority": 7
        },
        {
            "description": "迁移请求 -> 视角对比 + 反思",
            "conditions": {
                "intent_type": LearnerIntentType.TRANSFER_REQUEST
            },
            "actions": ["perspective_contrast", "reflective_prompting"],
            "priority": 7
        },
        {
            "description": "对话深度>=10 -> 阶段性总结",
            "conditions": {
                "min_dialogue_depth": 10
            },
            "actions": ["summary_closure"],
            "priority": 6
        },
        {
            "description": "事实性问题 -> 证据扩展",
            "conditions": {
                "intent_type": LearnerIntentType.FACTUAL_QUESTION
            },
            "actions": ["evidence_expansion"],
            "priority": 5
        },
        {
            "description": "默认动作",
            "conditions": {},
            "actions": ["evidence_expansion"],
            "priority": 1
        }
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Agent 控制器

        Args:
            config: 来自 config.yaml 的完整配置
        """
        self.agent_config = config.get("agent", {})
        self.policy = self.agent_config.get("policy", "rule_based")
        self.exploration_rate = self.agent_config.get("exploration_rate", 0.1)
        self.action_space = self.agent_config.get("actions", list(self.ACTION_PROMPT_MAP.keys()))

        # 加载外部动作空间配置（如存在）
        action_space_path = config.get("paths", {}).get("action_space", "agent/action_space.json")
        self._load_action_space(action_space_path)

        # 动作历史（用于多样性保证）
        self._action_history: List[str] = []

        logger.info(f"AgentController initialized | policy={self.policy}")

    def _load_action_space(self, path: str):
        """从 JSON 文件加载动作空间定义（可选）"""
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 合并外部提示词定义
                if "action_prompt_map" in data:
                    self.ACTION_PROMPT_MAP.update(data["action_prompt_map"])
                logger.info(f"Action space loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load action space from {path}: {e}")

    def select_action(self, state: InteractionState) -> ActionResult:
        """
        根据当前状态选择最优动作

        Args:
            state: 当前交互状态 s_t

        Returns:
            ActionResult 包含动作名称和提示词指令
        """
        if self.policy == "rule_based":
            return self._rule_based_selection(state)
        elif self.policy == "stochastic":
            return self._stochastic_selection(state)
        else:
            return self._rule_based_selection(state)

    def _rule_based_selection(
        self, state: InteractionState
    ) -> ActionResult:
        """基于规则的确定性动作选择"""
        # 按优先级排序规则
        sorted_rules = sorted(
            self.RULE_TABLE, key=lambda r: r["priority"], reverse=True
        )

        matched_actions = None
        matched_rule = None

        for rule in sorted_rules:
            if self._check_conditions(rule["conditions"], state):
                matched_actions = rule["actions"]
                matched_rule = rule["description"]
                break

        if not matched_actions:
            matched_actions = ["evidence_expansion"]
            matched_rule = "default"

        # 选择动作（考虑多样性：避免连续重复）
        selected = self._select_with_diversity(matched_actions)

        self._action_history.append(selected)
        if len(self._action_history) > 20:
            self._action_history = self._action_history[-20:]

        prompt_instruction = self.ACTION_PROMPT_MAP.get(
            selected,
            "Provide a culturally grounded and pedagogically appropriate response."
        )

        return ActionResult(
            action_name=selected,
            prompt_instruction=prompt_instruction,
            confidence=0.85,
            reasoning=matched_rule,
            metadata={
                "rule_matched": matched_rule,
                "candidates": matched_actions,
                "dialogue_depth": state.behavior.dialogue_depth
            }
        )

    def _stochastic_selection(
        self, state: InteractionState
    ) -> ActionResult:
        """随机探索动作选择（用于实验对比）"""
        # epsilon-greedy: 以 exploration_rate 概率随机选择
        if random.random() < self.exploration_rate:
            selected = random.choice(self.action_space)
            reasoning = "random_exploration"
        else:
            result = self._rule_based_selection(state)
            return result

        prompt_instruction = self.ACTION_PROMPT_MAP.get(
            selected,
            "Provide a culturally grounded response."
        )
        return ActionResult(
            action_name=selected,
            prompt_instruction=prompt_instruction,
            confidence=0.5,
            reasoning=reasoning
        )

    def _check_conditions(
        self, conditions: Dict[str, Any], state: InteractionState
    ) -> bool:
        """检查规则条件是否满足"""
        for key, value in conditions.items():
            if key == "confusion_level":
                if state.confusion_level != value:
                    return False
            elif key == "intent_type":
                if state.intent_type != value:
                    return False
            elif key == "min_dialogue_depth":
                if state.behavior.dialogue_depth < value:
                    return False
            elif key == "min_clarification_count":
                if state.behavior.clarification_count < value:
                    return False
        return True

    def _select_with_diversity(
        self, candidates: List[str]
    ) -> str:
        """从候选动作中选择，避免连续重复"""
        recent = self._action_history[-2:] if len(self._action_history) >= 2 else []
        for action in candidates:
            if action not in recent:
                return action
        # 如果全部都最近用过，则返回第一个候选
        return candidates[0]

    def get_multi_actions(
        self, state: InteractionState, n: int = 2
    ) -> List[ActionResult]:
        """
        选择多个动作（用于 Full CEIEF 条件下的复合动作）

        Args:
            state: 当前状态
            n: 最多选择的动作数量

        Returns:
            动作结果列表
        """
        primary = self.select_action(state)
        results = [primary]

        if n > 1:
            # 尝试选择一个互补动作
            complement_map = {
                "evidence_expansion": "reflective_prompting",
                "clarification": "evidence_expansion",
                "perspective_contrast": "reflective_prompting",
                "reflective_prompting": "perspective_contrast",
                "simplification": "clarification",
                "role_anchoring": "evidence_expansion",
                "summary_closure": "reflective_prompting"
            }
            complement_name = complement_map.get(primary.action_name, "reflective_prompting")
            if complement_name != primary.action_name:
                complement_instruction = self.ACTION_PROMPT_MAP.get(
                    complement_name, ""
                )
                results.append(ActionResult(
                    action_name=complement_name,
                    prompt_instruction=complement_instruction,
                    confidence=0.7,
                    reasoning="complement_action"
                ))

        return results

    def combine_action_instructions(
        self, actions: List[ActionResult]
    ) -> str:
        """将多个动作指令合并为单一指令文本"""
        if len(actions) == 1:
            return actions[0].prompt_instruction
        instructions = []
        for i, action in enumerate(actions, 1):
            instructions.append(f"{i}. [{action.action_name}] {action.prompt_instruction}")
        return "\n".join(instructions)

    def log_action(
        self, action: ActionResult, session_id: str, turn: int
    ) -> Dict[str, Any]:
        """生成动作日志记录"""
        return {
            "session_id": session_id,
            "turn": turn,
            "action_name": action.action_name,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "metadata": action.metadata
        }
