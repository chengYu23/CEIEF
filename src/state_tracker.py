"""
CEIEF State Tracker Module
文化教育交互增强框架 - 状态追踪器

负责追踪学习者行为特征向量、对话深度、困惑信号、证据请求频次等，
构建 Agent 所需的状态表示 s_t = [g_t; u_t; m_t; b_t]。
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class LearnerIntentType(str, Enum):
    """学习者意图类型"""
    FACTUAL_QUESTION = "factual_question"
    EXPLANATORY_QUESTION = "explanatory_question"
    COMPARISON_REQUEST = "comparison_request"
    CLARIFICATION_REQUEST = "clarification_request"
    REFLECTION = "reflection"
    CHALLENGE = "challenge"
    AGREEMENT = "agreement"
    OFF_TOPIC = "off_topic"
    TRANSFER_REQUEST = "transfer_request"


class ConfusionLevel(str, Enum):
    """学习者困惑程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BehaviorFeatures:
    """行为特征向量 b_t"""
    dialogue_depth: int = 0
    clarification_count: int = 0
    evidence_request_count: int = 0
    perspective_switch_count: int = 0
    rephrase_count: int = 0
    question_count: int = 0
    short_input_count: int = 0
    keyword_repetition_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialogue_depth": self.dialogue_depth,
            "clarification_count": self.clarification_count,
            "evidence_request_count": self.evidence_request_count,
            "perspective_switch_count": self.perspective_switch_count,
            "rephrase_count": self.rephrase_count,
            "question_count": self.question_count,
            "short_input_count": self.short_input_count,
            "keyword_repetition_count": self.keyword_repetition_count
        }


@dataclass
class DialogueMemory:
    """对话记忆 m_t"""
    turns: List[Dict[str, str]] = field(default_factory=list)
    window_size: int = 10
    mentioned_entities: List[str] = field(default_factory=list)
    previous_actions: List[str] = field(default_factory=list)

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.window_size * 2:
            self.turns = self.turns[-self.window_size * 2:]

    def get_recent_turns(self, n: int = 5) -> List[Dict[str, str]]:
        return self.turns[-n * 2:] if len(self.turns) >= n * 2 else self.turns

    def get_context_text(self, n: int = 3) -> str:
        recent = self.get_recent_turns(n)
        lines = []
        for turn in recent:
            role_label = "学习者" if turn["role"] == "user" else "助手"
            lines.append(f"{role_label}: {turn['content'][:200]}")
        return "\n".join(lines)


@dataclass
class InteractionState:
    """完整交互状态 s_t = [g_t; u_t; m_t; b_t]"""
    graph_context: List[str] = field(default_factory=list)
    current_query: str = ""
    intent_type: LearnerIntentType = LearnerIntentType.FACTUAL_QUESTION
    confusion_level: ConfusionLevel = ConfusionLevel.LOW
    detected_entities: List[str] = field(default_factory=list)
    memory: DialogueMemory = field(default_factory=DialogueMemory)
    behavior: BehaviorFeatures = field(default_factory=BehaviorFeatures)
    session_id: str = ""
    task_family: str = ""
    role_profile: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_context": self.graph_context,
            "current_query": self.current_query,
            "intent_type": self.intent_type.value,
            "confusion_level": self.confusion_level.value,
            "detected_entities": self.detected_entities,
            "behavior": self.behavior.to_dict(),
            "session_id": self.session_id,
            "task_family": self.task_family,
            "role_profile": self.role_profile
        }


class StateTracker:
    """
    CEIEF 状态追踪器
    分析学习者输入，更新行为特征，维护对话记忆，
    输出完整状态供 Agent 控制器使用。
    """

    CLARIFICATION_SIGNALS = [
        "什么意思", "不理解", "能解释", "什么叫", "怎么理解", "不明白",
        "再说一遍", "详细说", "能举例", "举个例子"
    ]
    EVIDENCE_SIGNALS = [
        "为什么", "怎么知道", "有什么依据", "来源", "证据", "根据"
    ]
    COMPARISON_SIGNALS = [
        "比较", "对比", "区别", "不同", "相比", "差异", "异同"
    ]
    REFLECTION_SIGNALS = [
        "我觉得", "我认为", "感觉", "好像", "想到", "联系到"
    ]
    TRANSFER_SIGNALS = [
        "现代", "现在", "今天", "当代", "学校", "生活中", "实际上"
    ]

    KNOWN_ENTITIES = [
        "端午节", "春节", "中秋节", "清明节", "重阳节", "七夕",
        "屈原", "孔子", "孟子", "王昭君", "花木兰",
        "孝道", "礼仪", "集体记忆", "社区认同", "文化认同",
        "成人礼", "祭祖", "祭祀", "婚礼", "葬礼",
        "儒家", "道家", "佛教", "传统文化", "非物质文化遗产",
        "价值观", "和谐", "尊重", "仁义", "礼义廉耻"
    ]

    def __init__(self, config: Dict[str, Any]):
        self.dialogue_config = config.get("dialogue", {})
        self.memory_window = self.dialogue_config.get("memory_window", 10)
        self.current_state: Optional[InteractionState] = None
        self._previous_keywords: List[str] = []
        logger.info("StateTracker initialized")

    def initialize_session(
        self,
        session_id: str,
        task_family: str = "",
        role_profile: str = ""
    ) -> InteractionState:
        """初始化新的对话会话状态"""
        self.current_state = InteractionState(
            memory=DialogueMemory(window_size=self.memory_window),
            behavior=BehaviorFeatures(),
            session_id=session_id,
            task_family=task_family,
            role_profile=role_profile
        )
        self._previous_keywords = []
        logger.info(f"Session initialized: {session_id}")
        return self.current_state

    def update(
        self,
        learner_query: str,
        kg_evidence: List[str],
        assistant_response: Optional[str] = None,
        last_action: Optional[str] = None
    ) -> InteractionState:
        """更新状态（每轮调用一次）"""
        if self.current_state is None:
            self.current_state = InteractionState()

        state = self.current_state
        state.current_query = learner_query
        state.graph_context = kg_evidence

        state.memory.add_turn("user", learner_query)
        if assistant_response:
            state.memory.add_turn("assistant", assistant_response)
        if last_action:
            state.memory.previous_actions.append(last_action)

        self._update_behavior(state, learner_query)
        state.intent_type = self._detect_intent(learner_query)
        state.confusion_level = self._assess_confusion(state)
        state.detected_entities = self._extract_entities(learner_query)

        logger.debug(
            f"State updated | depth={state.behavior.dialogue_depth} "
            f"| intent={state.intent_type.value} "
            f"| confusion={state.confusion_level.value}"
        )
        return state

    def _update_behavior(self, state: InteractionState, query: str):
        """更新行为特征向量"""
        b = state.behavior
        b.dialogue_depth += 1

        if any(sig in query for sig in self.CLARIFICATION_SIGNALS):
            b.clarification_count += 1
        if any(sig in query for sig in self.EVIDENCE_SIGNALS):
            b.evidence_request_count += 1
        if any(sig in query for sig in self.COMPARISON_SIGNALS):
            b.perspective_switch_count += 1
        if len(query) < 20:
            b.short_input_count += 1
        if "?" in query or "？" in query:
            b.question_count += 1

        current_keywords = self._extract_keywords(query)
        repeated = set(current_keywords) & set(self._previous_keywords)
        if repeated:
            b.keyword_repetition_count += len(repeated)
        self._previous_keywords = current_keywords

    def _detect_intent(self, query: str) -> LearnerIntentType:
        """检测学习者意图类型"""
        if any(sig in query for sig in self.CLARIFICATION_SIGNALS):
            return LearnerIntentType.CLARIFICATION_REQUEST
        if any(sig in query for sig in self.EVIDENCE_SIGNALS):
            return LearnerIntentType.EXPLANATORY_QUESTION
        if any(sig in query for sig in self.COMPARISON_SIGNALS):
            return LearnerIntentType.COMPARISON_REQUEST
        if any(sig in query for sig in self.REFLECTION_SIGNALS):
            return LearnerIntentType.REFLECTION
        if any(sig in query for sig in self.TRANSFER_SIGNALS):
            return LearnerIntentType.TRANSFER_REQUEST
        return LearnerIntentType.FACTUAL_QUESTION

    def _assess_confusion(self, state: InteractionState) -> ConfusionLevel:
        """评估学习者困惑程度"""
        b = state.behavior
        score = 0
        if b.clarification_count >= 2:
            score += 2
        elif b.clarification_count == 1:
            score += 1
        if b.short_input_count >= 2:
            score += 1
        if b.keyword_repetition_count >= 3:
            score += 1
        if state.intent_type == LearnerIntentType.CLARIFICATION_REQUEST:
            score += 1
        if score >= 3:
            return ConfusionLevel.HIGH
        elif score >= 1:
            return ConfusionLevel.MEDIUM
        return ConfusionLevel.LOW

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        cleaned = re.sub(r'[，。！？、；：""''《》【】()（）\s]', ' ', text)
        tokens = cleaned.split()
        return [t for t in tokens if len(t) >= 2]

    def _extract_entities(self, text: str) -> List[str]:
        """从查询中提取文化实体"""
        return [e for e in self.KNOWN_ENTITIES if e in text]

    def get_state_summary(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        if self.current_state is None:
            return {}
        return self.current_state.to_dict()
