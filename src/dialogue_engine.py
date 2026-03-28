"""
CEIEF Dialogue Engine Module
文化教育交互增强框架 - 对话引擎

核心编排层，整合 KGRetriever、AgentController、StateTracker 与 LLMClient，
实现四种实验条件下的多轮对话流程：
  - LLM-only
  - KG + LLM
  - LLM + Agent
  - Full CEIEF (KG + LLM + Agent)
"""

import uuid
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.llm_client import LLMClient, Message, LLMResponse
from src.kg_retriever import KGRetriever, SubgraphSummary
from src.agent_controller import AgentController, ActionResult
from src.state_tracker import StateTracker, InteractionState


# 实验条件枚举
CONDITION_LLM_ONLY = "llm_only"
CONDITION_KG_LLM = "kg_llm"
CONDITION_LLM_AGENT = "llm_agent"
CONDITION_FULL_CEIEF = "full_ceief"


@dataclass
class TurnLog:
    """单轮对话日志"""
    turn_id: int
    session_id: str
    condition: str
    learner_query: str
    kg_evidence: List[str]
    selected_actions: List[str]
    action_reasoning: str
    llm_response: str
    state_snapshot: Dict[str, Any]
    latency_ms: float
    timestamp: str
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "condition": self.condition,
            "learner_query": self.learner_query,
            "kg_evidence": self.kg_evidence,
            "selected_actions": self.selected_actions,
            "action_reasoning": self.action_reasoning,
            "llm_response": self.llm_response,
            "state_snapshot": self.state_snapshot,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "usage": self.usage
        }


@dataclass
class DialogueSession:
    """对话会话记录"""
    session_id: str
    condition: str
    task_family: str
    role_profile: str
    turns: List[TurnLog] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "condition": self.condition,
            "task_family": self.task_family,
            "role_profile": self.role_profile,
            "turns": [t.to_dict() for t in self.turns],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_turns": self.total_turns
        }


class DialogueEngine:
    """
    CEIEF 对话引擎
    编排三层架构完成多轮文化教育交互。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话引擎

        Args:
            config: 来自 config.yaml 的完整配置
        """
        self.config = config
        self.dialogue_config = config.get("dialogue", {})
        self.logging_config = config.get("logging", {})
        self.max_turns = self.dialogue_config.get("max_turns", 20)

        # 初始化各组件
        self.llm_client = LLMClient(config["llm"])
        self.kg_retriever = KGRetriever(config)
        self.agent_controller = AgentController(config)
        self.state_tracker = StateTracker(config)

        # 当前会话
        self.current_session: Optional[DialogueSession] = None
        self._dialogue_history: List[Message] = []

        logger.info("DialogueEngine initialized")

    def load_system_prompt(self, condition: str, role_profile: str = "") -> str:
        """
        根据实验条件加载对应系统提示词

        Args:
            condition: 实验条件名称
            role_profile: 角色描述文本

        Returns:
            系统提示词字符串
        """
        prompts_dir = Path(self.config.get("paths", {}).get("prompts_dir", "prompts/"))

        condition_prompt_map = {
            CONDITION_LLM_ONLY: prompts_dir / "system" / "llm_only_prompt.md",
            CONDITION_KG_LLM: prompts_dir / "system" / "kg_llm_prompt.md",
            CONDITION_LLM_AGENT: prompts_dir / "system" / "llm_agent_prompt.md",
            CONDITION_FULL_CEIEF: prompts_dir / "system" / "ceief_global_prompt.md"
        }

        prompt_path = condition_prompt_map.get(condition, condition_prompt_map[CONDITION_LLM_ONLY])

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                base_prompt = f.read()
        else:
            base_prompt = self._get_default_system_prompt(condition)
            logger.warning(f"Prompt file not found: {prompt_path}, using default")

        return base_prompt

    def _get_default_system_prompt(self, condition: str) -> str:
        """返回默认系统提示词（备用）"""
        base = (
            "You are a cultural education assistant specializing in Chinese cultural history "
            "and traditions. Your role is to facilitate deep cultural understanding through "
            "historically grounded, contextually appropriate responses.\n\n"
            "Guidelines:\n"
            "- Maintain cultural accuracy and historical context\n"
            "- Adapt your explanations to the learner's level of understanding\n"
            "- Use concrete examples to illustrate abstract cultural concepts\n"
            "- Encourage critical thinking about cultural values and practices"
        )
        return base

    def start_session(
        self,
        condition: str,
        task_family: str = "historical_explanation",
        role_profile: str = "cultural_teacher"
    ) -> str:
        """
        开始新的对话会话

        Args:
            condition: 实验条件
            task_family: 任务家族
            role_profile: 角色描述

        Returns:
            session_id
        """
        session_id = str(uuid.uuid4())
        self.current_session = DialogueSession(
            session_id=session_id,
            condition=condition,
            task_family=task_family,
            role_profile=role_profile,
            start_time=datetime.utcnow().isoformat()
        )
        self._dialogue_history = []
        self.state_tracker.initialize_session(
            session_id=session_id,
            task_family=task_family,
            role_profile=role_profile
        )
        logger.info(f"Session started: {session_id} | condition={condition}")
        return session_id

    def process_turn(
        self,
        learner_query: str,
        condition: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理单轮学习者输入，返回系统响应

        完整流程（伪代码）：
        输入 learner_query
        → KG 检索 evidence_set
        → 构建 subgraph_summary
        → 计算 state_features
        → Agent 选择 selected_action
        → 组装 prompt_package
        → LLM 生成 response
        → 记录 logs
        → 输出 response

        Args:
            learner_query: 学习者输入文本
            condition: 实验条件（覆盖session中的条件）

        Returns:
            包含响应和元数据的字典
        """
        if self.current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        effective_condition = condition or self.current_session.condition
        turn_id = len(self.current_session.turns) + 1
        t0 = time.time()

        # === Step 1: KG 检索 ===
        kg_evidence: List[str] = []
        subgraph: Optional[SubgraphSummary] = None

        if effective_condition in [CONDITION_KG_LLM, CONDITION_FULL_CEIEF]:
            subgraph = self.kg_retriever.retrieve_subgraph(
                query=learner_query,
                use_neo4j=True
            )
            kg_evidence = subgraph.to_evidence_list()
            logger.debug(f"KG retrieved {len(kg_evidence)} triples")

        # === Step 2: 状态更新 ===
        last_response = (
            self._dialogue_history[-1].content
            if self._dialogue_history and self._dialogue_history[-1].role == "assistant"
            else None
        )
        last_action = (
            self.current_session.turns[-1].selected_actions[0]
            if self.current_session.turns
            else None
        )
        state: InteractionState = self.state_tracker.update(
            learner_query=learner_query,
            kg_evidence=kg_evidence,
            assistant_response=last_response,
            last_action=last_action
        )

        # === Step 3: Agent 动作选择 ===
        selected_actions: List[ActionResult] = []
        action_instructions: Optional[str] = None

        if effective_condition in [CONDITION_LLM_AGENT, CONDITION_FULL_CEIEF]:
            selected_actions = self.agent_controller.get_multi_actions(state, n=2)
            action_instructions = self.agent_controller.combine_action_instructions(
                selected_actions
            )
            logger.debug(
                f"Agent selected actions: {[a.action_name for a in selected_actions]}"
            )

        # === Step 4: 加载系统提示词 ===
        system_prompt = self.load_system_prompt(
            effective_condition,
            self.current_session.role_profile
        )

        # === Step 5: LLM 生成 ===
        llm_response: LLMResponse = self.llm_client.generate(
            system_prompt=system_prompt,
            learner_query=learner_query,
            history=self._dialogue_history,
            kg_evidence=kg_evidence if kg_evidence else None,
            action_instructions=action_instructions,
            role_profile=self.current_session.role_profile
        )

        latency_ms = (time.time() - t0) * 1000

        # === Step 6: 更新对话历史 ===
        self._dialogue_history.append(Message(role="user", content=learner_query))
        self._dialogue_history.append(
            self.llm_client.format_response_to_message(llm_response)
        )

        # 保持历史窗口
        window = self.dialogue_config.get("memory_window", 10)
        if len(self._dialogue_history) > window * 2:
            self._dialogue_history = self._dialogue_history[-window * 2:]

        # === Step 7: 记录日志 ===
        turn_log = TurnLog(
            turn_id=turn_id,
            session_id=self.current_session.session_id,
            condition=effective_condition,
            learner_query=learner_query,
            kg_evidence=kg_evidence,
            selected_actions=[a.action_name for a in selected_actions],
            action_reasoning=(
                selected_actions[0].reasoning if selected_actions else "none"
            ),
            llm_response=llm_response.content,
            state_snapshot=state.to_dict(),
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            usage=llm_response.usage
        )
        self.current_session.turns.append(turn_log)
        self.current_session.total_turns = len(self.current_session.turns)

        logger.info(
            f"Turn {turn_id} completed | condition={effective_condition} "
            f"| latency={latency_ms:.1f}ms"
        )

        return {
            "turn_id": turn_id,
            "response": llm_response.content,
            "kg_evidence": kg_evidence,
            "actions": [a.action_name for a in selected_actions],
            "state": {
                "intent": state.intent_type.value,
                "confusion": state.confusion_level.value,
                "depth": state.behavior.dialogue_depth
            },
            "usage": llm_response.usage,
            "latency_ms": latency_ms
        }

    def end_session(self) -> DialogueSession:
        """结束当前会话并返回完整日志"""
        if self.current_session is None:
            raise RuntimeError("No active session")
        self.current_session.end_time = datetime.utcnow().isoformat()
        session = self.current_session
        self.current_session = None
        self._dialogue_history = []
        logger.info(f"Session ended: {session.session_id} | turns={session.total_turns}")
        return session

    def save_session_log(
        self, session: DialogueSession, output_dir: str = "logs/raw_dialogues/"
    ):
        """将会话日志保存为 JSON 文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{session.session_id}_{session.condition}.json"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Session log saved: {filepath}")
        return str(filepath)

    def run_demo_session(
        self,
        condition: str = CONDITION_FULL_CEIEF,
        queries: Optional[List[str]] = None
    ) -> DialogueSession:
        """
        运行演示会话（用于测试）

        Args:
            condition: 实验条件
            queries: 学习者输入列表

        Returns:
            完整的 DialogueSession
        """
        if queries is None:
            queries = [
                "请解释端午节不仅是纪念屈原，也是维系社区文化记忆的节日。",
                "为什么集体记忆对于文化认同如此重要？",
                "传统节庆与现代学校活动之间的文化意义有什么不同？"
            ]

        self.start_session(
            condition=condition,
            task_family="historical_explanation",
            role_profile="cultural_teacher"
        )

        for query in queries:
            try:
                result = self.process_turn(query)
                print(f"\n[Turn {result['turn_id']}]")
                print(f"Q: {query}")
                print(f"A: {result['response'][:300]}...")
                print(f"   KG: {len(result['kg_evidence'])} triples, "
                      f"Actions: {result['actions']}")
            except Exception as e:
                logger.error(f"Turn failed: {e}")
                break

        return self.end_session()

    def close(self):
        """释放资源"""
        self.kg_retriever.close()
        logger.info("DialogueEngine closed")
