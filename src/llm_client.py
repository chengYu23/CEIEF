"""
CEIEF LLM Client Module
文化教育交互增强框架 - 大语言模型客户端

负责与 Anthropic Claude API 进行通信，支持多轮对话、提示词组装与响应解析。
"""

import time
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import anthropic
import httpx
from loguru import logger


@dataclass
class Message:
    """单条对话消息结构"""
    role: str          # 'user' or 'assistant'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM响应结构"""
    content: str
    model: str
    usage: Dict[str, int]
    stop_reason: str
    latency_ms: float
    raw_response: Any = None


class LLMClient:
    """
    CEIEF LLM 客户端
    封装对 Anthropic Claude API 的访问，支持：
    - 系统提示词注入
    - 多轮对话历史管理
    - 知识图谱证据注入
    - 动作指令附加
    - 自动重试机制
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LLM 客户端

        Args:
            config: 来自 config.yaml 的 llm 配置节
        """
        self.config = config
        self.model = config.get("model", "claude-sonnet-4-6")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 60)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 2)

        # 初始化 Anthropic 客户端，支持自定义 base_url
        self.client = anthropic.Anthropic(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.anthropic.com"),
            http_client=httpx.Client(timeout=self.timeout)
        )

        logger.info(f"LLMClient initialized with model={self.model}")

    def build_prompt_package(
        self,
        system_prompt: str,
        learner_query: str,
        history: List[Message],
        kg_evidence: Optional[List[str]] = None,
        action_instructions: Optional[str] = None,
        role_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        组装完整提示词包

        Args:
            system_prompt: 系统级指令
            learner_query: 学习者当前输入
            history: 历史对话消息列表
            kg_evidence: 知识图谱检索到的三元组列表
            action_instructions: Agent 选定动作对应的提示指令片段
            role_profile: 角色扮演描述文本

        Returns:
            包含 system 和 messages 的提示词包字典
        """
        # 构建系统提示词
        full_system = system_prompt

        if role_profile:
            full_system += f"\n\n## Role Profile\n{role_profile}"

        if kg_evidence and len(kg_evidence) > 0:
            evidence_text = "\n".join(f"- {e}" for e in kg_evidence)
            full_system += f"\n\n## Retrieved Cultural Evidence from Knowledge Graph\n{evidence_text}"

        if action_instructions:
            full_system += f"\n\n## Interaction Action Guidance\n{action_instructions}"

        # 构建消息列表（历史 + 当前输入）
        messages = []
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": learner_query})

        return {
            "system": full_system,
            "messages": messages
        }

    def generate(
        self,
        system_prompt: str,
        learner_query: str,
        history: List[Message],
        kg_evidence: Optional[List[str]] = None,
        action_instructions: Optional[str] = None,
        role_profile: Optional[str] = None
    ) -> LLMResponse:
        """
        生成 LLM 响应（同步）

        Args:
            system_prompt: 系统提示词
            learner_query: 学习者输入
            history: 对话历史
            kg_evidence: KG 检索证据
            action_instructions: Agent 动作指令
            role_profile: 角色描述

        Returns:
            LLMResponse 对象
        """
        prompt_pkg = self.build_prompt_package(
            system_prompt=system_prompt,
            learner_query=learner_query,
            history=history,
            kg_evidence=kg_evidence,
            action_instructions=action_instructions,
            role_profile=role_profile
        )

        for attempt in range(1, self.retry_attempts + 1):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    system=prompt_pkg["system"],
                    messages=prompt_pkg["messages"]
                )
                latency_ms = (time.time() - start_time) * 1000

                content = response.content[0].text if response.content else ""

                llm_response = LLMResponse(
                    content=content,
                    model=response.model,
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    },
                    stop_reason=response.stop_reason or "end_turn",
                    latency_ms=latency_ms,
                    raw_response=response
                )

                logger.info(
                    f"LLM response generated | tokens={response.usage.input_tokens}+{response.usage.output_tokens} "
                    f"| latency={latency_ms:.1f}ms"
                )
                return llm_response

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise

            except anthropic.APIConnectionError as e:
                logger.error(f"Connection error (attempt {attempt}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
                else:
                    raise

            except anthropic.APIError as e:
                logger.error(f"API error (attempt {attempt}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def generate_llm_only(
        self,
        system_prompt: str,
        learner_query: str,
        history: List[Message]
    ) -> LLMResponse:
        """LLM-only 条件：不使用 KG 证据，不使用 Agent 动作"""
        return self.generate(
            system_prompt=system_prompt,
            learner_query=learner_query,
            history=history,
            kg_evidence=None,
            action_instructions=None
        )

    def format_response_to_message(self, response: LLMResponse) -> Message:
        """将 LLMResponse 转换为 Message 对象"""
        return Message(
            role="assistant",
            content=response.content,
            metadata={
                "model": response.model,
                "usage": response.usage,
                "latency_ms": response.latency_ms,
                "stop_reason": response.stop_reason
            }
        )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # 快速测试
    config = load_config()
    client = LLMClient(config["llm"])

    system_prompt = (
        "You are a cultural education assistant specializing in Chinese cultural history. "
        "Provide insightful, contextually grounded responses."
    )
    learner_query = "请解释端午节的文化意义。"

    response = client.generate(
        system_prompt=system_prompt,
        learner_query=learner_query,
        history=[],
        kg_evidence=[
            "(端午节, commemorates, 屈原)",
            "(端午节, embodies, 集体记忆)",
            "(集体记忆, transfers_to, 社区认同)"
        ]
    )
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage}")
