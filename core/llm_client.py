import json
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from openai import OpenAI
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class LLMClient(LLM):
    """Enhanced LLM client with function-calling support for the agent."""

    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "silicon_flow"

    def _get_client(self) -> OpenAI:
        return OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )

    # ------------------------------------------------------------------
    # Core LLM call (used by LangChain chains)
    # ------------------------------------------------------------------
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = ""
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        content += choice.message.content
            else:
                return "Error: LLM did not return a valid response."
            if stop is not None:
                for s in stop:
                    if s in content:
                        content = content[:content.index(s)]
            return content
        except Exception as e:
            logger.error("LLM API call error: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Chat-style call with message list
    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        if response.choices:
            return response.choices[0].message.content
        return ""

    # ------------------------------------------------------------------
    # Agent-oriented: choose tools given a query
    # ------------------------------------------------------------------
    def plan_tools(
        self,
        user_query: str,
        tool_descriptions: List[Dict[str, str]],
        conversation_history: str = "",
        data_schema: str = "",
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to produce an ordered list of tool calls (ReAct-style).

        Returns a list of dicts: [{"tool": "<name>", "args": {...}, "reason": "..."}]
        """
        tools_text = "\n".join(
            f"- **{t['name']}**: {t['description']}  参数: {t.get('parameters', '无')}"
            for t in tool_descriptions
        )

        prompt = f"""你是一位专业的数据分析Agent。根据用户的查询，你需要决定调用哪些工具来完成分析任务。

## 可用工具
{tools_text}

## 数据库表结构
{data_schema}

## 对话历史
{conversation_history if conversation_history else "无"}

## 用户查询
{user_query}

## 要求
1. 分析用户的意图，选择合适的工具
2. 如果需要多个步骤，按顺序列出所有需要调用的工具
3. 对于每个工具调用，说明原因和参数
4. 如果是简单的问候/闲聊，返回空列表并在reason中说明
5. 尽量提供全面的分析，善用多个工具组合

请以**严格**的JSON格式返回，不要包含其他文字。格式如下：
```json
[
  {{"tool": "工具名称", "args": {{"参数名": "参数值"}}, "reason": "调用原因"}}
]
```

如果是普通对话而非数据分析请求，返回：
```json
[{{"tool": "general_chat", "args": {{"message": "用户消息"}}, "reason": "这是普通对话"}}]
```"""

        response = self.chat([{"role": "user", "content": prompt}])
        return self._parse_tool_plan(response)

    def _parse_tool_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse the JSON tool plan from LLM response."""
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
            text = text.strip()
            plan = json.loads(text)
            if isinstance(plan, list):
                return plan
            return [plan]
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool plan JSON, attempting recovery: %s", response[:300])
            try:
                import re
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass
            return [{"tool": "general_chat", "args": {"message": response}, "reason": "解析失败，作为普通对话处理"}]

    # ------------------------------------------------------------------
    # Generate a natural-language summary of tool results
    # ------------------------------------------------------------------
    def summarise_results(
        self,
        user_query: str,
        tool_results: List[Dict[str, Any]],
        conversation_history: str = "",
    ) -> str:
        results_text = ""
        for i, r in enumerate(tool_results, 1):
            results_text += f"\n### 步骤 {i}: {r.get('tool', 'unknown')} \n"
            results_text += f"**原因**: {r.get('reason', '')}\n"
            results_text += f"**结果**: {r.get('result', '')}\n"
            if r.get("image_path"):
                results_text += f"**图表**: 已生成图表 {r['image_path']}\n"

        prompt = f"""你是欧莱雅集团的智能数据分析助手 BeautyInsight。
请基于以下分析结果，为用户生成一份专业、易懂的分析报告。

## 对话历史
{conversation_history if conversation_history else "无"}

## 用户查询
{user_query}

## 分析步骤与结果
{results_text}

## 要求
1. 用专业但易懂的中文回答
2. 突出关键数据洞察
3. 如果有多个分析步骤，综合所有结果给出完整报告
4. 如果有图表生成，提及图表的关键信息
5. 适当给出业务建议或进一步分析方向
6. 格式清晰，使用适当的标题和列表"""

        return self.chat([{"role": "user", "content": prompt}])

    def general_chat(self, question: str) -> str:
        """Handle non-data-related conversations."""
        prompt = f"""你是欧莱雅集团的智能数据分析助手 BeautyInsight，专注于美妆行业数据分析。

专业领域：
- 精通欧莱雅集团的销售数据分析
- 帮助进行销量趋势、市场表现、品类分析等
- 擅长通过图表直观展示数据洞察

用户问题: "{question}"

要求：
- 用专业、友好的语言回答
- 如果用户询问功能，介绍数据分析和可视化能力并举例
- 适时建议可以进行的深入分析"""

        return self.chat([{"role": "user", "content": prompt}])
