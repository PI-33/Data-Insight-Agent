import re
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_classic.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.database import DatabaseManager
from core.llm_client import LLMClient
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLQueryTool(BaseTool):
    """Convert natural-language questions to SQL, execute, and return results."""

    name = "sql_query"
    description = (
        "将自然语言问题转换为SQL查询并执行，返回查询结果。"
        "适用于所有需要从数据库检索数据的场景。"
    )
    parameters_description = '{"question": "用户的自然语言问题", "context": "(可选)对话上下文"}'

    def __init__(self, db: DatabaseManager, llm: LLMClient):
        self.db = db
        self.llm = llm
        self.chain = self._build_chain()

    def _clean_sql_response(self, response: str) -> str:
        if "SQLQuery:" in response:
            cleaned = response.split("SQLQuery:", 1)[1].strip()
        elif "SELECT" in response.upper():
            start_idx = response.upper().find("SELECT")
            cleaned = response[start_idx:].strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```")[0].strip()
        else:
            logger.warning("Cannot extract SQL from response: %s", response[:200])
            cleaned = response
        cleaned = cleaned.rstrip(";").strip() + ""
        logger.debug("Cleaned SQL: %s", cleaned)
        return cleaned

    def _build_chain(self):
        write_query = create_sql_query_chain(self.llm, self.db.langchain_db)
        answer_prompt = PromptTemplate.from_template(
            """基于以下信息回答问题：

对话历史：
{context}

当前问题：{question}
生成的 SQL 查询：{clean_query}
数据库返回结果：{result}

请用自然语言给出简洁、专业的答案。如果结果中的数值为 0，明确说明"没有记录"。
用中文回答，格式清晰。"""
        )
        chain = (
            RunnablePassthrough.assign(
                question=lambda x: x["question"],
                context=lambda x: x.get("context", "无"),
            )
            .assign(
                clean_query=write_query | RunnableLambda(self._clean_sql_response)
            )
            .assign(
                result=lambda x: self.db.execute_sql(x["clean_query"])
            )
            .assign(
                response=(
                    {
                        "question": lambda x: x["question"],
                        "context": lambda x: x["context"],
                        "clean_query": lambda x: x["clean_query"],
                        "result": lambda x: x["result"],
                    }
                    | answer_prompt
                    | self.llm
                    | StrOutputParser()
                )
            )
            | {
                "response": lambda x: x["response"],
                "clean_query": lambda x: x["clean_query"],
                "sql_result": lambda x: x["result"],
            }
        )
        return chain

    def execute(self, **kwargs) -> Dict[str, Any]:
        question = kwargs.get("question", "")
        context = kwargs.get("context", "无")

        if not question:
            return {"success": False, "result": "请提供查询问题"}

        result = self.chain.invoke({"question": question, "context": context})
        response = result["response"]
        sql = result["clean_query"]
        raw = result["sql_result"]

        return {
            "success": True,
            "result": response,
            "data": {"sql": sql, "raw_result": raw},
        }
