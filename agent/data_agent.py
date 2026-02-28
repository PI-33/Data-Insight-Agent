"""
DataAnalysisAgent – ReAct-style orchestrator inspired by Microsoft AutoGen.

The agent receives a user query, plans which tools to call, executes them
sequentially (passing intermediate results to subsequent steps when needed),
and finally synthesises a natural-language report.
"""

from typing import Any, Dict, List, Optional, Tuple

from agent.tool_registry import ToolRegistry
from core.database import DatabaseManager
from core.dialogue_context import DialogueContext
from core.llm_client import LLMClient
from tools.data_inspector import DataInspectorTool
from tools.data_profiling import DataProfilingTool
from tools.data_visualization import DataVisualizationTool
from tools.report_generator import ReportGeneratorTool
from tools.sql_query import SQLQueryTool
from tools.statistical_analysis import StatisticalAnalysisTool
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_STEPS = 8


class DataAnalysisAgent:
    """Central agent that orchestrates data-analysis tools."""

    def __init__(self, db_uri: str = "sqlite:///data/order_database.db"):
        self.db = DatabaseManager(db_uri)
        self.llm = LLMClient()
        self.context = DialogueContext()
        self.registry = ToolRegistry()
        self._register_tools()
        self._data_schema = self.db.get_table_info()
        logger.info("DataAnalysisAgent initialised with %d tools", len(self.registry.list_tools()))

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def _register_tools(self):
        self.registry.register(DataInspectorTool(self.db))
        self.registry.register(SQLQueryTool(self.db, self.llm))
        self.registry.register(DataVisualizationTool(self.db, self.llm))
        self.registry.register(StatisticalAnalysisTool(self.db, self.llm))
        self.registry.register(DataProfilingTool(self.db))
        self.registry.register(ReportGeneratorTool(self.llm))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query end-to-end and return the response.

        Returns:
            {
                "response": str,       # final natural-language answer
                "tool_results": [...], # per-step tool outputs
                "images": [str, ...],  # paths to generated charts
            }
        """
        logger.info(">>> New query: %s", user_query)

        self.context.add_message("user", user_query)
        history = self.context.get_formatted_history()

        # --- Step 1: Plan ---
        plan = self.llm.plan_tools(
            user_query,
            self.registry.get_descriptions(),
            conversation_history=history,
            data_schema=self._data_schema,
        )
        logger.info("Plan: %s", plan)

        # --- Handle general chat ---
        if len(plan) == 1 and plan[0].get("tool") == "general_chat":
            answer = self.llm.general_chat(user_query)
            self.context.add_message("assistant", answer, {"type": "general"})
            return {"response": answer, "tool_results": [], "images": []}

        # --- Step 2: Execute tools ---
        tool_results: List[Dict[str, Any]] = []
        images: List[str] = []
        accumulated_context = ""

        for i, step in enumerate(plan[:MAX_STEPS]):
            tool_name = step.get("tool", "")
            args = step.get("args", {})
            reason = step.get("reason", "")

            if tool_name == "general_chat":
                continue

            if tool_name == "report_generator" and accumulated_context:
                args.setdefault("analysis_results", accumulated_context)
                args.setdefault("question", user_query)

            if "context" not in args:
                args["context"] = history

            logger.info("Step %d: %s – %s", i + 1, tool_name, reason)
            result = self.registry.execute(tool_name, **args)

            result["tool"] = tool_name
            result["reason"] = reason
            tool_results.append(result)

            if result.get("image_path"):
                images.append(result["image_path"])

            result_text = result.get("result", "")
            accumulated_context += f"\n\n### {tool_name} ({reason})\n{result_text}"

        # --- Step 3: Synthesise ---
        if tool_results:
            final_response = self.llm.summarise_results(
                user_query, tool_results, history
            )
        else:
            final_response = self.llm.general_chat(user_query)

        self.context.add_message(
            "assistant",
            final_response,
            {
                "type": "analysis",
                "tools_used": [r["tool"] for r in tool_results],
                "images": images,
            },
        )

        return {
            "response": final_response,
            "tool_results": tool_results,
            "images": images,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def clear_context(self):
        self.context.clear_context()

    def get_session_info(self) -> Dict[str, Any]:
        return self.context.get_session_info()

    def get_available_tools(self) -> List[Dict[str, str]]:
        return self.registry.get_descriptions()
