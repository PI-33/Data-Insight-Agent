import os
from datetime import datetime
from typing import Any, Dict, List

from core.llm_client import LLMClient
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_DIR = "reports"
os.makedirs(_REPORT_DIR, exist_ok=True)


class ReportGeneratorTool(BaseTool):
    """Generate structured analysis reports from tool results."""

    name = "report_generator"
    description = (
        "根据已完成的分析结果生成结构化的数据分析报告。"
        "整合多个分析步骤的结果，给出综合洞察和建议。"
    )
    parameters_description = (
        '{"question": "原始分析需求", '
        '"analysis_results": "之前各步骤的分析结果", '
        '"report_type": "summary/detailed/executive"}'
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def execute(self, **kwargs) -> Dict[str, Any]:
        question = kwargs.get("question", "")
        analysis_results = kwargs.get("analysis_results", "")
        report_type = kwargs.get("report_type", "summary")

        if not analysis_results:
            return {"success": False, "result": "没有可用的分析结果"}

        report = self._generate_report(question, analysis_results, report_type)
        path = self._save_report(report, report_type)

        return {
            "success": True,
            "result": report,
            "data": {"report_path": path, "report_type": report_type},
        }

    def _generate_report(self, question: str, results: str, report_type: str) -> str:
        type_instructions = {
            "summary": "生成简洁的摘要报告，突出关键发现和结论。",
            "detailed": "生成详细的分析报告，包含完整的数据解读和多角度分析。",
            "executive": "生成面向管理层的简报，侧重业务洞察和决策建议。",
        }

        prompt = f"""你是欧莱雅集团的高级数据分析师。请基于以下分析结果生成专业报告。

## 分析需求
{question}

## 分析结果
{results}

## 报告要求
{type_instructions.get(report_type, type_instructions['summary'])}

## 格式要求
1. 使用清晰的标题层级
2. 关键数据加粗
3. 给出具体的业务洞察
4. 提供可操作的建议
5. 如有必要，指出数据局限性和进一步分析方向"""

        return self.llm.chat([{"role": "user", "content": prompt}])

    @staticmethod
    def _save_report(report: str, report_type: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_REPORT_DIR, f"report_{report_type}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Report saved: %s", path)
        return path
