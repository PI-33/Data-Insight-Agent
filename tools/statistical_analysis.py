from typing import Any, Dict, Optional

import pandas as pd

from core.database import DatabaseManager
from core.llm_client import LLMClient
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalysisTool(BaseTool):
    """Perform statistical analysis on database data."""

    name = "statistical_analysis"
    description = (
        "对数据进行统计分析，包括描述性统计、分组统计、相关性分析、趋势分析等。"
        "适用于需要深入统计信息的场景。"
    )
    parameters_description = (
        '{"question": "分析需求描述", '
        '"analysis_type": "descriptive/group_by/correlation/trend/top_n/comparison", '
        '"sql": "(可选)直接提供SQL"}'
    )

    def __init__(self, db: DatabaseManager, llm: LLMClient):
        self.db = db
        self.llm = llm

    def execute(self, **kwargs) -> Dict[str, Any]:
        question = kwargs.get("question", "")
        analysis_type = kwargs.get("analysis_type", "descriptive")
        sql = kwargs.get("sql")

        if sql:
            df = self.db.execute_sql_df(sql)
        elif question:
            sql = self._generate_analysis_sql(question, analysis_type)
            if not sql:
                return {"success": False, "result": "无法生成分析SQL"}
            df = self._safe_query(sql)
            if df is None:
                return {"success": False, "result": f"SQL执行失败: {sql}"}
        else:
            return {"success": False, "result": "请提供分析需求"}

        if df.empty:
            return {"success": False, "result": "查询结果为空"}

        if analysis_type == "descriptive":
            return self._descriptive(df, question)
        elif analysis_type == "correlation":
            return self._correlation(df, question)
        elif analysis_type == "trend":
            return self._trend(df, question)
        elif analysis_type == "top_n":
            return self._top_n(df, question)
        elif analysis_type == "comparison":
            return self._comparison(df, question)
        else:
            return self._descriptive(df, question)

    # ------------------------------------------------------------------
    def _generate_analysis_sql(self, question: str, analysis_type: str) -> Optional[str]:
        schema = self.db.get_table_info()
        prompt = f"""你是SQL专家。为以下统计分析需求生成SQL。

表结构:
{schema}

分析需求: {question}
分析类型: {analysis_type}

只返回SQL，不要其他说明。"""
        response = self.llm.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    @staticmethod
    def _extract_sql(text: str) -> Optional[str]:
        text = text.strip()
        if "```sql" in text:
            text = text.split("```sql", 1)[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```")[0].strip()
        if "SELECT" in text.upper():
            idx = text.upper().find("SELECT")
            return text[idx:].rstrip(";").strip()
        return None

    def _safe_query(self, sql: str) -> Optional[pd.DataFrame]:
        try:
            return self.db.execute_sql_df(sql)
        except Exception as e:
            logger.error("Statistical SQL failed: %s | %s", sql, e)
            return None

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------
    def _descriptive(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {
                "success": True,
                "result": f"查询结果（{len(df)}行 x {len(df.columns)}列）：\n{df.head(20).to_markdown(index=False)}",
                "dataframe": df,
            }

        stats = numeric.describe().round(2)
        text = f"## 描述性统计\n\n{stats.to_markdown()}\n\n"
        text += f"数据共 {len(df)} 行, {len(df.columns)} 列。\n"
        for col in numeric.columns:
            skew = numeric[col].skew()
            text += f"- **{col}** 偏度={skew:.2f} ({'右偏' if skew > 0.5 else '左偏' if skew < -0.5 else '近似正态'})\n"

        return {
            "success": True,
            "result": text,
            "data": {"stats": stats.to_dict()},
            "dataframe": df,
        }

    def _correlation(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return {"success": False, "result": "数值列不足2列，无法进行相关性分析"}

        corr = numeric.corr().round(3)
        text = f"## 相关性分析\n\n{corr.to_markdown()}\n\n"
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong.append((corr.columns[i], corr.columns[j], val))
        if strong:
            text += "### 强相关字段对\n"
            for a, b, v in strong:
                text += f"- {a} ↔ {b}: {v:.3f}\n"

        return {
            "success": True,
            "result": text,
            "data": {"correlation": corr.to_dict()},
            "dataframe": df,
        }

    def _trend(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_cols:
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].notna().sum() > len(df) * 0.5:
                        date_cols.append(col)
                        break
                except Exception:
                    pass

        text = f"## 趋势分析\n\n数据包含 {len(df)} 个时间点。\n"
        numeric = df.select_dtypes(include="number")
        for col in numeric.columns:
            vals = numeric[col].dropna()
            if len(vals) < 2:
                continue
            pct_change = ((vals.iloc[-1] - vals.iloc[0]) / vals.iloc[0] * 100) if vals.iloc[0] != 0 else 0
            text += f"- **{col}**: 起始={vals.iloc[0]:,.2f}, 结束={vals.iloc[-1]:,.2f}, 变化率={pct_change:+.1f}%\n"

        text += f"\n### 数据预览\n{df.head(10).to_markdown(index=False)}\n"
        return {"success": True, "result": text, "dataframe": df}

    def _top_n(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        text = f"## Top-N 分析\n\n{df.to_markdown(index=False)}\n"
        return {"success": True, "result": text, "dataframe": df}

    def _comparison(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        text = f"## 对比分析\n\n{df.to_markdown(index=False)}\n\n"
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            for col in numeric.columns:
                text += f"- **{col}** 最大值行: {df.iloc[numeric[col].idxmax()].to_dict()}\n"
        return {"success": True, "result": text, "dataframe": df}
