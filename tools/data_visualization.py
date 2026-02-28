import ast
import io
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

from core.database import DatabaseManager
from core.llm_client import LLMClient
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)

_IMG_DIR = "viz_images"
os.makedirs(_IMG_DIR, exist_ok=True)


class DataVisualizationTool(BaseTool):
    """Create data visualisations: line, bar, pie, scatter, histogram, heatmap."""

    name = "data_visualization"
    description = (
        "根据数据生成可视化图表，支持折线图、柱状图、饼图、散点图、直方图、热力图等。"
        "可以接受自然语言描述自动选择图表类型，也可以直接传入数据。"
    )
    parameters_description = (
        '{"question": "可视化描述", "chart_type": "(可选)line/bar/pie/scatter/histogram/heatmap", '
        '"sql": "(可选)直接提供SQL", "title": "(可选)图表标题"}'
    )

    def __init__(self, db: DatabaseManager, llm: LLMClient):
        self.db = db
        self.llm = llm
        self._setup_fonts()

    def _setup_fonts(self):
        try:
            font_paths = font_manager.findSystemFonts()
            chinese_fonts = [
                f for f in font_paths
                if any(n in f.lower() for n in ["simhei", "noto", "wqy", "wenquanyi", "microsoftyahei"])
            ]
            if chinese_fonts:
                self._font_prop = font_manager.FontProperties(fname=chinese_fonts[0])
            else:
                self._font_prop = font_manager.FontProperties(family="sans-serif")
        except Exception:
            self._font_prop = font_manager.FontProperties(family="sans-serif")
        plt.rcParams["axes.unicode_minus"] = False

    def execute(self, **kwargs) -> Dict[str, Any]:
        question = kwargs.get("question", "")
        chart_type = kwargs.get("chart_type")
        sql = kwargs.get("sql")
        title = kwargs.get("title", "")

        if sql:
            df = self._execute_sql_to_df(sql)
        elif question:
            sql = self._generate_viz_sql(question)
            if not sql:
                return {"success": False, "result": "无法为该问题生成可视化SQL"}
            df = self._execute_sql_to_df(sql)
        else:
            return {"success": False, "result": "请提供可视化描述或SQL"}

        if df is None or df.empty:
            return {"success": False, "result": "查询未返回数据，无法生成可视化", "data": {"sql": sql}}

        if not chart_type:
            chart_type = self._infer_chart_type(df, question)

        if not title:
            title = self._generate_title(question, chart_type)

        img_path = self._create_chart(df, chart_type, title)
        if not img_path:
            return {"success": False, "result": "图表生成失败", "data": {"sql": sql}}

        summary = self._summarise_data(df)
        return {
            "success": True,
            "result": f"已生成{self._chart_type_cn(chart_type)}。\n{summary}",
            "image_path": img_path,
            "data": {"sql": sql, "chart_type": chart_type, "rows": len(df)},
            "dataframe": df,
        }

    # ------------------------------------------------------------------
    # SQL generation for visualisation
    # ------------------------------------------------------------------
    def _generate_viz_sql(self, question: str) -> Optional[str]:
        schema = self.db.get_table_info()
        prompt = f"""你是一位SQL专家。请为以下可视化需求生成SQL查询。

数据库表结构:
{schema}

可视化需求: {question}

要求:
1. SQL结果的第一列应为X轴（通常是时间或分类字段）
2. 其余列为Y轴数值
3. 按X轴排序
4. 只返回SQL语句，不要其他内容
5. 使用聚合函数时注意GROUP BY"""

        response = self.llm.chat([{"role": "user", "content": prompt}])
        sql = self._extract_sql(response)
        logger.info("Generated viz SQL: %s", sql)
        return sql

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

    def _execute_sql_to_df(self, sql: str) -> Optional[pd.DataFrame]:
        try:
            df = self.db.execute_sql_df(sql)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass
                try:
                    if df[col].dtype == object:
                        test = pd.to_datetime(df[col], errors="coerce")
                        if test.notna().sum() > len(df) * 0.5:
                            df[col] = test
                except Exception:
                    pass
            return df
        except Exception as e:
            logger.error("SQL execution failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Chart type inference
    # ------------------------------------------------------------------
    def _infer_chart_type(self, df: pd.DataFrame, question: str) -> str:
        q = question.lower()
        keyword_map = {
            "pie": ["饼图", "占比", "比例", "构成", "pie"],
            "scatter": ["散点", "相关性", "scatter", "关系"],
            "histogram": ["直方图", "分布", "频率", "histogram"],
            "heatmap": ["热力图", "heatmap", "相关矩阵"],
            "line": ["趋势", "变化", "走势", "trend", "折线", "line"],
            "bar": ["柱状", "对比", "排名", "bar", "top"],
        }
        for ctype, keywords in keyword_map.items():
            for kw in keywords:
                if kw in q:
                    return ctype

        x_col = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            return "line"
        if len(df) <= 15:
            return "bar"
        return "bar"

    # ------------------------------------------------------------------
    # Chart creation
    # ------------------------------------------------------------------
    def _create_chart(self, df: pd.DataFrame, chart_type: str, title: str) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.set_style("whitegrid")

            method = getattr(self, f"_draw_{chart_type}", None)
            if method is None:
                method = self._draw_bar
            method(df, ax, title)

            plt.tight_layout()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(_IMG_DIR, f"viz_{chart_type}_{ts}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Chart saved: %s", path)
            return path
        except Exception as e:
            logger.error("Chart creation failed: %s", e, exc_info=True)
            plt.close("all")
            return None

    def _draw_line(self, df: pd.DataFrame, ax, title: str):
        x = df.columns[0]
        for col in df.columns[1:]:
            if pd.api.types.is_numeric_dtype(df[col]):
                ax.plot(df[x], df[col], marker="o", label=col)
        ax.set_xlabel(x, fontproperties=self._font_prop)
        ax.set_ylabel("数值", fontproperties=self._font_prop)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_fontproperties(self._font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self._font_prop)
        if len(df.columns) > 2:
            ax.legend(prop=self._font_prop)

    def _draw_bar(self, df: pd.DataFrame, ax, title: str):
        x = df.columns[0]
        numeric_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = [df.columns[1]]
            df[numeric_cols[0]] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")

        x_vals = df[x].astype(str)
        width = 0.8 / max(len(numeric_cols), 1)
        positions = np.arange(len(x_vals))

        for i, col in enumerate(numeric_cols):
            offset = (i - len(numeric_cols) / 2 + 0.5) * width
            ax.bar(positions + offset, df[col], width=width, label=col)

        ax.set_xticks(positions)
        ax.set_xticklabels(x_vals, rotation=45, ha="right", fontproperties=self._font_prop)
        ax.set_xlabel(x, fontproperties=self._font_prop)
        ax.set_ylabel("数值", fontproperties=self._font_prop)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self._font_prop)
        if len(numeric_cols) > 1:
            ax.legend(prop=self._font_prop)

    def _draw_pie(self, df: pd.DataFrame, ax, title: str):
        label_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        values = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        labels = df[label_col].astype(str)
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        for t in texts:
            t.set_fontproperties(self._font_prop)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)

    def _draw_scatter(self, df: pd.DataFrame, ax, title: str):
        x = df.columns[0]
        y = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), alpha=0.6)
        ax.set_xlabel(x, fontproperties=self._font_prop)
        ax.set_ylabel(y, fontproperties=self._font_prop)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)

    def _draw_histogram(self, df: pd.DataFrame, ax, title: str):
        numeric_cols = df.select_dtypes(include="number").columns
        col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(data, bins=min(30, max(10, len(data) // 5)), edgecolor="white", alpha=0.7)
        ax.set_xlabel(col, fontproperties=self._font_prop)
        ax.set_ylabel("频次", fontproperties=self._font_prop)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)

    def _draw_heatmap(self, df: pd.DataFrame, ax, title: str):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            self._draw_bar(df, ax, title)
            return
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(title, fontproperties=self._font_prop, fontsize=14)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _summarise_data(self, df: pd.DataFrame) -> str:
        parts = [f"数据包含 {len(df)} 行, {len(df.columns)} 列。"]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                parts.append(
                    f"- {col}: 总和={df[col].sum():,.2f}, "
                    f"均值={df[col].mean():,.2f}, "
                    f"最大={df[col].max():,.2f}, 最小={df[col].min():,.2f}"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                parts.append(f"- {col}: 范围 {df[col].min()} ~ {df[col].max()}")
        return "\n".join(parts)

    @staticmethod
    def _chart_type_cn(chart_type: str) -> str:
        mapping = {
            "line": "折线图",
            "bar": "柱状图",
            "pie": "饼图",
            "scatter": "散点图",
            "histogram": "直方图",
            "heatmap": "热力图",
        }
        return mapping.get(chart_type, "图表")

    @staticmethod
    def _generate_title(question: str, chart_type: str) -> str:
        if question:
            return question[:50]
        return f"数据{DataVisualizationTool._chart_type_cn(chart_type)}"
