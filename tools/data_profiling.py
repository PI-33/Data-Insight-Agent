from typing import Any, Dict

import pandas as pd

from core.database import DatabaseManager
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class DataProfilingTool(BaseTool):
    """Profile a table: missing values, distributions, outliers, data quality."""

    name = "data_profiling"
    description = (
        "数据画像/质量分析：检测缺失值、异常值、数据分布、唯一值统计等。"
        "帮助全面了解数据质量和特征分布。"
    )
    parameters_description = (
        '{"table_name": "(可选)表名", '
        '"profile_type": "full/missing/distribution/outliers/unique", '
        '"column_name": "(可选)指定列"}'
    )

    def __init__(self, db: DatabaseManager):
        self.db = db

    def execute(self, **kwargs) -> Dict[str, Any]:
        table_name = kwargs.get("table_name")
        profile_type = kwargs.get("profile_type", "full")
        column_name = kwargs.get("column_name")

        tables = self.db.get_table_names()
        if not table_name:
            table_name = tables[0] if tables else None
        if not table_name:
            return {"success": False, "result": "没有可用的数据表"}

        row_count = self.db.get_row_count(table_name)
        sample_size = min(row_count, 5000)
        df = self.db.execute_sql_df(
            f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
        )

        if profile_type == "missing":
            return self._missing_analysis(df, table_name, row_count)
        elif profile_type == "distribution":
            return self._distribution_analysis(df, table_name, column_name)
        elif profile_type == "outliers":
            return self._outlier_analysis(df, table_name)
        elif profile_type == "unique":
            return self._unique_analysis(df, table_name)
        else:
            return self._full_profile(df, table_name, row_count)

    # ------------------------------------------------------------------
    def _full_profile(self, df: pd.DataFrame, table_name: str, total_rows: int) -> Dict[str, Any]:
        text = f"## 数据画像: `{table_name}`\n\n"
        text += f"- 总行数: {total_rows:,}\n"
        text += f"- 采样量: {len(df):,}\n"
        text += f"- 总列数: {len(df.columns)}\n\n"

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        text += "### 缺失值概况\n\n"
        text += "| 字段 | 缺失数 | 缺失率 |\n|------|--------|--------|\n"
        for col in df.columns:
            if missing[col] > 0:
                text += f"| {col} | {missing[col]} | {missing_pct[col]}% |\n"
        if missing.sum() == 0:
            text += "| (所有字段) | 0 | 0% |\n"

        text += "\n### 数值列统计\n\n"
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            stats = numeric.describe().round(2)
            text += stats.to_markdown() + "\n"

        text += "\n### 分类列统计\n\n"
        categorical = df.select_dtypes(include="object")
        for col in categorical.columns[:10]:
            nunique = df[col].nunique()
            top = df[col].value_counts().head(5)
            text += f"**{col}** ({nunique} 个唯一值)\n"
            for val, cnt in top.items():
                text += f"  - {val}: {cnt} ({cnt / len(df) * 100:.1f}%)\n"

        return {"success": True, "result": text, "dataframe": df}

    def _missing_analysis(self, df: pd.DataFrame, table_name: str, total_rows: int) -> Dict[str, Any]:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        text = f"## 缺失值分析: `{table_name}`\n\n"
        text += "| 字段 | 缺失数 | 缺失率 | 建议 |\n|------|--------|--------|------|\n"
        for col in df.columns:
            cnt = missing[col]
            pct = missing_pct[col]
            if pct > 50:
                advice = "考虑删除该列"
            elif pct > 20:
                advice = "需要填充处理"
            elif pct > 0:
                advice = "少量缺失可填充"
            else:
                advice = "完整"
            text += f"| {col} | {cnt} | {pct}% | {advice} |\n"

        total_missing = missing.sum()
        total_cells = len(df) * len(df.columns)
        text += f"\n整体缺失率: {total_missing}/{total_cells} ({total_missing / total_cells * 100:.2f}%)\n"
        return {"success": True, "result": text}

    def _distribution_analysis(self, df: pd.DataFrame, table_name: str, column_name: str = None) -> Dict[str, Any]:
        text = f"## 数据分布分析: `{table_name}`\n\n"
        cols = [column_name] if column_name and column_name in df.columns else df.select_dtypes(include="number").columns[:5]

        for col in cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            text += f"### {col}\n"
            text += f"- 均值: {series.mean():,.2f}\n"
            text += f"- 中位数: {series.median():,.2f}\n"
            text += f"- 标准差: {series.std():,.2f}\n"
            text += f"- 偏度: {series.skew():.2f}\n"
            text += f"- 峰度: {series.kurtosis():.2f}\n"
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            text += f"- 异常值数量(IQR法): {outlier_count}\n\n"

        return {"success": True, "result": text, "dataframe": df}

    def _outlier_analysis(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        text = f"## 异常值分析: `{table_name}`\n\n"
        numeric = df.select_dtypes(include="number")
        for col in numeric.columns:
            series = numeric[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            if len(outliers) > 0:
                text += f"### {col}\n"
                text += f"- 正常范围: [{lower:,.2f}, {upper:,.2f}]\n"
                text += f"- 异常值数量: {len(outliers)} ({len(outliers) / len(series) * 100:.1f}%)\n"
                text += f"- 异常值范围: [{outliers.min():,.2f}, {outliers.max():,.2f}]\n\n"

        if "无" not in text and "###" not in text:
            text += "未检测到明显异常值。\n"
        return {"success": True, "result": text}

    def _unique_analysis(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        text = f"## 唯一值分析: `{table_name}`\n\n"
        text += "| 字段 | 唯一值数 | 唯一率 | 类型 |\n|------|----------|--------|------|\n"
        for col in df.columns:
            nunique = df[col].nunique()
            pct = nunique / len(df) * 100
            dtype = str(df[col].dtype)
            text += f"| {col} | {nunique} | {pct:.1f}% | {dtype} |\n"
        return {"success": True, "result": text}
