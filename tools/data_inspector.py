from typing import Any, Dict

import pandas as pd

from core.database import DatabaseManager
from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class DataInspectorTool(BaseTool):
    """Inspect database schema, understand table headers and data structure."""

    name = "data_inspector"
    description = (
        "查看数据库表结构、字段信息、数据样例和基本统计。"
        "用于了解数据集的结构和内容，是进行数据分析的第一步。"
    )
    parameters_description = (
        '{"table_name": "(可选)表名", "inspect_type": "schema/sample/overview/column_detail"}'
    )

    def __init__(self, db: DatabaseManager):
        self.db = db

    def execute(self, **kwargs) -> Dict[str, Any]:
        table_name = kwargs.get("table_name")
        inspect_type = kwargs.get("inspect_type", "overview")

        tables = self.db.get_table_names()
        if not table_name:
            table_name = tables[0] if tables else None
        if not table_name:
            return {"success": False, "result": "数据库中没有可用的表"}

        if inspect_type == "schema":
            return self._inspect_schema(table_name)
        elif inspect_type == "sample":
            return self._inspect_sample(table_name)
        elif inspect_type == "column_detail":
            return self._inspect_column_detail(table_name, kwargs.get("column_name"))
        else:
            return self._inspect_overview(table_name)

    def _inspect_schema(self, table_name: str) -> Dict[str, Any]:
        columns = self.db.get_column_info(table_name)
        schema_text = f"## 表 `{table_name}` 结构\n\n"
        schema_text += "| 序号 | 字段名 | 类型 | 非空 | 主键 |\n"
        schema_text += "|------|--------|------|------|------|\n"
        for col in columns:
            schema_text += (
                f"| {col['cid']} | {col['name']} | {col['type']} "
                f"| {'是' if col['notnull'] else '否'} "
                f"| {'是' if col['pk'] else '否'} |\n"
            )
        return {
            "success": True,
            "result": schema_text,
            "data": {"table_name": table_name, "columns": columns},
        }

    def _inspect_sample(self, table_name: str) -> Dict[str, Any]:
        df = self.db.get_sample_data(table_name, limit=5)
        sample_text = f"## 表 `{table_name}` 数据样例（前5行）\n\n"
        sample_text += df.to_markdown(index=False)
        return {
            "success": True,
            "result": sample_text,
            "data": {"table_name": table_name, "sample": df.to_dict()},
            "dataframe": df,
        }

    def _inspect_column_detail(self, table_name: str, column_name: str = None) -> Dict[str, Any]:
        columns = self.db.get_column_info(table_name)
        if column_name:
            columns = [c for c in columns if c["name"] == column_name]
            if not columns:
                return {"success": False, "result": f"未找到列: {column_name}"}

        result_parts = []
        for col in columns:
            col_name = col["name"]
            distinct_vals = self.db.get_distinct_values(table_name, col_name, limit=10)
            part = f"### 字段: `{col_name}`\n"
            part += f"- 类型: {col['type']}\n"
            part += f"- 示例值: {distinct_vals}\n"
            result_parts.append(part)

        return {
            "success": True,
            "result": "\n".join(result_parts),
            "data": {"table_name": table_name},
        }

    def _inspect_overview(self, table_name: str) -> Dict[str, Any]:
        row_count = self.db.get_row_count(table_name)
        columns = self.db.get_column_info(table_name)
        numeric_cols = self.db.get_numeric_columns(table_name)
        date_cols = self.db.get_date_columns(table_name)
        sample = self.db.get_sample_data(table_name, limit=3)

        text = f"## 数据概览: `{table_name}`\n\n"
        text += f"- **总行数**: {row_count:,}\n"
        text += f"- **总列数**: {len(columns)}\n"
        text += f"- **数值型字段** ({len(numeric_cols)}): {', '.join(numeric_cols)}\n"
        text += f"- **日期型字段** ({len(date_cols)}): {', '.join(date_cols)}\n\n"
        text += "### 字段列表\n"
        for col in columns:
            text += f"- `{col['name']}` ({col['type']})\n"
        text += f"\n### 数据样例（前3行）\n{sample.to_markdown(index=False)}\n"

        return {
            "success": True,
            "result": text,
            "data": {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "numeric_columns": numeric_cols,
                "date_columns": date_cols,
            },
        }
