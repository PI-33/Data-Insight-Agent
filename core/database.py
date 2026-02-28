import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.utilities import SQLDatabase

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_URI = "sqlite:///data/order_database.db"
_DEFAULT_DB_PATH = "data/order_database.db"


class DatabaseManager:
    """Centralised database access used by all tools."""

    def __init__(self, db_uri: str = _DEFAULT_DB_URI):
        self.db_uri = db_uri
        self.langchain_db = SQLDatabase.from_uri(db_uri)
        logger.info("DatabaseManager initialised: %s", db_uri)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------
    def get_table_names(self) -> List[str]:
        return self.langchain_db.get_usable_table_names()

    def get_table_info(self, table_name: Optional[str] = None) -> str:
        if table_name:
            return self.langchain_db.get_table_info([table_name])
        return self.langchain_db.get_table_info()

    def get_column_info(self, table_name: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(_DEFAULT_DB_PATH)
        try:
            cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                })
            return columns
        finally:
            conn.close()

    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        conn = sqlite3.connect(_DEFAULT_DB_PATH)
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        finally:
            conn.close()

    def get_row_count(self, table_name: str) -> int:
        conn = sqlite3.connect(_DEFAULT_DB_PATH)
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cur.fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    def execute_sql(self, sql: str) -> str:
        """Execute SQL via LangChain and return the raw result string."""
        from langchain_community.tools import QuerySQLDataBaseTool
        tool = QuerySQLDataBaseTool(db=self.langchain_db)
        return tool.invoke(sql)

    def execute_sql_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return the result as a DataFrame."""
        conn = sqlite3.connect(_DEFAULT_DB_PATH)
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()

    def get_distinct_values(self, table_name: str, column_name: str, limit: int = 20) -> List[Any]:
        conn = sqlite3.connect(_DEFAULT_DB_PATH)
        try:
            cur = conn.execute(
                f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT {limit}"
            )
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_numeric_columns(self, table_name: str) -> List[str]:
        cols = self.get_column_info(table_name)
        numeric_types = {"INT", "INTEGER", "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"}
        return [
            c["name"] for c in cols
            if any(t in c["type"].upper() for t in numeric_types)
        ]

    def get_date_columns(self, table_name: str) -> List[str]:
        cols = self.get_column_info(table_name)
        date_types = {"DATE", "DATETIME", "TIMESTAMP"}
        return [
            c["name"] for c in cols
            if any(t in c["type"].upper() for t in date_types)
        ]
