from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Abstract base class that every analysis tool must extend."""

    name: str = "base_tool"
    description: str = "Base tool"
    parameters_description: str = "无"

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Run the tool and return a result dict.

        Every implementation MUST return a dict with at least:
        - "success": bool
        - "result": str  (human-readable summary)

        Optionally:
        - "data": any structured data
        - "image_path": str  (path to generated chart)
        - "dataframe": pd.DataFrame
        """
        ...

    def get_description(self) -> Dict[str, str]:
        """Return metadata used by the agent to choose tools."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_description,
        }

    def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """Wrapper that catches unexpected errors."""
        try:
            logger.info("Executing tool [%s] with args: %s", self.name, kwargs)
            result = self.execute(**kwargs)
            logger.info("Tool [%s] completed successfully", self.name)
            return result
        except Exception as e:
            logger.error("Tool [%s] failed: %s", self.name, e, exc_info=True)
            return {
                "success": False,
                "result": f"工具 {self.name} 执行失败: {str(e)}",
            }
