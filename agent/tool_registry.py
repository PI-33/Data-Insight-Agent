from typing import Dict, List, Optional

from tools.base_tool import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry that the agent queries for available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_descriptions(self) -> List[Dict[str, str]]:
        return [t.get_description() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> dict:
        tool = self.get(name)
        if tool is None:
            return {"success": False, "result": f"未找到工具: {name}"}
        return tool.safe_execute(**kwargs)
