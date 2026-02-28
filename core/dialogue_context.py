from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class DialogueContext:
    """Manages conversation history and metadata for multi-turn interactions."""

    def __init__(self, max_history: int = 50):
        self.messages: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None

    def start_new_session(self) -> str:
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = datetime.now()
        self.messages = []
        logger.info("Started new dialogue session: %s", self.current_session_id)
        return self.current_session_id

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.current_session_id:
            self.start_new_session()

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)

        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context_window(self, window_size: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-window_size:] if self.messages else []

    def get_formatted_history(self, window_size: int = 10) -> str:
        recent = self.get_context_window(window_size)
        if not recent:
            return "无历史对话"
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_all_messages(self) -> List[Dict[str, Any]]:
        return self.messages

    def clear_context(self):
        self.messages = []
        self.current_session_id = None
        self.session_start_time = None
        logger.info("Cleared dialogue context")

    def get_session_info(self) -> Dict[str, Any]:
        if not self.current_session_id:
            return {"status": "no_active_session"}
        return {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time.isoformat(),
            "message_count": len(self.messages),
            "duration": (datetime.now() - self.session_start_time).total_seconds(),
        }
