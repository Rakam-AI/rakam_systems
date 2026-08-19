"""Chat History Components.

This module provides implementations for chat history management.
"""

from typing import TYPE_CHECKING

from ..._optional import make_getattr as _make_getattr
from .json_chat_history import JSONChatHistory
from .sql_chat_history import SQLChatHistory

if TYPE_CHECKING:  # import path unchanged for type checkers
    from .postgres_chat_history import PostgresChatHistory

# PostgresChatHistory needs psycopg2, which ships in the optional [postgres]
# extra -- resolved on first access so the extra stays optional (see _optional).
__getattr__ = _make_getattr(__name__, ["PostgresChatHistory"])

__all__ = ["JSONChatHistory", "SQLChatHistory", "PostgresChatHistory"]
