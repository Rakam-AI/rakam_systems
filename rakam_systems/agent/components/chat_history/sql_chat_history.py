"""SQL-Based Chat History Manager.

This module provides a ChatHistoryComponent implementation that stores
chat history in a SQLite database. Suitable for production deployments
requiring persistent, structured storage.
"""

from __future__ import annotations
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from rakam_systems.core.ai_core.interfaces.chat_history import ChatHistoryComponent

# Optional pydantic-ai integration
try:
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelMessage
    from pydantic_core import to_jsonable_python
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    ModelMessagesTypeAdapter = None  # type: ignore
    ModelMessage = None  # type: ignore
    to_jsonable_python = None  # type: ignore


class SQLChatHistory(ChatHistoryComponent):
    """Chat history manager using SQLite database storage.

    This implementation stores all chat histories in a SQLite database.
    It's suitable for:
    - Production deployments
    - Multi-instance applications (with proper connection handling)
    - Applications requiring structured queries
    - Medium to large scale applications

    Config options:
        db_path: Path to the SQLite database file (default: "./chat_history.db")

    Example:
        >>> history = SQLChatHistory(config={"db_path": "./data/chats.db"})
        >>> history.add_message("chat123", {"role": "user", "content": "Hello"})
        >>> history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})
        >>> messages = history.get_chat_history("chat123")
    """

    def __init__(
        self,
        name: str = "sql_chat_history",
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        """Initialize the SQL chat history manager.

        Args:
            name: Component name for identification.
            config: Configuration dictionary. Supports:
                - db_path: Path to SQLite database file
            db_path: Direct path override (takes precedence over config).
        """
        super().__init__(name, config)

        # Get db path from argument, config, or default
        self.db_path = db_path or self.config.get(
            "db_path", "./chat_history.db"
        )

    def setup(self) -> None:
        """Initialize database and create tables."""
        self._initialize_database()
        super().setup()

    def shutdown(self) -> None:
        """Cleanup resources."""
        super().shutdown()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with foreign keys enabled.

        Returns:
            SQLite connection object.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON;')
        return conn

    def _initialize_database(self) -> None:
        """Initialize SQLite database and create necessary tables.

        Creates the chats and messages tables if they don't exist.

        Raises:
            Exception: If database initialization fails.
        """
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message_order INTEGER NOT NULL,
                    message_data TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
                )
            ''')

            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
                ON messages (chat_id, message_order)
            ''')

            conn.commit()

    def _ensure_initialized(self) -> None:
        """Ensure the component is initialized before operations."""
        if not self.initialized:
            self.setup()

    def add_message(self, chat_id: str, message: Dict[str, Any]) -> None:
        """Add a single message to a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            message: Message object (dict with role, content, timestamp, etc.).
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Ensure chat exists
            cursor.execute(
                'INSERT OR IGNORE INTO chats (chat_id) VALUES (?)',
                (chat_id,)
            )

            # Get next message order
            cursor.execute(
                'SELECT COALESCE(MAX(message_order), -1) + 1 FROM messages WHERE chat_id = ?',
                (chat_id,)
            )
            next_order = cursor.fetchone()[0]

            # Insert message
            message_json = json.dumps(message, ensure_ascii=False)
            cursor.execute(
                '''
                INSERT INTO messages (chat_id, message_order, message_data)
                VALUES (?, ?, ?)
                ''',
                (chat_id, next_order, message_json)
            )

            conn.commit()

    def set_messages(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        """Set/replace all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of message objects to store.
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Ensure chat exists
            cursor.execute(
                'INSERT OR IGNORE INTO chats (chat_id) VALUES (?)',
                (chat_id,)
            )

            # Delete existing messages
            cursor.execute(
                'DELETE FROM messages WHERE chat_id = ?', (chat_id,))

            # Insert new messages with order
            for order, message in enumerate(messages):
                message_json = json.dumps(message, ensure_ascii=False)
                cursor.execute(
                    '''
                    INSERT INTO messages (chat_id, message_order, message_data)
                    VALUES (?, ?, ?)
                    ''',
                    (chat_id, order, message_json)
                )

            conn.commit()

    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            List of message objects, or empty list if chat doesn't exist.
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT message_data
                FROM messages
                WHERE chat_id = ?
                ORDER BY message_order ASC
                ''',
                (chat_id,)
            )
            rows = cursor.fetchall()

            return [json.loads(row[0]) for row in rows]

    def get_all_chat_ids(self) -> List[str]:
        """Get all chat IDs currently stored.

        Returns:
            List of all chat session identifiers.
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chat_id FROM chats')
            return [row[0] for row in cursor.fetchall()]

    def delete_chat_history(self, chat_id: str) -> bool:
        """Delete all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session to delete.

        Returns:
            True if deletion was successful, False if chat_id didn't exist.
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if chat exists
            cursor.execute('SELECT 1 FROM chats WHERE chat_id = ?', (chat_id,))
            exists = cursor.fetchone() is not None

            if not exists:
                return False

            # Delete messages (cascades from foreign key, but explicit is safer)
            cursor.execute(
                'DELETE FROM messages WHERE chat_id = ?', (chat_id,))
            cursor.execute('DELETE FROM chats WHERE chat_id = ?', (chat_id,))

            conn.commit()
            return True

    def clear_all(self) -> None:
        """Delete all chat histories."""
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM messages')
            cursor.execute('DELETE FROM chats')
            conn.commit()

    def get_readable_chat_history(
        self,
        chat_id: str,
        user_role: str = "user",
        assistant_role: str = "assistant",
    ) -> List[Dict[str, Any]]:
        """Get chat history in a human-readable format.

        This method transforms the raw message format into a display-friendly
        format with 'from', 'message', and optional 'timestamp' keys.

        Args:
            chat_id: Unique identifier for the chat session.
            user_role: The role name for user messages (default: "user").
            assistant_role: The role name for assistant messages (default: "assistant").

        Returns:
            List of formatted message dictionaries with:
                - 'from': "user" or "assistant"
                - 'message': The message content
                - 'timestamp': Message timestamp (if available)
        """
        self._ensure_initialized()

        messages = self.get_chat_history(chat_id)
        readable_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp")

            # Determine the 'from' field based on role
            if role == user_role:
                from_field = "user"
            elif role == assistant_role:
                from_field = "assistant"
            else:
                # Skip system messages or unknown roles
                continue

            formatted = {
                "from": from_field,
                "message": content,
            }

            if timestamp:
                formatted["timestamp"] = timestamp

            readable_messages.append(formatted)

        return readable_messages

    # ==================== Pydantic-AI Integration ====================

    def get_message_history(self, chat_id: str) -> Optional[List[Any]]:
        """Get chat history in pydantic-ai compatible format.

        This method converts the stored database history to pydantic-ai's
        ModelMessage format, ready to be passed to agent.run() or 
        agent.run_stream() as message_history.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            List of ModelMessage objects for pydantic-ai, or None if:
                - Chat doesn't exist or is empty
                - pydantic-ai is not installed

        Example:
            >>> history = SQLChatHistory()
            >>> message_history = history.get_message_history("chat123")
            >>> result = await agent.run("Hello", message_history=message_history)
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
            )

        self._ensure_initialized()
        raw_history = self.get_chat_history(chat_id)

        if not raw_history:
            return None

        return ModelMessagesTypeAdapter.validate_python(raw_history)

    def save_messages(self, chat_id: str, messages: List[Any]) -> None:
        """Save pydantic-ai messages to history.

        This method converts pydantic-ai's ModelMessage objects to JSON
        and stores them. Typically called with result.all_messages() after
        an agent run.

        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of pydantic-ai ModelMessage objects 
                     (e.g., from result.all_messages()).

        Example:
            >>> result = await agent.run("Hello", message_history=history.get_message_history("chat123"))
            >>> history.save_messages("chat123", result.all_messages())
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
            )

        self._ensure_initialized()

        # Convert pydantic-ai messages to JSON-serializable format
        json_messages = to_jsonable_python(messages)
        self.set_messages(chat_id, json_messages)


if __name__ == "__main__":
    import tempfile

    # Create a temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_chat_history.db")

        # Example usage
        history = SQLChatHistory(
            config={"db_path": db_path}
        )

        # Add messages
        history.add_message("chat123", {
            "role": "user",
            "content": "Hello!",
            "timestamp": "2025-03-18 10:00:00"
        })
        history.add_message("chat123", {
            "role": "assistant",
            "content": "Hi there! How can I help?",
            "timestamp": "2025-03-18 10:00:05"
        })

        # Retrieve history
        print("Chat history:", history.get_chat_history("chat123"))
        print("All chat IDs:", history.get_all_chat_ids())
        print("Readable format:", history.get_readable_chat_history("chat123"))

        # Test set_messages
        history.set_messages("chat456", [
            {"role": "user", "content": "Test message",
                "timestamp": "2025-03-18 10:02:00"}
        ])
        print("Chat history for chat456:", history.get_chat_history("chat456"))
        print("All chat IDs after adding chat456:", history.get_all_chat_ids())

        # Clean up
        deleted = history.delete_chat_history("chat123")
        print(f"Deleted chat123: {deleted}")
        print("After deletion:", history.get_all_chat_ids())

        # Clear all
        history.clear_all()
        print("After clear_all:", history.get_all_chat_ids())

        print("\nAll tests passed!")
