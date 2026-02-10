"""JSON-Based Chat History Manager.

This module provides a ChatHistoryComponent implementation that stores
chat history in a JSON file. Suitable for development, testing, and
single-instance deployments.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from rakam_systems_core.ai_core.interfaces.chat_history import \
    ChatHistoryComponent

# Optional pydantic-ai integration
try:
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
    from pydantic_core import to_jsonable_python
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    ModelMessagesTypeAdapter = None  # type: ignore
    ModelMessage = None  # type: ignore
    to_jsonable_python = None  # type: ignore


class JSONChatHistory(ChatHistoryComponent):
    """Chat history manager using JSON file storage.

    This implementation stores all chat histories in a single JSON file.
    It's suitable for:
    - Development and testing
    - Single-instance deployments
    - Small to medium scale applications

    For production with multiple instances, consider using a database-backed
    implementation instead.

    Config options:
        storage_path: Path to the JSON file (default: "./chat_history.json")
        auto_save: Whether to save after each modification (default: True)
        indent: JSON indentation for readability (default: 4)

    Example:
        >>> history = JSONChatHistory(config={"storage_path": "./data/chats.json"})
        >>> history.add_message("chat123", {"role": "user", "content": "Hello"})
        >>> history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})
        >>> messages = history.get_chat_history("chat123")
    """

    def __init__(
        self,
        name: str = "json_chat_history",
        config: Optional[Dict[str, Any]] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """Initialize the JSON chat history manager.

        Args:
            name: Component name for identification.
            config: Configuration dictionary. Supports:
                - storage_path: Path to JSON file
                - auto_save: Save after each modification (default: True)
                - indent: JSON indentation (default: 4)
            storage_path: Direct path override (takes precedence over config).
        """
        super().__init__(name, config)

        # Get storage path from argument, config, or default
        self.storage_path = storage_path or self.config.get(
            "storage_path", "./chat_history.json"
        )
        self.auto_save = self.config.get("auto_save", True)
        self.indent = self.config.get("indent", 4)

        # In-memory cache of chat history
        self._chat_history: Dict[str, List[Dict[str, Any]]] = {}

    def setup(self) -> None:
        """Initialize storage and load existing history."""
        self._initialize_storage()
        super().setup()

    def shutdown(self) -> None:
        """Save and cleanup resources."""
        self._save()
        super().shutdown()

    def _initialize_storage(self) -> None:
        """Initialize storage: create directory and load existing data."""
        # Ensure directory exists
        storage_dir = os.path.dirname(self.storage_path)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

        # Load existing data or create new file
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self._chat_history = json.loads(content)
                    else:
                        self._chat_history = {}
            except (json.JSONDecodeError, IOError):
                # Invalid JSON or IO error - start fresh
                self._chat_history = {}
                self._save()
        else:
            # Create new file
            self._chat_history = {}
            self._save()

    def _save(self) -> None:
        """Save current chat history to JSON file."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._chat_history, f,
                          indent=self.indent, ensure_ascii=False)
        except IOError as e:
            raise IOError(
                f"Failed to save chat history to {self.storage_path}: {e}")

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

        if chat_id not in self._chat_history:
            self._chat_history[chat_id] = []

        self._chat_history[chat_id].append(message)

        if self.auto_save:
            self._save()

    def set_messages(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        """Set/replace all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of message objects to store.
        """
        self._ensure_initialized()
        # Copy to avoid reference issues
        self._chat_history[chat_id] = list(messages)

        if self.auto_save:
            self._save()

    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            List of message objects, or empty list if chat doesn't exist.
        """
        self._ensure_initialized()
        return self._chat_history.get(chat_id, [])

    def get_all_chat_ids(self) -> List[str]:
        """Get all chat IDs currently stored.

        Returns:
            List of all chat session identifiers.
        """
        self._ensure_initialized()
        return list(self._chat_history.keys())

    def delete_chat_history(self, chat_id: str) -> bool:
        """Delete all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session to delete.

        Returns:
            True if deletion was successful, False if chat_id didn't exist.
        """
        self._ensure_initialized()

        if chat_id not in self._chat_history:
            return False

        del self._chat_history[chat_id]

        if self.auto_save:
            self._save()

        return True

    def clear_all(self) -> None:
        """Delete all chat histories."""
        self._ensure_initialized()
        self._chat_history = {}

        if self.auto_save:
            self._save()

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

    def save(self) -> None:
        """Manually save the current state to disk.

        Useful when auto_save is disabled.
        """
        self._ensure_initialized()
        self._save()

    def reload(self) -> None:
        """Reload chat history from disk, discarding in-memory changes."""
        self._initialize_storage()

    # ==================== Pydantic-AI Integration ====================

    def get_message_history(self, chat_id: str) -> Optional[List[Any]]:
        """Get chat history in pydantic-ai compatible format.

        This method converts the stored JSON history to pydantic-ai's
        ModelMessage format, ready to be passed to agent.run() or 
        agent.run_stream() as message_history.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            List of ModelMessage objects for pydantic-ai, or None if:
                - Chat doesn't exist or is empty
                - pydantic-ai is not installed

        Example:
            >>> history = JSONChatHistory()
            >>> message_history = history.get_message_history("chat123")
            >>> result = await agent.run("Hello", message_history=message_history)
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
            )

        self._ensure_initialized()
        raw_history = self._chat_history.get(chat_id, [])

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
        self._chat_history[chat_id] = json_messages

        if self.auto_save:
            self._save()


if __name__ == "__main__":
    # Example usage
    history = JSONChatHistory(
        config={"storage_path": "./test_chat_history.json"}
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

    # Clean up
    history.delete_chat_history("chat123")
    print("After deletion:", history.get_all_chat_ids())

    # Remove test file
    import os
    os.remove("./test_chat_history.json")
