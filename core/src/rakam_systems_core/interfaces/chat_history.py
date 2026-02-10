"""Chat History Interface.

This module defines the abstract interface for chat history management components.
Implementations can use different storage backends (JSON, database, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..base import BaseComponent


class ChatHistoryComponent(BaseComponent, ABC):
    """Abstract interface for chat history management.
    
    This component manages conversation history across multiple chat sessions,
    providing CRUD operations for messages and chat sessions.
    
    Implementations should handle:
    - Storing and retrieving messages by chat_id
    - Persistence of chat history (file, database, etc.)
    - Optional: message formatting for display
    """

    def __init__(
        self,
        name: str = "chat_history",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the chat history component.
        
        Args:
            name: Component name for identification.
            config: Configuration dictionary with storage-specific settings.
        """
        super().__init__(name, config)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Default run method - delegates to get_chat_history if chat_id provided."""
        if args and isinstance(args[0], str):
            return self.get_chat_history(args[0])
        raise ValueError("run() requires a chat_id as the first argument")

    @abstractmethod
    def add_message(self, chat_id: str, message: Dict[str, Any]) -> None:
        """Add a single message to a chat session.
        
        Args:
            chat_id: Unique identifier for the chat session.
            message: Message object containing content, role, timestamp, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def set_messages(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        """Set/replace all messages for a chat session.
        
        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of message objects to store.
        """
        raise NotImplementedError

    @abstractmethod
    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat session.
        
        Args:
            chat_id: Unique identifier for the chat session.
            
        Returns:
            List of message objects, or empty list if chat doesn't exist.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_chat_ids(self) -> List[str]:
        """Get all chat IDs currently stored.
        
        Returns:
            List of all chat session identifiers.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_chat_history(self, chat_id: str) -> bool:
        """Delete all messages for a chat session.
        
        Args:
            chat_id: Unique identifier for the chat session to delete.
            
        Returns:
            True if deletion was successful, False if chat_id didn't exist.
        """
        raise NotImplementedError

    def clear_all(self) -> None:
        """Delete all chat histories. Override for efficient implementation."""
        for chat_id in self.get_all_chat_ids():
            self.delete_chat_history(chat_id)

    def get_message_count(self, chat_id: str) -> int:
        """Get the number of messages in a chat session.
        
        Args:
            chat_id: Unique identifier for the chat session.
            
        Returns:
            Number of messages, or 0 if chat doesn't exist.
        """
        return len(self.get_chat_history(chat_id))

    def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat session exists.
        
        Args:
            chat_id: Unique identifier for the chat session.
            
        Returns:
            True if chat exists, False otherwise.
        """
        return chat_id in self.get_all_chat_ids()
