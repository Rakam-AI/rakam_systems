"""PostgreSQL-Based Chat History Manager.

This module provides a ChatHistoryComponent implementation that stores
chat history in a PostgreSQL database. Suitable for production deployments
requiring persistent, structured storage with multi-instance support.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from psycopg2.pool import SimpleConnectionPool

from rakam_systems.core.ai_core.interfaces.chat_history import \
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


class PostgresChatHistory(ChatHistoryComponent):
    """Chat history manager using PostgreSQL database storage.

    This implementation stores all chat histories in a PostgreSQL database.
    It's suitable for:
    - Production deployments
    - Multi-instance applications with concurrent access
    - Applications requiring structured queries and scalability
    - Large scale applications

    Config options:
        host: PostgreSQL host (default: "localhost")
        port: PostgreSQL port (default: 5432)
        database: Database name (default: "vectorstore_db")
        user: Database user (default: "postgres")
        password: Database password (default: "postgres")
        schema: Schema name for chat tables (default: "public")
        min_connections: Minimum pool connections (default: 1)
        max_connections: Maximum pool connections (default: 10)

    Environment variables (override config):
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

    Example:
        >>> history = PostgresChatHistory(
        ...     config={
        ...         "host": "localhost",
        ...         "database": "vectorstore_db",
        ...         "user": "postgres",
        ...         "password": "postgres"
        ...     }
        ... )
        >>> history.setup()
        >>> history.add_message("chat123", {"role": "user", "content": "Hello"})
        >>> history.add_message("chat123", {"role": "assistant", "content": "Hi there!"})
        >>> messages = history.get_chat_history("chat123")
    """

    def __init__(
        self,
        name: str = "postgres_chat_history",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Initialize the PostgreSQL chat history manager.

        Args:
            name: Component name for identification.
            config: Configuration dictionary. Supports:
                - host: PostgreSQL host
                - port: PostgreSQL port
                - database: Database name
                - user: Database user
                - password: Database password
                - schema: Schema name (default: "public")
                - min_connections: Min pool size (default: 1)
                - max_connections: Max pool size (default: 10)
            **kwargs: Direct parameter overrides (host, port, database, user, password, schema).
        """
        super().__init__(name, config)

        # Get connection parameters from kwargs, config, or environment
        self.host = kwargs.get('host') or self.config.get(
            'host') or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = kwargs.get('port') or self.config.get(
            'port') or int(os.getenv('POSTGRES_PORT', '5432'))
        self.database = kwargs.get('database') or self.config.get(
            'database') or os.getenv('POSTGRES_DB', 'vectorstore_db')
        self.user = kwargs.get('user') or self.config.get(
            'user') or os.getenv('POSTGRES_USER', 'postgres')
        self.password = kwargs.get('password') or self.config.get(
            'password') or os.getenv('POSTGRES_PASSWORD', 'postgres')
        self.schema = kwargs.get(
            'schema') or self.config.get('schema', 'public')

        # Connection pool settings
        self.min_connections = self.config.get('min_connections', 1)
        self.max_connections = self.config.get('max_connections', 10)

        self._pool: Optional[SimpleConnectionPool] = None

    def setup(self) -> None:
        """Initialize database connection pool and create tables."""
        self._initialize_pool()
        self._initialize_database()
        super().setup()

    def shutdown(self) -> None:
        """Cleanup resources and close connection pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
        super().shutdown()

    def _initialize_pool(self) -> None:
        """Initialize PostgreSQL connection pool.

        Raises:
            Exception: If connection pool initialization fails.
        """
        try:
            self._pool = SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        except Exception as e:
            raise Exception(
                f"Failed to initialize PostgreSQL connection pool: {e}")

    def _get_connection(self):
        """Get a connection from the pool.

        Returns:
            psycopg2 connection object.

        Raises:
            Exception: If pool is not initialized or connection fails.
        """
        if not self._pool:
            raise Exception(
                "Connection pool not initialized. Call setup() first.")
        return self._pool.getconn()

    def _return_connection(self, conn) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection to return.
        """
        if self._pool:
            self._pool.putconn(conn)

    def _initialize_database(self) -> None:
        """Initialize PostgreSQL database and create necessary tables.

        Creates the chat_sessions and chat_messages tables if they don't exist.
        Uses a separate schema if specified.

        Raises:
            Exception: If database initialization fails.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create schema if not exists (if not using public)
            if self.schema != 'public':
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {self.schema}')

            # Create chat_sessions table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.schema}.chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create chat_messages table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.schema}.chat_messages (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    message_order INTEGER NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES {self.schema}.chat_sessions (chat_id) ON DELETE CASCADE
                )
            ''')

            # Create indexes for faster lookups
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id 
                ON {self.schema}.chat_messages (chat_id, message_order)
            ''')

            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_chat_messages_data 
                ON {self.schema}.chat_messages USING GIN (message_data)
            ''')

            # Create trigger to update updated_at timestamp
            cursor.execute(f'''
                CREATE OR REPLACE FUNCTION {self.schema}.update_chat_session_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    UPDATE {self.schema}.chat_sessions 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE chat_id = NEW.chat_id;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            ''')

            cursor.execute(f'''
                DROP TRIGGER IF EXISTS trigger_update_chat_session 
                ON {self.schema}.chat_messages
            ''')

            cursor.execute(f'''
                CREATE TRIGGER trigger_update_chat_session
                AFTER INSERT ON {self.schema}.chat_messages
                FOR EACH ROW
                EXECUTE FUNCTION {self.schema}.update_chat_session_timestamp()
            ''')

            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to initialize database: {e}")
        finally:
            if conn:
                self._return_connection(conn)

    def _ensure_initialized(self) -> None:
        """Ensure the component is initialized before operations."""
        if not self.initialized:
            self.setup()

    def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat session exists.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            True if chat exists, False otherwise.
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT 1 FROM {self.schema}.chat_sessions WHERE chat_id = %s',
                (chat_id,)
            )
            return cursor.fetchone() is not None
        finally:
            if conn:
                self._return_connection(conn)

    def add_message(self, chat_id: str, message: Dict[str, Any]) -> None:
        """Add a single message to a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            message: Message object (dict with role, content, timestamp, etc.).
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Ensure chat session exists
            cursor.execute(
                f'INSERT INTO {self.schema}.chat_sessions (chat_id) VALUES (%s) ON CONFLICT (chat_id) DO NOTHING',
                (chat_id,)
            )

            # Get next message order
            cursor.execute(
                f'SELECT COALESCE(MAX(message_order), -1) + 1 FROM {self.schema}.chat_messages WHERE chat_id = %s',
                (chat_id,)
            )
            next_order = cursor.fetchone()[0]

            # Insert message
            message_json = json.dumps(message, ensure_ascii=False)
            cursor.execute(
                f'''
                INSERT INTO {self.schema}.chat_messages (chat_id, message_order, message_data)
                VALUES (%s, %s, %s::jsonb)
                ''',
                (chat_id, next_order, message_json)
            )

            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to add message: {e}")
        finally:
            if conn:
                self._return_connection(conn)

    def set_messages(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        """Set/replace all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of message objects to store.
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Ensure chat session exists
            cursor.execute(
                f'INSERT INTO {self.schema}.chat_sessions (chat_id) VALUES (%s) ON CONFLICT (chat_id) DO NOTHING',
                (chat_id,)
            )

            # Delete existing messages
            cursor.execute(
                f'DELETE FROM {self.schema}.chat_messages WHERE chat_id = %s',
                (chat_id,)
            )

            # Insert new messages with order
            for order, message in enumerate(messages):
                message_json = json.dumps(message, ensure_ascii=False)
                cursor.execute(
                    f'''
                    INSERT INTO {self.schema}.chat_messages (chat_id, message_order, message_data)
                    VALUES (%s, %s, %s::jsonb)
                    ''',
                    (chat_id, order, message_json)
                )

            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to set messages: {e}")
        finally:
            if conn:
                self._return_connection(conn)

    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            List of message objects, or empty list if chat doesn't exist.
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f'''
                SELECT message_data
                FROM {self.schema}.chat_messages
                WHERE chat_id = %s
                ORDER BY message_order ASC
                ''',
                (chat_id,)
            )
            rows = cursor.fetchall()

            return [row[0] for row in rows]

        finally:
            if conn:
                self._return_connection(conn)

    def get_all_chat_ids(self) -> List[str]:
        """Get all chat IDs currently stored.

        Returns:
            List of all chat session identifiers.
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT chat_id FROM {self.schema}.chat_sessions ORDER BY updated_at DESC')
            return [row[0] for row in cursor.fetchall()]
        finally:
            if conn:
                self._return_connection(conn)

    def delete_chat_history(self, chat_id: str) -> bool:
        """Delete all messages for a chat session.

        Args:
            chat_id: Unique identifier for the chat session to delete.

        Returns:
            True if deletion was successful, False if chat_id didn't exist.
        """
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if chat exists
            cursor.execute(
                f'SELECT 1 FROM {self.schema}.chat_sessions WHERE chat_id = %s',
                (chat_id,)
            )
            exists = cursor.fetchone() is not None

            if not exists:
                return False

            # Delete messages (cascades from foreign key)
            cursor.execute(
                f'DELETE FROM {self.schema}.chat_messages WHERE chat_id = %s',
                (chat_id,)
            )
            cursor.execute(
                f'DELETE FROM {self.schema}.chat_sessions WHERE chat_id = %s',
                (chat_id,)
            )

            conn.commit()
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to delete chat history: {e}")
        finally:
            if conn:
                self._return_connection(conn)

    def clear_all(self) -> None:
        """Delete all chat histories."""
        self._ensure_initialized()

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {self.schema}.chat_messages')
            cursor.execute(f'DELETE FROM {self.schema}.chat_sessions')
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to clear all chats: {e}")
        finally:
            if conn:
                self._return_connection(conn)

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
            >>> history = PostgresChatHistory()
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
    # Example usage
    print("PostgresChatHistory Example")
    print("=" * 50)

    # Initialize with default settings (uses environment variables)
    history = PostgresChatHistory(
        config={
            "host": "localhost",
            "database": "vectorstore_db",
            "user": "postgres",
            "password": "postgres"
        }
    )

    try:
        history.setup()
        print("✓ Connected to PostgreSQL")

        # Add messages
        history.add_message("chat123", {
            "role": "user",
            "content": "Hello!",
            "timestamp": "2025-12-16 10:00:00"
        })
        history.add_message("chat123", {
            "role": "assistant",
            "content": "Hi there! How can I help?",
            "timestamp": "2025-12-16 10:00:05"
        })

        # Retrieve history
        print("\nChat history:", history.get_chat_history("chat123"))
        print("\nAll chat IDs:", history.get_all_chat_ids())
        print("\nReadable format:", history.get_readable_chat_history("chat123"))

        # Test set_messages
        history.set_messages("chat456", [
            {"role": "user", "content": "Test message",
                "timestamp": "2025-12-16 10:02:00"}
        ])
        print("\nChat history for chat456:",
              history.get_chat_history("chat456"))
        print("All chat IDs after adding chat456:", history.get_all_chat_ids())

        # Clean up
        deleted = history.delete_chat_history("chat123")
        print(f"\nDeleted chat123: {deleted}")
        print("After deletion:", history.get_all_chat_ids())

        # Clear all
        history.clear_all()
        print("After clear_all:", history.get_all_chat_ids())

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker-compose up -d vectordb")
    finally:
        history.shutdown()
