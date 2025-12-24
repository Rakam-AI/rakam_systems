from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rakam_systems.core.ai_utils import logging

from ..base import BaseComponent

logger = logging.getLogger(__name__)


class MCPServer(BaseComponent):
    """
    Model Context Protocol (MCP) Server.

    A message-based component registry that routes messages between components.
    Supports both synchronous and asynchronous operations.

    Features:
    - Component registration and discovery
    - Message routing with sender/receiver pattern
    - Support for both sync and async message handlers
    - Automatic argument extraction from messages
    - Error handling and logging

    Example:
        >>> server = MCPServer(name="my_mcp_server")
        >>> server.setup()
        >>> 
        >>> # Register components
        >>> search_tool = SearchTool(name="search")
        >>> server.register_component(search_tool)
        >>> 
        >>> # Send messages
        >>> result = server.send_message(
        ...     sender="client",
        ...     receiver="search",
        ...     message={'arguments': {'query': 'test'}}
        ... )
        >>> 
        >>> # Async version
        >>> result = await server.asend_message(
        ...     sender="client",
        ...     receiver="search",
        ...     message={'arguments': {'query': 'test'}}
        ... )
    """

    def __init__(
        self,
        name: str = "mcp_server",
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True
    ) -> None:
        """
        Initialize MCP Server.

        Args:
            name: Name of the MCP server
            config: Optional configuration dictionary
            enable_logging: Whether to enable detailed logging
        """
        super().__init__(name, config)
        self._registry: Dict[str, BaseComponent] = {}
        self._enable_logging = enable_logging

        if self._enable_logging:
            logger.debug(f"Initialized MCPServer: {name}")

    def register_component(self, component: BaseComponent) -> None:
        """
        Register a component with the MCP server.

        Args:
            component: Component to register

        Raises:
            ValueError: If component with same name already registered
        """
        if component.name in self._registry:
            logger.warning(
                f"Component '{component.name}' already registered, overwriting"
            )

        self._registry[component.name] = component

        if self._enable_logging:
            logger.debug(f"Registered component: {component.name}")

    def unregister_component(self, component_name: str) -> bool:
        """
        Unregister a component from the MCP server.

        Args:
            component_name: Name of component to unregister

        Returns:
            True if component was unregistered, False if not found
        """
        if component_name in self._registry:
            del self._registry[component_name]
            if self._enable_logging:
                logger.debug(f"Unregistered component: {component_name}")
            return True
        return False

    def get_component(self, component_name: str) -> Optional[BaseComponent]:
        """
        Get a registered component by name.

        Args:
            component_name: Name of the component

        Returns:
            The component if found, None otherwise
        """
        return self._registry.get(component_name)

    def list_components(self) -> List[str]:
        """
        List all registered component names.

        Returns:
            Sorted list of component names
        """
        return sorted(self._registry.keys())

    def has_component(self, component_name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            component_name: Name of the component

        Returns:
            True if component is registered, False otherwise
        """
        return component_name in self._registry

    def send_message(
        self,
        sender: str,
        receiver: str,
        message: Dict[str, Any]
    ) -> Any:
        """
        Send a message to a registered component (synchronous).

        Args:
            sender: Name of the message sender
            receiver: Name of the target component
            message: Message dictionary containing action and arguments

        Returns:
            Result from the target component

        Raises:
            KeyError: If receiver component is not registered
            Exception: Any exception raised by the component

        Example:
            >>> result = server.send_message(
            ...     sender="client",
            ...     receiver="search_tool",
            ...     message={
            ...         'action': 'invoke_tool',
            ...         'arguments': {'query': 'test', 'top_k': 5}
            ...     }
            ... )
        """
        if receiver not in self._registry:
            error_msg = f"Receiver '{receiver}' not registered. Available: {self.list_components()}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        target = self._registry[receiver]

        if self._enable_logging:
            logger.debug(
                f"Routing message from '{sender}' to '{receiver}': "
                f"{message.get('action', 'unknown')}"
            )

        try:
            # Check for custom message handler
            handler = getattr(target, "handle_message", None)
            if callable(handler):
                return handler(sender=sender, message=message)

            # Default: extract arguments and call run()
            if isinstance(message, dict) and 'arguments' in message:
                arguments = message['arguments']
                if isinstance(arguments, dict):
                    return target.run(**arguments)
                else:
                    return target.run(arguments)

            return target.run(message)

        except Exception as e:
            logger.error(
                f"Error routing message to '{receiver}': {str(e)}",
                exc_info=True
            )
            raise

    async def asend_message(
        self,
        sender: str,
        receiver: str,
        message: Dict[str, Any]
    ) -> Any:
        """
        Send a message to a registered component (asynchronous).

        Supports both async and sync components. If the component has
        async handlers, they will be awaited. Otherwise, sync handlers
        will be executed in the event loop.

        Args:
            sender: Name of the message sender
            receiver: Name of the target component
            message: Message dictionary containing action and arguments

        Returns:
            Result from the target component

        Raises:
            KeyError: If receiver component is not registered
            Exception: Any exception raised by the component

        Example:
            >>> result = await server.asend_message(
            ...     sender="client",
            ...     receiver="search_tool",
            ...     message={
            ...         'action': 'invoke_tool',
            ...         'arguments': {'query': 'test', 'top_k': 5}
            ...     }
            ... )
        """
        if receiver not in self._registry:
            error_msg = f"Receiver '{receiver}' not registered. Available: {self.list_components()}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        target = self._registry[receiver]

        if self._enable_logging:
            logger.debug(
                f"Routing async message from '{sender}' to '{receiver}': "
                f"{message.get('action', 'unknown')}"
            )

        try:
            # Check for async message handler
            handler = getattr(target, "handle_message", None)
            if callable(handler):
                if asyncio.iscoroutinefunction(handler):
                    return await handler(sender=sender, message=message)
                else:
                    return handler(sender=sender, message=message)

            # Check for async run method
            run_method = getattr(target, "run", None)
            if run_method is None:
                raise AttributeError(
                    f"Component '{receiver}' has no 'run' method")

            # Extract arguments
            if isinstance(message, dict) and 'arguments' in message:
                arguments = message['arguments']

                # Call async or sync run method
                if asyncio.iscoroutinefunction(run_method):
                    if isinstance(arguments, dict):
                        return await run_method(**arguments)
                    else:
                        return await run_method(arguments)
                else:
                    # Sync method - call directly
                    if isinstance(arguments, dict):
                        return run_method(**arguments)
                    else:
                        return run_method(arguments)

            # No arguments, just call run
            if asyncio.iscoroutinefunction(run_method):
                return await run_method(message)
            else:
                return run_method(message)

        except Exception as e:
            logger.error(
                f"Error routing async message to '{receiver}': {str(e)}",
                exc_info=True
            )
            raise

    def run(self, *args, **kwargs) -> List[str]:
        """
        Return list of registered components.

        This is the BaseComponent interface implementation.

        Returns:
            Sorted list of component names
        """
        return self.list_components()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Dictionary with server stats
        """
        return {
            'name': self.name,
            'component_count': len(self._registry),
            'components': self.list_components(),
            'logging_enabled': self._enable_logging
        }

    def __repr__(self) -> str:
        return f"MCPServer(name='{self.name}', components={len(self._registry)})"

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._registry)

    def __contains__(self, component_name: str) -> bool:
        """Check if component is registered."""
        return component_name in self._registry
