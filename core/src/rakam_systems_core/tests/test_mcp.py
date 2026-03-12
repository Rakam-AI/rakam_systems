import asyncio

import pytest
from rakam_systems_core.base import BaseComponent
from rakam_systems_core.mcp.mcp_server import MCPServer


class SyncComponent(BaseComponent):
    def run(self, x=None):
        return f"sync:{x}"


class SyncHandlerComponent(BaseComponent):
    def run(self, *args, **kwargs):
        return "should_not_be_called"

    def handle_message(self, sender, message):
        return f"handled_by:{sender}"


class AsyncComponent(BaseComponent):
    async def run(self, x=None):
        return f"async:{x}"


class AsyncHandlerComponent(BaseComponent):
    async def run(self, *args, **kwargs):
        return "should_not_be_called"

    async def handle_message(self, sender, message):
        return f"async_handled:{sender}"


class ErrorComponent(BaseComponent):
    def run(self, *args, **kwargs):
        raise ValueError("boom")



def test_register_and_get_component():
    server = MCPServer(enable_logging=False)
    comp = SyncComponent("comp")

    server.register_component(comp)

    assert server.get_component("comp") is comp
    assert server.has_component("comp")
    assert "comp" in server
    assert len(server) == 1


def test_unregister_component():
    server = MCPServer(enable_logging=False)
    comp = SyncComponent("comp")

    server.register_component(comp)

    assert server.unregister_component("comp") is True
    assert server.get_component("comp") is None
    assert len(server) == 0

    # Unregister missing
    assert server.unregister_component("missing") is False


def test_list_components_sorted():
    server = MCPServer(enable_logging=False)

    server.register_component(SyncComponent("b"))
    server.register_component(SyncComponent("a"))

    assert server.list_components() == ["a", "b"]



def test_send_message_with_arguments_dict():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("tool"))

    result = server.send_message(
        sender="client",
        receiver="tool",
        message={"arguments": {"x": 5}},
    )

    assert result == "sync:5"


def test_send_message_with_arguments_non_dict():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("tool"))

    result = server.send_message(
        sender="client",
        receiver="tool",
        message={"arguments": 10},
    )

    assert result == "sync:10"


def test_send_message_without_arguments():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("tool"))

    result = server.send_message(
        sender="client",
        receiver="tool",
        message="hello",
    )

    assert result == "sync:hello"


def test_send_message_with_handler():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncHandlerComponent("tool"))

    result = server.send_message(
        sender="client",
        receiver="tool",
        message={"arguments": {"x": 1}},
    )

    assert result == "handled_by:client"


def test_send_message_missing_receiver():
    server = MCPServer(enable_logging=False)

    with pytest.raises(KeyError):
        server.send_message("client", "missing", {})


def test_send_message_propagates_exception():
    server = MCPServer(enable_logging=False)
    server.register_component(ErrorComponent("tool"))

    with pytest.raises(ValueError):
        server.send_message("client", "tool", {})



@pytest.mark.asyncio
async def test_async_run_component():
    server = MCPServer(enable_logging=False)
    server.register_component(AsyncComponent("tool"))

    result = await server.asend_message(
        sender="client",
        receiver="tool",
        message={"arguments": {"x": 7}},
    )

    assert result == "async:7"


@pytest.mark.asyncio
async def test_async_handler_component():
    server = MCPServer(enable_logging=False)
    server.register_component(AsyncHandlerComponent("tool"))

    result = await server.asend_message(
        sender="client",
        receiver="tool",
        message={},
    )

    assert result == "async_handled:client"


@pytest.mark.asyncio
async def test_async_sync_component():
    """Ensure sync component works in async mode."""
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("tool"))

    result = await server.asend_message(
        sender="client",
        receiver="tool",
        message={"arguments": {"x": 3}},
    )

    assert result == "sync:3"


@pytest.mark.asyncio
async def test_async_missing_receiver():
    server = MCPServer(enable_logging=False)

    with pytest.raises(KeyError):
        await server.asend_message("client", "missing", {})


@pytest.mark.asyncio
async def test_async_exception_propagation():
    server = MCPServer(enable_logging=False)
    server.register_component(ErrorComponent("tool"))

    with pytest.raises(ValueError):
        await server.asend_message("client", "tool", {})



def test_run_returns_component_list():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("a"))
    server.register_component(SyncComponent("b"))

    result = server.run()
    assert result == ["a", "b"]


def test_get_stats():
    server = MCPServer(enable_logging=False)
    server.register_component(SyncComponent("a"))

    stats = server.get_stats()

    assert stats["name"] == server.name
    assert stats["component_count"] == 1
    assert stats["components"] == ["a"]
    assert stats["logging_enabled"] is False


def test_repr():
    server = MCPServer(name="test_server", enable_logging=False)
    server.register_component(SyncComponent("x"))

    rep = repr(server)

    assert "test_server" in rep
    assert "components=1" in rep
