import pytest
from unittest.mock import MagicMock

from rakam_systems_core.interfaces.tool_registry import ToolRegistry, ToolMetadata, ToolMode
from rakam_systems_core.interfaces.tool import ToolComponent, FunctionToolComponent


@pytest.fixture
def registry():
    return ToolRegistry()


def make_tool(name="my_tool", description="A tool"):
    def fn(query: str) -> str:
        return f"result: {query}"

    return ToolComponent.from_function(
        function=fn,
        name=name,
        description=description,
        json_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )


def test_tool_mode_constants():
    assert ToolMode.DIRECT == "direct"
    assert ToolMode.MCP == "mcp"


def test_tool_metadata_defaults():
    meta = ToolMetadata(name="t", description="desc", mode=ToolMode.DIRECT)
    assert meta.category == "general"
    assert meta.tags == []
    assert meta.mcp_server is None
    assert meta.mcp_tool_name is None


def test_tool_metadata_to_dict():
    meta = ToolMetadata(
        name="search",
        description="search tool",
        mode=ToolMode.MCP,
        mcp_server="my_server",
        mcp_tool_name="do_search",
        category="retrieval",
        tags=["search", "web"],
    )
    d = meta.to_dict()
    assert d["name"] == "search"
    assert d["mode"] == ToolMode.MCP
    assert d["mcp_server"] == "my_server"
    assert d["category"] == "retrieval"
    assert "search" in d["tags"]


def test_registry_initial_state(registry):
    assert len(registry) == 0
    assert registry.get_all_tools() == []


def test_register_mcp_tool(registry):
    registry.register_mcp_tool(
        name="web_search",
        mcp_server="search_server",
        mcp_tool_name="search",
        description="Search the web",
        category="retrieval",
        tags=["search"],
    )
    assert "web_search" in registry
    assert len(registry) == 1


def test_register_mcp_tool_metadata(registry):
    registry.register_mcp_tool(
        name="web_search",
        mcp_server="search_server",
        mcp_tool_name="search",
        description="Search the web",
    )
    meta = registry.get_tool("web_search")
    assert meta.mode == ToolMode.MCP
    assert meta.mcp_server == "search_server"
    assert meta.mcp_tool_name == "search"


def test_register_tool_instance(registry):
    tool = make_tool("calc", "Calculator")
    registry.register_tool_instance(tool)
    assert "calc" in registry
    assert len(registry) == 1


def test_register_tool_instance_invalid_type(registry):
    with pytest.raises(ValueError, match="Unsupported tool type"):
        registry.register_tool_instance("not_a_tool")


def test_register_direct_tool(registry):
    def add(x: int, y: int) -> int:
        return x + y

    registry.register_direct_tool(
        name="add",
        function=add,
        description="Add two numbers",
        json_schema={"type": "object"},
    )
    assert "add" in registry


def test_register_duplicate_raises(registry):
    registry.register_mcp_tool("t", "srv", "t", "tool")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_mcp_tool("t", "srv2", "t2", "tool2")


def test_get_tool_returns_none_for_unknown(registry):
    assert registry.get_tool("nonexistent") is None


def test_get_all_tools(registry):
    registry.register_mcp_tool("t1", "s", "t1", "tool1")
    registry.register_mcp_tool("t2", "s", "t2", "tool2")
    tools = registry.get_all_tools()
    assert len(tools) == 2


def test_get_tools_by_category(registry):
    registry.register_mcp_tool("t1", "s", "t1", "tool1", category="cat_a")
    registry.register_mcp_tool("t2", "s", "t2", "tool2", category="cat_b")
    registry.register_mcp_tool("t3", "s", "t3", "tool3", category="cat_a")

    cat_a = registry.get_tools_by_category("cat_a")
    assert len(cat_a) == 2

    cat_b = registry.get_tools_by_category("cat_b")
    assert len(cat_b) == 1

    assert registry.get_tools_by_category("unknown") == []


def test_get_tools_by_tag(registry):
    registry.register_mcp_tool("t1", "s", "t1", "tool1", tags=["search", "web"])
    registry.register_mcp_tool("t2", "s", "t2", "tool2", tags=["search"])
    registry.register_mcp_tool("t3", "s", "t3", "tool3", tags=["db"])

    search_tools = registry.get_tools_by_tag("search")
    assert len(search_tools) == 2

    db_tools = registry.get_tools_by_tag("db")
    assert len(db_tools) == 1

    assert registry.get_tools_by_tag("nope") == []


def test_get_tools_by_mode(registry):
    registry.register_mcp_tool("mcp1", "s", "t1", "mcp tool")
    registry.register_direct_tool(
        "direct1",
        function=lambda q: q,
        description="direct",
        json_schema={"type": "object"},
    )

    mcp_tools = registry.get_tools_by_mode(ToolMode.MCP)
    direct_tools = registry.get_tools_by_mode(ToolMode.DIRECT)

    assert len(mcp_tools) == 1
    assert len(direct_tools) == 1
    assert mcp_tools[0].name == "mcp1"


def test_list_categories(registry):
    registry.register_mcp_tool("t1", "s", "t1", "t", category="cat_a")
    registry.register_mcp_tool("t2", "s", "t2", "t", category="cat_b")
    cats = registry.list_categories()
    assert "cat_a" in cats
    assert "cat_b" in cats


def test_list_tags(registry):
    registry.register_mcp_tool("t1", "s", "t1", "t", tags=["foo", "bar"])
    tags = registry.list_tags()
    assert "foo" in tags
    assert "bar" in tags


def test_unregister_tool(registry):
    registry.register_mcp_tool("t1", "s", "t1", "t", category="c", tags=["tag1"])
    assert "t1" in registry
    result = registry.unregister_tool("t1")
    assert result is True
    assert "t1" not in registry
    assert len(registry) == 0
    assert "c" not in registry.list_categories()
    assert "tag1" not in registry.list_tags()


def test_unregister_nonexistent_tool(registry):
    result = registry.unregister_tool("ghost")
    assert result is False


def test_clear(registry):
    registry.register_mcp_tool("t1", "s", "t1", "t1")
    registry.register_mcp_tool("t2", "s", "t2", "t2")
    registry.clear()
    assert len(registry) == 0
    assert registry.list_categories() == []
    assert registry.list_tags() == []


def test_contains_operator(registry):
    registry.register_mcp_tool("my_tool", "s", "t", "tool")
    assert "my_tool" in registry
    assert "other_tool" not in registry


def test_repr(registry):
    r = repr(registry)
    assert "ToolRegistry" in r
    assert "tools=0" in r


def test_function_tool_run():
    def add(x, y):
        return x + y

    tool = FunctionToolComponent(
        function=add,
        name="add",
        description="add",
        json_schema={"type": "object"},
    )
    assert tool.run(2, 3) == 5


def test_function_tool_is_not_async():
    def sync_fn(q):
        return q

    tool = FunctionToolComponent(
        function=sync_fn,
        name="sync",
        description="sync",
        json_schema={"type": "object"},
    )
    assert tool.is_async is False


def test_function_tool_is_async():
    async def async_fn(q):
        return q

    tool = FunctionToolComponent(
        function=async_fn,
        name="async",
        description="async",
        json_schema={"type": "object"},
    )
    assert tool.is_async is True


@pytest.mark.asyncio
async def test_function_tool_acall_sync():
    def fn(q):
        return f"result:{q}"

    tool = FunctionToolComponent(
        function=fn,
        name="t",
        description="t",
        json_schema={"type": "object"},
    )
    result = await tool.acall("hello")
    assert result == "result:hello"


@pytest.mark.asyncio
async def test_function_tool_acall_async():
    async def fn(q):
        return f"async:{q}"

    tool = FunctionToolComponent(
        function=fn,
        name="t",
        description="t",
        json_schema={"type": "object"},
    )
    result = await tool.acall("world")
    assert result == "async:world"
