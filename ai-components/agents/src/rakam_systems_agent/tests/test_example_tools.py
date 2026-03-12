import pytest

from rakam_systems_agent.components.tools.example_tools import (
    get_current_weather,
    calculate_distance,
    format_currency,
    translate_text,
    analyze_sentiment,
    WebSearchTool,
    DatabaseQueryTool,
    FileProcessorTool,
    get_all_example_tools,
)


# ----------------------------
# Async function tools
# ----------------------------

@pytest.mark.asyncio
async def test_get_current_weather_celsius():
    result = await get_current_weather("London")
    assert result["location"] == "London"
    assert result["units"] == "celsius"
    assert result["temperature"] == 22
    assert "condition" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_get_current_weather_fahrenheit():
    result = await get_current_weather("New York", units="fahrenheit")
    assert result["temperature"] == 72
    assert result["units"] == "fahrenheit"


@pytest.mark.asyncio
async def test_calculate_distance():
    dist = await calculate_distance(0.0, 0.0, 1.0, 0.0)
    assert isinstance(dist, float)
    assert dist > 0


@pytest.mark.asyncio
async def test_calculate_distance_same_point():
    dist = await calculate_distance(10.0, 20.0, 10.0, 20.0)
    assert dist == 0.0


@pytest.mark.asyncio
async def test_translate_text_english():
    result = await translate_text("Hello", target_language="en")
    assert result["original"] == "Hello"
    assert result["translated"] == "Hello"
    assert result["target_language"] == "en"


@pytest.mark.asyncio
async def test_translate_text_spanish():
    result = await translate_text("Hello", target_language="es")
    assert "[ES]" in result["translated"]


@pytest.mark.asyncio
async def test_translate_text_unknown_language():
    result = await translate_text("Hello", target_language="xx")
    assert "[xx]" in result["translated"]


# ----------------------------
# Sync function tools
# ----------------------------

def test_format_currency_usd():
    result = format_currency(1000.5)
    assert "$" in result
    assert "1,000.50" in result


def test_format_currency_eur():
    result = format_currency(500.0, currency="EUR")
    assert "€" in result


def test_format_currency_unknown():
    result = format_currency(100.0, currency="CHF")
    assert "CHF" in result


def test_analyze_sentiment_positive():
    result = analyze_sentiment("This is great and amazing!")
    assert result["sentiment"] == "positive"
    assert result["score"] == 0.7
    assert result["positive_indicators"] > 0


def test_analyze_sentiment_negative():
    result = analyze_sentiment("This is terrible and awful!")
    assert result["sentiment"] == "negative"
    assert result["score"] == 0.3


def test_analyze_sentiment_neutral():
    result = analyze_sentiment("This is a test")
    assert result["sentiment"] == "neutral"
    assert result["score"] == 0.5


def test_analyze_sentiment_structure():
    result = analyze_sentiment("hello world")
    assert "text" in result
    assert "sentiment" in result
    assert "score" in result
    assert "positive_indicators" in result
    assert "negative_indicators" in result


# ----------------------------
# WebSearchTool Tests
# ----------------------------

def test_web_search_tool_init():
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "Search the web" in tool.description


def test_web_search_tool_run():
    tool = WebSearchTool()
    result = tool.run("python testing")
    assert result["query"] == "python testing"
    assert len(result["results"]) == 2
    assert result["total_results"] == 2


def test_web_search_tool_results_structure():
    tool = WebSearchTool()
    result = tool.run("test")
    for r in result["results"]:
        assert "title" in r
        assert "url" in r
        assert "snippet" in r


@pytest.mark.asyncio
async def test_web_search_tool_arun():
    tool = WebSearchTool()
    result = await tool.arun("async test")
    assert result["query"] == "async test"
    assert "results" in result


# ----------------------------
# DatabaseQueryTool Tests
# ----------------------------

def test_db_query_tool_init():
    tool = DatabaseQueryTool()
    assert tool.name == "database_query"


def test_db_query_users():
    tool = DatabaseQueryTool()
    result = tool.run("SELECT * FROM users")
    assert result["table"] == "users"
    assert len(result["results"]) > 0
    assert result["count"] > 0


def test_db_query_products():
    tool = DatabaseQueryTool()
    result = tool.run("SELECT * FROM products")
    assert result["table"] == "products"
    assert len(result["results"]) > 0


def test_db_query_unknown_table():
    tool = DatabaseQueryTool()
    result = tool.run("SELECT * FROM unknown_table")
    assert result["table"] == "unknown"
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_db_query_arun():
    tool = DatabaseQueryTool()
    result = await tool.arun("SELECT * FROM users")
    assert result["table"] == "users"


# ----------------------------
# FileProcessorTool Tests
# ----------------------------

def test_file_processor_init():
    tool = FileProcessorTool()
    assert tool.name == "file_processor"


def test_file_processor_run():
    tool = FileProcessorTool()
    result = tool.run("/path/to/file.txt")
    assert result["file"] == "/path/to/file.txt"
    assert result["status"] == "processed"
    assert result["lines"] == 42
    assert result["words"] == 256


@pytest.mark.asyncio
async def test_file_processor_arun():
    tool = FileProcessorTool()
    result = await tool.arun("file.txt")
    assert result["status"] == "processed"


# ----------------------------
# get_all_example_tools Tests
# ----------------------------

def test_get_all_example_tools_count():
    tools = get_all_example_tools()
    assert len(tools) == 5


def test_get_all_example_tools_structure():
    tools = get_all_example_tools()
    for tool in tools:
        assert "name" in tool
        assert "function" in tool
        assert "description" in tool
        assert "json_schema" in tool
        assert "category" in tool
        assert "tags" in tool


def test_get_all_example_tools_names():
    tools = get_all_example_tools()
    names = {t["name"] for t in tools}
    assert "get_current_weather" in names
    assert "calculate_distance" in names
    assert "format_currency" in names
    assert "translate_text" in names
    assert "analyze_sentiment" in names
