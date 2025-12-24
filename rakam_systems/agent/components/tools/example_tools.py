"""
Example tools demonstrating both direct and MCP invocation patterns.
These tools can be used for testing and as templates for custom tools.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from rakam_systems.core.ai_core.interfaces.tool import ToolComponent


# === Simple Direct Tools ===

async def get_current_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather for a location (mock implementation).

    Args:
        location: City name or location
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Dictionary with weather information
    """
    await asyncio.sleep(0.5)  # Simulate API call
    return {
        "location": location,
        "temperature": 22 if units == "celsius" else 72,
        "units": units,
        "condition": "sunny",
        "humidity": 45,
        "timestamp": datetime.now().isoformat(),
    }


async def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two geographic coordinates (mock implementation).

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point

    Returns:
        Distance in kilometers
    """
    await asyncio.sleep(0.3)
    # Simplified calculation (not accurate, just for demo)
    import math
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance = math.sqrt(dlat**2 + dlon**2) * 111  # Rough km conversion
    return round(distance, 2)


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a number as currency.

    Args:
        amount: Amount to format
        currency: Currency code (USD, EUR, GBP, etc.)

    Returns:
        Formatted currency string
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"


async def translate_text(text: str, target_language: str = "en") -> Dict[str, str]:
    """
    Translate text to target language (mock implementation).

    Args:
        text: Text to translate
        target_language: Target language code (en, es, fr, de, etc.)

    Returns:
        Dictionary with original and translated text
    """
    await asyncio.sleep(0.7)
    # Mock translation - just returns the same text with a note
    translations = {
        "en": text,
        "es": f"[ES] {text}",
        "fr": f"[FR] {text}",
        "de": f"[DE] {text}",
    }
    return {
        "original": text,
        "translated": translations.get(target_language, f"[{target_language}] {text}"),
        "target_language": target_language,
    }


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text (mock implementation).

    Args:
        text: Text to analyze

    Returns:
        Dictionary with sentiment analysis results
    """
    # Very simple mock sentiment analysis
    positive_words = ["good", "great", "excellent",
                      "amazing", "wonderful", "happy"]
    negative_words = ["bad", "terrible", "awful", "horrible", "sad", "angry"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        sentiment = "positive"
        score = 0.7
    elif negative_count > positive_count:
        sentiment = "negative"
        score = 0.3
    else:
        sentiment = "neutral"
        score = 0.5

    return {
        "text": text,
        "sentiment": sentiment,
        "score": score,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
    }


# === Tool Components ===

class WebSearchTool(ToolComponent):
    """
    Web search tool component (mock implementation).
    Can be used directly or exposed via MCP.
    """

    def __init__(self, name: str = "web_search", config: Optional[Dict] = None):
        super().__init__(
            name=name,
            config=config,
            description="Search the web for information",
            json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        )

    def run(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search (mock implementation).

        Args:
            query: Search query

        Returns:
            Dictionary with search results
        """
        # Mock search results
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result 1 for '{query}'",
                    "url": "https://example.com/1",
                    "snippet": f"This is a mock result for the query: {query}",
                },
                {
                    "title": f"Result 2 for '{query}'",
                    "url": "https://example.com/2",
                    "snippet": f"Another mock result about {query}",
                },
            ],
            "total_results": 2,
        }

    async def arun(self, query: str) -> Dict[str, Any]:
        """Async version of run."""
        await asyncio.sleep(0.8)  # Simulate network delay
        return self.run(query)


class DatabaseQueryTool(ToolComponent):
    """
    Database query tool component (mock implementation).
    Demonstrates a tool that could be exposed via MCP for security.
    """

    def __init__(self, name: str = "database_query", config: Optional[Dict] = None):
        super().__init__(
            name=name,
            config=config,
            description="Query the database",
            json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Database query string"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        )
        self._mock_db = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            "products": [
                {"id": 1, "name": "Widget", "price": 19.99},
                {"id": 2, "name": "Gadget", "price": 29.99},
            ],
        }

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute a database query (mock implementation).

        Args:
            query: Query string (simplified, e.g., "SELECT * FROM users")

        Returns:
            Dictionary with query results
        """
        # Very simple mock query parser
        query_lower = query.lower()

        if "users" in query_lower:
            table = "users"
            results = self._mock_db["users"]
        elif "products" in query_lower:
            table = "products"
            results = self._mock_db["products"]
        else:
            table = "unknown"
            results = []

        return {
            "query": query,
            "table": table,
            "results": results,
            "count": len(results),
        }

    async def arun(self, query: str) -> Dict[str, Any]:
        """Async version of run."""
        await asyncio.sleep(0.4)  # Simulate database query time
        return self.run(query)


class FileProcessorTool(ToolComponent):
    """
    File processor tool component (mock implementation).
    Demonstrates a tool that processes files.
    """

    def __init__(self, name: str = "file_processor", config: Optional[Dict] = None):
        super().__init__(
            name=name,
            config=config,
            description="Process and analyze files",
            json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "File path or processing command"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        )

    def run(self, query: str) -> Dict[str, Any]:
        """
        Process a file (mock implementation).

        Args:
            query: File path or processing command

        Returns:
            Dictionary with processing results
        """
        return {
            "file": query,
            "status": "processed",
            "lines": 42,
            "words": 256,
            "characters": 1543,
            "type": "text/plain",
        }

    async def arun(self, query: str) -> Dict[str, Any]:
        """Async version of run."""
        await asyncio.sleep(0.6)  # Simulate file processing
        return self.run(query)


# === Helper Functions for Tool Registration ===

def get_all_example_tools() -> List[Dict[str, Any]]:
    """
    Get configuration for all example tools.

    Returns:
        List of tool configuration dictionaries
    """
    return [
        {
            "name": "get_current_weather",
            "function": get_current_weather,
            "description": "Get the current weather for a location",
            "json_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units (celsius or fahrenheit)",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "category": "utility",
            "tags": ["weather", "external"],
        },
        {
            "name": "calculate_distance",
            "function": calculate_distance,
            "description": "Calculate distance between two geographic coordinates",
            "json_schema": {
                "type": "object",
                "properties": {
                    "lat1": {"type": "number", "description": "Latitude of first point"},
                    "lon1": {"type": "number", "description": "Longitude of first point"},
                    "lat2": {"type": "number", "description": "Latitude of second point"},
                    "lon2": {"type": "number", "description": "Longitude of second point"},
                },
                "required": ["lat1", "lon1", "lat2", "lon2"],
                "additionalProperties": False,
            },
            "category": "math",
            "tags": ["geography", "calculation"],
        },
        {
            "name": "format_currency",
            "function": format_currency,
            "description": "Format a number as currency",
            "json_schema": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to format"},
                    "currency": {
                        "type": "string",
                        "description": "Currency code",
                        "enum": ["USD", "EUR", "GBP", "JPY"],
                        "default": "USD"
                    },
                },
                "required": ["amount"],
                "additionalProperties": False,
            },
            "category": "utility",
            "tags": ["formatting", "currency"],
        },
        {
            "name": "translate_text",
            "function": translate_text,
            "description": "Translate text to target language",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {
                        "type": "string",
                        "description": "Target language code",
                        "enum": ["en", "es", "fr", "de"],
                        "default": "en"
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "category": "nlp",
            "tags": ["translation", "language"],
        },
        {
            "name": "analyze_sentiment",
            "function": analyze_sentiment,
            "description": "Analyze sentiment of text",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "category": "nlp",
            "tags": ["sentiment", "analysis"],
        },
    ]
