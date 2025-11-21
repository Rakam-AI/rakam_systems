"""
LLM Gateway tools for calling generation functions through the tool system.

These tools allow agents to use LLM generation capabilities as tools,
enabling meta-reasoning, delegation, and multi-model workflows.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

from ai_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse
from ai_agents.components.llm_gateway import get_llm_gateway


# === LLM Gateway Tool Functions ===

async def llm_generate(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate text using an LLM through the gateway.
    
    This tool allows agents to call LLM generation as a tool, enabling:
    - Multi-step reasoning
    - Delegation to specialized models
    - Meta-reasoning workflows
    
    Args:
        user_prompt: The main prompt/question for the LLM
        system_prompt: Optional system prompt to set context/behavior
        model: Optional model string (e.g., "openai:gpt-4o", "mistral:mistral-large-latest")
        temperature: Optional temperature for generation (0.0-1.0)
        max_tokens: Optional maximum tokens to generate
    
    Returns:
        Dictionary containing:
        - content: The generated text
        - model: Model used for generation
        - usage: Token usage information
        - finish_reason: Why generation stopped
    """
    # Create gateway with specified model or use default
    gateway = get_llm_gateway(model=model, temperature=temperature)
    
    # Create request
    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Generate response
    response = gateway.generate(request)
    
    # Return structured result
    return {
        "content": response.content,
        "model": response.model,
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "metadata": response.metadata,
    }


async def llm_generate_structured(
    user_prompt: str,
    schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate structured output using an LLM through the gateway.
    
    This tool allows agents to request structured data from LLMs,
    ensuring responses conform to a specific schema.
    
    Args:
        user_prompt: The main prompt/question for the LLM
        schema: JSON schema defining the expected output structure
        system_prompt: Optional system prompt to set context/behavior
        model: Optional model string (e.g., "openai:gpt-4o")
        temperature: Optional temperature for generation (0.0-1.0)
        max_tokens: Optional maximum tokens to generate
    
    Returns:
        Dictionary containing:
        - structured_output: The parsed structured output
        - raw_content: The raw text response
        - model: Model used for generation
        - usage: Token usage information
    """
    # Create gateway with specified model or use default
    gateway = get_llm_gateway(model=model, temperature=temperature)
    
    # Create a dynamic Pydantic model from the schema
    # For now, we'll use JSON mode and parse the response
    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format="json",
    )
    
    # Generate response
    response = gateway.generate(request)
    
    # Try to parse as JSON
    try:
        structured_output = json.loads(response.content)
    except json.JSONDecodeError:
        structured_output = {"error": "Failed to parse structured output", "raw": response.content}
    
    # Return structured result
    return {
        "structured_output": structured_output,
        "raw_content": response.content,
        "model": response.model,
        "usage": response.usage,
    }


async def llm_count_tokens(
    text: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Count tokens in text using the LLM gateway's tokenizer.
    
    Useful for:
    - Checking prompt lengths before generation
    - Estimating costs
    - Managing context windows
    
    Args:
        text: Text to count tokens for
        model: Optional model string to use for tokenization
    
    Returns:
        Dictionary containing:
        - token_count: Number of tokens in the text
        - model: Model used for tokenization
        - text_length: Character length of text
    """
    # Create gateway
    gateway = get_llm_gateway(model=model)
    
    # Count tokens
    token_count = gateway.count_tokens(text, model=gateway.model)
    
    return {
        "token_count": token_count,
        "model": gateway.model,
        "text_length": len(text),
    }


async def llm_multi_model_generate(
    user_prompt: str,
    models: list[str],
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate responses from multiple models in parallel.
    
    This tool enables:
    - Comparing outputs across models
    - Consensus building
    - Model ensemble approaches
    
    Args:
        user_prompt: The main prompt/question for the LLMs
        models: List of model strings (e.g., ["openai:gpt-4o", "mistral:mistral-large-latest"])
        system_prompt: Optional system prompt to set context/behavior
        temperature: Optional temperature for generation (0.0-1.0)
        max_tokens: Optional maximum tokens to generate
    
    Returns:
        Dictionary containing:
        - responses: List of responses from each model
        - model_count: Number of models queried
    """
    import asyncio
    
    async def generate_from_model(model_string: str) -> Dict[str, Any]:
        """Helper to generate from a single model."""
        gateway = get_llm_gateway(model=model_string, temperature=temperature)
        
        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        response = gateway.generate(request)
        
        return {
            "model": response.model,
            "content": response.content,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
        }
    
    # Generate from all models in parallel
    tasks = [generate_from_model(model) for model in models]
    responses = await asyncio.gather(*tasks)
    
    return {
        "responses": responses,
        "model_count": len(models),
    }


async def llm_summarize(
    text: str,
    model: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Summarize text using an LLM.
    
    A convenience tool for text summarization.
    
    Args:
        text: Text to summarize
        model: Optional model string
        max_length: Optional maximum length for summary in words
    
    Returns:
        Dictionary containing:
        - summary: The generated summary
        - original_length: Length of original text
        - summary_length: Length of summary
        - model: Model used
    """
    gateway = get_llm_gateway(model=model)
    
    # Build prompt
    length_instruction = f" Keep it under {max_length} words." if max_length else ""
    system_prompt = "You are a helpful assistant that creates concise, accurate summaries."
    user_prompt = f"Please summarize the following text:{length_instruction}\n\n{text}"
    
    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    response = gateway.generate(request)
    
    return {
        "summary": response.content,
        "original_length": len(text.split()),
        "summary_length": len(response.content.split()),
        "model": response.model,
        "usage": response.usage,
    }


async def llm_extract_entities(
    text: str,
    entity_types: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract named entities from text using an LLM.
    
    Args:
        text: Text to extract entities from
        entity_types: Optional list of entity types to extract (e.g., ["person", "organization", "location"])
        model: Optional model string
    
    Returns:
        Dictionary containing:
        - entities: Extracted entities
        - model: Model used
    """
    gateway = get_llm_gateway(model=model)
    
    # Build prompt
    if entity_types:
        types_str = ", ".join(entity_types)
        entity_instruction = f" Focus on these entity types: {types_str}."
    else:
        entity_instruction = ""
    
    system_prompt = "You are a helpful assistant that extracts named entities from text. Return results as JSON."
    user_prompt = f"Extract named entities from the following text.{entity_instruction} Return as JSON with entity type as keys and lists of entities as values.\n\nText: {text}"
    
    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format="json",
    )
    
    response = gateway.generate(request)
    
    # Try to parse as JSON
    try:
        entities = json.loads(response.content)
    except json.JSONDecodeError:
        entities = {"error": "Failed to parse entities", "raw": response.content}
    
    return {
        "entities": entities,
        "model": response.model,
        "usage": response.usage,
    }


async def llm_translate(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Translate text using an LLM.
    
    Args:
        text: Text to translate
        target_language: Target language (e.g., "Spanish", "French", "German")
        source_language: Optional source language (auto-detected if not specified)
        model: Optional model string
    
    Returns:
        Dictionary containing:
        - translation: The translated text
        - source_language: Source language used
        - target_language: Target language
        - model: Model used
    """
    gateway = get_llm_gateway(model=model)
    
    # Build prompt
    source_instruction = f" from {source_language}" if source_language else ""
    system_prompt = "You are a helpful assistant that translates text accurately."
    user_prompt = f"Translate the following text{source_instruction} to {target_language}:\n\n{text}"
    
    request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    response = gateway.generate(request)
    
    return {
        "translation": response.content,
        "source_language": source_language or "auto-detected",
        "target_language": target_language,
        "model": response.model,
        "usage": response.usage,
    }


# === Helper Functions for Tool Registration ===

def get_all_llm_gateway_tools() -> list[Dict[str, Any]]:
    """
    Get configuration for all LLM gateway tools.
    
    Returns:
        List of tool configuration dictionaries ready for registration
    """
    return [
        {
            "name": "llm_generate",
            "function": llm_generate,
            "description": "Generate text using an LLM through the gateway. Enables multi-step reasoning and delegation.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "The main prompt/question for the LLM"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set context/behavior"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string (e.g., 'openai:gpt-4o', 'mistral:mistral-large-latest')"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional temperature for generation (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional maximum tokens to generate",
                        "minimum": 1
                    }
                },
                "required": ["user_prompt"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["generation", "llm", "delegation"],
        },
        {
            "name": "llm_generate_structured",
            "function": llm_generate_structured,
            "description": "Generate structured output using an LLM, ensuring responses conform to a schema.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "The main prompt/question for the LLM"
                    },
                    "schema": {
                        "type": "object",
                        "description": "JSON schema defining the expected output structure"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set context/behavior"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional temperature for generation (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional maximum tokens to generate",
                        "minimum": 1
                    }
                },
                "required": ["user_prompt", "schema"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["generation", "structured", "llm"],
        },
        {
            "name": "llm_count_tokens",
            "function": llm_count_tokens,
            "description": "Count tokens in text using the LLM gateway's tokenizer. Useful for checking prompt lengths.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to count tokens for"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string to use for tokenization"
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["tokens", "utility", "llm"],
        },
        {
            "name": "llm_multi_model_generate",
            "function": llm_multi_model_generate,
            "description": "Generate responses from multiple models in parallel for comparison or consensus.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "The main prompt/question for the LLMs"
                    },
                    "models": {
                        "type": "array",
                        "description": "List of model strings to query",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set context/behavior"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional temperature for generation (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional maximum tokens to generate",
                        "minimum": 1
                    }
                },
                "required": ["user_prompt", "models"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["generation", "multi-model", "ensemble", "llm"],
        },
        {
            "name": "llm_summarize",
            "function": llm_summarize,
            "description": "Summarize text using an LLM.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Optional maximum length for summary in words",
                        "minimum": 1
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["summarization", "nlp", "llm"],
        },
        {
            "name": "llm_extract_entities",
            "function": llm_extract_entities,
            "description": "Extract named entities from text using an LLM.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract entities from"
                    },
                    "entity_types": {
                        "type": "array",
                        "description": "Optional list of entity types to extract",
                        "items": {
                            "type": "string"
                        }
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string"
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["entities", "nlp", "extraction", "llm"],
        },
        {
            "name": "llm_translate",
            "function": llm_translate,
            "description": "Translate text using an LLM.",
            "json_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to translate"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language (e.g., 'Spanish', 'French', 'German')"
                    },
                    "source_language": {
                        "type": "string",
                        "description": "Optional source language (auto-detected if not specified)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model string"
                    }
                },
                "required": ["text", "target_language"],
                "additionalProperties": False,
            },
            "category": "llm",
            "tags": ["translation", "nlp", "llm"],
        },
    ]

