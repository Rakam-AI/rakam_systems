"""
Example demonstrating the centralized LLM Gateway system.

This example shows:
1. Creating gateways for different providers (OpenAI, Mistral)
2. Using the factory for configuration-driven model selection
3. Text generation
4. Structured output generation
5. Streaming responses
6. Token counting
7. Configuration-based gateway creation
"""

import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up example API keys (in production, use environment variables)
# os.environ["OPENAI_API_KEY"] = "your-key-here"
# os.environ["MISTRAL_API_KEY"] = "your-key-here"


def example_1_basic_text_generation():
    """Example 1: Basic text generation with different providers."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Text Generation")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMRequest,
        get_llm_gateway,
    )
    
    # Create OpenAI gateway
    print("\n--- Using OpenAI ---")
    openai_gateway = get_llm_gateway(model="openai:gpt-4o-mini", temperature=0.7)
    
    request = LLMRequest(
        system_prompt="You are a helpful assistant that provides concise answers.",
        user_prompt="What is artificial intelligence? Answer in 2 sentences.",
        temperature=0.7,
    )
    
    response = openai_gateway.generate(request)
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage}")
    
    # Create Mistral gateway (skip if no API key)
    if os.getenv("MISTRAL_API_KEY"):
        print("\n--- Using Mistral ---")
        mistral_gateway = get_llm_gateway(model="mistral:mistral-small-latest", temperature=0.7)
        
        response = mistral_gateway.generate(request)
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage}")
    else:
        print("\n--- Skipping Mistral (API key not set) ---")


def example_2_structured_output():
    """Example 2: Structured output generation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Structured Output Generation")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMRequest,
        OpenAIGateway,
        MistralGateway,
    )
    
    # Define a Pydantic schema
    class Book(BaseModel):
        title: str = Field(description="The book title")
        author: str = Field(description="The author name")
        year: int = Field(description="Publication year")
        genre: str = Field(description="Book genre")
        summary: str = Field(description="Brief summary")
    
    request = LLMRequest(
        system_prompt="You are a librarian assistant.",
        user_prompt="Provide information about the book '1984' by George Orwell.",
        temperature=0.3,
    )
    
    # OpenAI structured output
    print("\n--- OpenAI Structured Output ---")
    openai_gateway = OpenAIGateway(model="gpt-4o-mini")
    book_info = openai_gateway.generate_structured(request, Book)
    print(f"Title: {book_info.title}")
    print(f"Author: {book_info.author}")
    print(f"Year: {book_info.year}")
    print(f"Genre: {book_info.genre}")
    print(f"Summary: {book_info.summary}")
    
    # Mistral structured output (skip if no API key)
    if os.getenv("MISTRAL_API_KEY"):
        print("\n--- Mistral Structured Output ---")
        mistral_gateway = MistralGateway(model="mistral-small-latest")
        book_info = mistral_gateway.generate_structured(request, Book)
        print(f"Title: {book_info.title}")
        print(f"Author: {book_info.author}")
        print(f"Year: {book_info.year}")
        print(f"Genre: {book_info.genre}")
        print(f"Summary: {book_info.summary}")
    else:
        print("\n--- Skipping Mistral (API key not set) ---")


def example_3_streaming():
    """Example 3: Streaming responses."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Streaming Responses")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMRequest,
        get_llm_gateway,
    )
    
    gateway = get_llm_gateway(model="openai:gpt-4o-mini")
    
    request = LLMRequest(
        system_prompt="You are a storyteller.",
        user_prompt="Write a very short story about a robot learning to paint.",
        temperature=0.8,
    )
    
    print("\n--- Streaming Story ---")
    for chunk in gateway.stream(request):
        print(chunk, end="", flush=True)
    print("\n")


def example_4_token_counting():
    """Example 4: Token counting."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Token Counting")
    print("="*80)
    
    from ai_agents.components.llm_gateway import get_llm_gateway
    
    gateway = get_llm_gateway(model="openai:gpt-4o")
    
    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we live and work.",
    ]
    
    for text in texts:
        token_count = gateway.count_tokens(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {token_count}")
        print(f"Characters: {len(text)}")
        print()


def example_5_factory_patterns():
    """Example 5: Using the factory for different creation patterns."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Factory Creation Patterns")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMGatewayFactory,
        LLMRequest,
    )
    
    # Pattern 1: Create from model string
    print("\n--- Pattern 1: Model String ---")
    gateway1 = LLMGatewayFactory.create_gateway("openai:gpt-4o-mini")
    print(f"Created gateway: provider={gateway1.provider}, model={gateway1.model}")
    
    # Pattern 2: Create from config dictionary (skip if no Mistral key)
    if os.getenv("MISTRAL_API_KEY"):
        print("\n--- Pattern 2: Config Dictionary ---")
        config = {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "temperature": 0.5,
        }
        gateway2 = LLMGatewayFactory.create_gateway_from_config(config)
        print(f"Created gateway: provider={gateway2.provider}, model={gateway2.model}")
    else:
        print("\n--- Pattern 2: Skipping (Mistral API key not set) ---")
    
    # Pattern 3: Get default gateway
    print("\n--- Pattern 3: Default Gateway ---")
    os.environ["DEFAULT_LLM_MODEL"] = "openai:gpt-4o-mini"
    gateway3 = LLMGatewayFactory.get_default_gateway()
    print(f"Created gateway: provider={gateway3.provider}, model={gateway3.model}")
    
    # Pattern 4: List available providers
    print("\n--- Pattern 4: List Providers ---")
    providers = LLMGatewayFactory.list_providers()
    print(f"Available providers: {providers}")
    
    for provider in providers:
        default_model = LLMGatewayFactory.get_default_model(provider)
        print(f"  {provider}: default model = {default_model}")


def example_6_config_driven_usage():
    """Example 6: Configuration-driven gateway usage."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Configuration-Driven Usage")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMGatewayFactory,
        LLMRequest,
    )
    
    # Simulate loading from a config file
    app_config = {
        "llm_gateways": {
            "default": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
            },
            "creative": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.9,
            },
            "analytical": {
                "provider": "mistral",
                "model": "mistral-small-latest",
                "temperature": 0.3,
            }
        }
    }
    
    # Create different gateways for different purposes
    default_gateway = LLMGatewayFactory.create_gateway_from_config(
        app_config["llm_gateways"]["default"]
    )
    creative_gateway = LLMGatewayFactory.create_gateway_from_config(
        app_config["llm_gateways"]["creative"]
    )
    
    print(f"Default: {default_gateway.provider}:{default_gateway.model} @ {default_gateway.default_temperature}")
    print(f"Creative: {creative_gateway.provider}:{creative_gateway.model} @ {creative_gateway.default_temperature}")
    
    # Only create analytical gateway if Mistral key is available
    if os.getenv("MISTRAL_API_KEY"):
        analytical_gateway = LLMGatewayFactory.create_gateway_from_config(
            app_config["llm_gateways"]["analytical"]
        )
        print(f"Analytical: {analytical_gateway.provider}:{analytical_gateway.model} @ {analytical_gateway.default_temperature}")
    else:
        print("Analytical: Skipping (Mistral API key not set)")
    
    # Use different gateways for different tasks
    request = LLMRequest(
        user_prompt="Write one sentence about AI.",
    )
    
    print("\n--- Default Gateway Response ---")
    response = default_gateway.generate(request)
    print(response.content)


def example_7_complex_structured_output():
    """Example 7: Complex structured output with nested models."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Complex Structured Output")
    print("="*80)
    
    from ai_agents.components.llm_gateway import (
        LLMRequest,
        OpenAIGateway,
    )
    
    # Define nested Pydantic schemas
    class Person(BaseModel):
        name: str = Field(description="Person's full name")
        role: str = Field(description="Person's role")
        age: int = Field(description="Person's age")
    
    class Company(BaseModel):
        name: str = Field(description="Company name")
        industry: str = Field(description="Industry sector")
        founded_year: int = Field(description="Year company was founded")
        employees: List[Person] = Field(description="Key employees")
        headquarters: str = Field(description="Headquarters location")
    
    gateway = OpenAIGateway(model="gpt-4o-mini")
    
    request = LLMRequest(
        system_prompt="You are a business analyst.",
        user_prompt="Provide information about a fictional tech startup called 'NeuralFlow AI' "
                   "founded in 2020, with 3 key employees.",
        temperature=0.5,
    )
    
    company_info = gateway.generate_structured(request, Company)
    
    print(f"\nCompany: {company_info.name}")
    print(f"Industry: {company_info.industry}")
    print(f"Founded: {company_info.founded_year}")
    print(f"Headquarters: {company_info.headquarters}")
    print(f"\nKey Employees:")
    for person in company_info.employees:
        print(f"  - {person.name}, {person.role}, Age {person.age}")


def example_8_error_handling():
    """Example 8: Error handling and fallbacks."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Error Handling")
    print("="*80)
    
    from ai_agents.components.llm_gateway import LLMGatewayFactory
    
    # Try to create gateway with invalid provider
    print("\n--- Invalid Provider ---")
    try:
        gateway = LLMGatewayFactory.create_gateway("invalid_provider:some-model")
    except ValueError as e:
        print(f"Error caught: {e}")
    
    # Try to create gateway without API key
    print("\n--- Missing API Key ---")
    try:
        # Temporarily remove API key
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        gateway = LLMGatewayFactory.create_gateway("openai:gpt-4o")
    except ValueError as e:
        print(f"Error caught: {e}")
        # Restore API key if it existed
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key
    
    print("\n--- Successful Creation with Valid Config ---")
    gateway = LLMGatewayFactory.create_gateway("openai:gpt-4o-mini")
    print(f"Gateway created successfully: {gateway.provider}:{gateway.model}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("LLM GATEWAY SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    print("\nNOTE: Make sure to set OPENAI_API_KEY environment variable to run examples.")
    print("MISTRAL_API_KEY is optional - Mistral examples will be skipped if not set.\n")
    
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. OpenAI examples will fail.")
    if not os.getenv("MISTRAL_API_KEY"):
        print("INFO: MISTRAL_API_KEY not set. Mistral examples will be skipped.")
    
    try:
        # Run examples
        example_1_basic_text_generation()
        example_2_structured_output()
        example_3_streaming()
        example_4_token_counting()
        example_5_factory_patterns()
        example_6_config_driven_usage()
        example_7_complex_structured_output()
        example_8_error_handling()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

