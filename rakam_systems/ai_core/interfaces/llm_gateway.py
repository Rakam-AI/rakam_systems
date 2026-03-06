from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Type, TypeVar
from pydantic import BaseModel
from ..base import BaseComponent

T = TypeVar("T", bound=BaseModel)


class LLMRequest(BaseModel):
    """Standardized LLM request structure."""
    system_prompt: Optional[str] = None
    user_prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None  # "text" or "json"
    json_schema: Optional[Type[BaseModel]] = None
    extra_params: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class LLMResponse(BaseModel):
    """Standardized LLM response structure."""
    content: str
    parsed_content: Optional[Any] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class LLMGateway(BaseComponent, ABC):
    """Abstract base class for LLM gateway implementations.
    
    This gateway provides a standardized interface for interacting with various
    LLM providers (OpenAI, Mistral, etc.) with support for:
    - Text generation
    - Structured output generation
    - Streaming responses
    - Token counting
    """
    
    def __init__(
        self,
        name: str = "llm_gateway",
        config: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        default_temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        super().__init__(name, config)
        self.provider = provider
        self.model = model
        self.default_temperature = default_temperature
        self.api_key = api_key
    
    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_structured(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Generate structured output conforming to a Pydantic schema.
        
        Args:
            request: Standardized LLM request
            schema: Pydantic model class to parse response into
            
        Returns:
            Instance of the schema class
        """
        raise NotImplementedError
    
    def stream(
        self,
        request: LLMRequest,
    ) -> Iterator[str]:
        """Stream token/segment responses.
        
        Args:
            request: Standardized LLM request
            
        Yields:
            String chunks from the LLM
        """
        # Default implementation yields full response
        response = self.generate(request)
        yield response.content
    
    @abstractmethod
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            model: Model name to determine encoding
            
        Returns:
            Number of tokens
        """
        raise NotImplementedError
    
    # Legacy methods for backward compatibility
    def run(self, prompt: str, **kwargs: Any) -> str:
        """Legacy synchronous text completion."""
        request = LLMRequest(
            user_prompt=prompt,
            system_prompt=kwargs.get("system_prompt"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            extra_params=kwargs,
        )
        response = self.generate(request)
        return response.content
