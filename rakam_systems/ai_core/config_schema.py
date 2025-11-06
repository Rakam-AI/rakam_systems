"""
Pydantic schemas for agent configuration and input/output tracking.

This module provides comprehensive schemas for:
1. Agent configuration (tools, models, prompts, skills)
2. Input/output tracking for agent methods
3. Evaluation data structures
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


# ============================================================================
# Configuration Schemas
# ============================================================================

class ToolMode(str, Enum):
    """Tool invocation modes."""
    DIRECT = "direct"
    MCP = "mcp"


class ToolConfigSchema(BaseModel):
    """Configuration schema for a single tool."""
    model_config = {"use_enum_values": True}
    
    name: str = Field(..., description="Unique name for the tool")
    type: ToolMode = Field(..., description="Tool invocation mode (direct or mcp)")
    description: str = Field(..., description="Human-readable description of the tool")
    
    # Direct tool fields
    module: Optional[str] = Field(None, description="Python module path for direct tools")
    function: Optional[str] = Field(None, description="Function name for direct tools")
    
    # MCP tool fields
    mcp_server: Optional[str] = Field(None, description="MCP server name for MCP tools")
    mcp_tool_name: Optional[str] = Field(None, description="Tool name on MCP server")
    
    # JSON Schema for tool parameters (renamed from 'schema' to avoid shadowing)
    json_schema: Optional[Dict[str, Any]] = Field(default_factory=dict, description="JSON Schema for tool parameters")
    
    # Organization
    category: Optional[str] = Field("general", description="Category for organizing tools")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering tools")
    takes_ctx: bool = Field(False, description="Whether tool takes context as first argument")
    
    @validator("module")
    def validate_direct_tool(cls, v, values):
        """Validate that direct tools have required fields."""
        if values.get("type") == ToolMode.DIRECT and not v:
            raise ValueError("Direct tools must specify 'module'")
        return v
    
    @validator("mcp_server")
    def validate_mcp_tool(cls, v, values):
        """Validate that MCP tools have required fields."""
        if values.get("type") == ToolMode.MCP and not v:
            raise ValueError("MCP tools must specify 'mcp_server'")
        return v


class ModelConfigSchema(BaseModel):
    """Configuration schema for LLM model settings."""
    model_config = {"extra": "allow"}  # Allow additional fields
    
    model: str = Field(..., description="Model identifier (e.g., 'openai:gpt-4o')")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens in response")
    parallel_tool_calls: bool = Field(True, description="Enable parallel tool execution")
    extra_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional model settings")


class PromptConfigSchema(BaseModel):
    """Configuration schema for system prompts and skill sets."""
    model_config = {"extra": "allow"}
    
    name: str = Field(..., description="Prompt identifier")
    system_prompt: str = Field(..., description="System prompt text")
    description: Optional[str] = Field(None, description="Description of this prompt/skill")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing prompts")
    
    # Skills can be structured like Anthropic's approach
    skills: List[str] = Field(default_factory=list, description="List of skills/capabilities")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Example interactions")


class AgentConfigSchema(BaseModel):
    """Complete agent configuration schema."""
    model_config = {"extra": "allow"}
    
    name: str = Field(..., description="Agent name/identifier")
    description: Optional[str] = Field(None, description="Agent description")
    
    # Model configuration
    llm_config: ModelConfigSchema = Field(..., description="LLM model configuration", alias="model_config")
    
    # Prompt configuration
    prompt_config: Union[str, PromptConfigSchema] = Field(
        ..., 
        description="Prompt configuration (name reference or full config)"
    )
    
    # Tools configuration
    tools: List[Union[str, ToolConfigSchema]] = Field(
        default_factory=list,
        description="List of tools (names or full configs)"
    )
    
    # Dependencies
    deps_type: Optional[str] = Field(None, description="Fully qualified class name for dependencies")
    
    # Tracking and evaluation
    enable_tracking: bool = Field(True, description="Enable input/output tracking")
    tracking_output_dir: str = Field("./agent_tracking", description="Directory for tracking outputs")
    
    # Additional settings
    stateful: bool = Field(False, description="Whether agent maintains state")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConfigFileSchema(BaseModel):
    """Root configuration file schema."""
    model_config = {"extra": "allow"}
    
    version: str = Field("1.0", description="Configuration file version")
    
    # Global settings
    global_settings: Dict[str, Any] = Field(default_factory=dict, description="Global settings")
    
    # Prompt library - reusable prompts
    prompts: Dict[str, PromptConfigSchema] = Field(
        default_factory=dict,
        description="Library of reusable prompts"
    )
    
    # Tool library - reusable tools
    tools: Dict[str, ToolConfigSchema] = Field(
        default_factory=dict,
        description="Library of reusable tools"
    )
    
    # Agents
    agents: Dict[str, AgentConfigSchema] = Field(
        default_factory=dict,
        description="Agent configurations"
    )


# ============================================================================
# Input/Output Tracking Schemas
# ============================================================================

class MethodInputSchema(BaseModel):
    """Schema for capturing method input data."""
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
    
    timestamp: datetime = Field(default_factory=datetime.now, description="When method was called")
    method_name: str = Field(..., description="Name of the method")
    agent_name: str = Field(..., description="Name of the agent")
    
    # Input data
    input_text: Optional[str] = Field(None, description="Input text if applicable")
    args: List[Any] = Field(default_factory=list, description="Positional arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Tracking metadata
    call_id: str = Field(..., description="Unique identifier for this call")
    parent_call_id: Optional[str] = Field(None, description="Parent call ID if nested")


class MethodOutputSchema(BaseModel):
    """Schema for capturing method output data."""
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
    
    timestamp: datetime = Field(default_factory=datetime.now, description="When method completed")
    method_name: str = Field(..., description="Name of the method")
    agent_name: str = Field(..., description="Name of the agent")
    
    # Output data
    output_text: Optional[str] = Field(None, description="Output text if applicable")
    result: Any = Field(None, description="Method return value")
    
    # Performance metrics
    duration_seconds: float = Field(..., description="Execution time in seconds")
    
    # Status
    success: bool = Field(..., description="Whether method executed successfully")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Tracking metadata
    call_id: str = Field(..., description="Unique identifier matching input")
    parent_call_id: Optional[str] = Field(None, description="Parent call ID if nested")


class MethodCallRecordSchema(BaseModel):
    """Complete record of a method call (input + output)."""
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
    
    call_id: str = Field(..., description="Unique identifier for this call")
    agent_name: str = Field(..., description="Name of the agent")
    method_name: str = Field(..., description="Name of the method")
    
    # Input/Output
    input_data: MethodInputSchema = Field(..., description="Input data")
    output_data: MethodOutputSchema = Field(..., description="Output data")
    
    # Timing
    started_at: datetime = Field(..., description="When call started")
    completed_at: datetime = Field(..., description="When call completed")
    duration_seconds: float = Field(..., description="Total execution time")
    
    # Evaluation fields (to be filled later)
    evaluation_score: Optional[float] = Field(None, description="Evaluation score if available")
    evaluation_notes: Optional[str] = Field(None, description="Evaluation notes")


class TrackingSessionSchema(BaseModel):
    """Schema for a tracking session containing multiple calls."""
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
    
    session_id: str = Field(..., description="Unique session identifier")
    agent_name: str = Field(..., description="Name of the agent")
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    ended_at: Optional[datetime] = Field(None, description="Session end time")
    
    # Calls in this session
    calls: List[MethodCallRecordSchema] = Field(default_factory=list, description="Method calls in session")
    
    # Session metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    
    # Summary statistics
    total_calls: int = Field(0, description="Total number of calls")
    successful_calls: int = Field(0, description="Number of successful calls")
    failed_calls: int = Field(0, description="Number of failed calls")
    total_duration: float = Field(0.0, description="Total execution time")
    
    def add_call(self, call_record: MethodCallRecordSchema) -> None:
        """Add a call record and update statistics."""
        self.calls.append(call_record)
        self.total_calls += 1
        if call_record.output_data.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_duration += call_record.duration_seconds
    
    def end_session(self) -> None:
        """Mark the session as ended."""
        self.ended_at = datetime.now()


# ============================================================================
# Evaluation Schemas
# ============================================================================

class EvaluationCriteriaSchema(BaseModel):
    """Schema for evaluation criteria."""
    name: str = Field(..., description="Criterion name")
    description: str = Field(..., description="What this criterion measures")
    weight: float = Field(1.0, ge=0.0, description="Weight for this criterion")
    min_score: float = Field(0.0, description="Minimum score")
    max_score: float = Field(1.0, description="Maximum score")


class EvaluationResultSchema(BaseModel):
    """Schema for evaluation results."""
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
    
    call_id: str = Field(..., description="Call ID being evaluated")
    evaluated_at: datetime = Field(default_factory=datetime.now, description="When evaluation occurred")
    evaluator: str = Field(..., description="Who/what performed the evaluation")
    
    # Scores per criterion
    scores: Dict[str, float] = Field(default_factory=dict, description="Scores by criterion")
    
    # Overall
    overall_score: float = Field(..., description="Overall weighted score")
    passed: bool = Field(..., description="Whether evaluation passed")
    
    # Feedback
    feedback: Optional[str] = Field(None, description="Textual feedback")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

