"""
Input/Output tracking system for agent methods.

This module provides:
1. Decorator for tracking method inputs/outputs
2. TrackingMixin for agents to enable tracking
3. CSV export functionality
4. Session management
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
import asyncio
import time
import uuid
from pathlib import Path
from datetime import datetime
import csv
import json

from .config_schema import (
    MethodInputSchema,
    MethodOutputSchema,
    MethodCallRecordSchema,
    TrackingSessionSchema,
)
from .interfaces.agent import AgentInput, AgentOutput


F = TypeVar('F', bound=Callable[..., Any])


class TrackingManager:
    """
    Manages tracking of agent method calls.
    
    Features:
    - Track inputs and outputs
    - Session management
    - CSV export
    - JSON export
    - Query and analysis
    """
    
    def __init__(self, output_dir: str = "./agent_tracking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, TrackingSessionSchema] = {}
        self.current_session_id: Optional[str] = None
        self.call_records: List[MethodCallRecordSchema] = []
    
    def start_session(self, agent_name: str, session_id: Optional[str] = None) -> str:
        """
        Start a new tracking session.
        
        Args:
            agent_name: Name of the agent
            session_id: Optional session ID (generates UUID if None)
            
        Returns:
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        
        session = TrackingSessionSchema(
            session_id=session_id,
            agent_name=agent_name,
            started_at=datetime.now(),
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        return session_id
    
    def end_session(self, session_id: Optional[str] = None) -> None:
        """
        End a tracking session.
        
        Args:
            session_id: Session to end (uses current if None)
        """
        session_id = session_id or self.current_session_id
        if session_id and session_id in self.sessions:
            self.sessions[session_id].end_session()
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[TrackingSessionSchema]:
        """Get a session by ID (current session if None)."""
        session_id = session_id or self.current_session_id
        return self.sessions.get(session_id) if session_id else None
    
    def record_call(
        self,
        agent_name: str,
        method_name: str,
        input_data: MethodInputSchema,
        output_data: MethodOutputSchema,
        session_id: Optional[str] = None,
    ) -> MethodCallRecordSchema:
        """
        Record a complete method call.
        
        Args:
            agent_name: Name of the agent
            method_name: Name of the method
            input_data: Input data schema
            output_data: Output data schema
            session_id: Session to add to (uses current if None)
            
        Returns:
            Created call record
        """
        record = MethodCallRecordSchema(
            call_id=input_data.call_id,
            agent_name=agent_name,
            method_name=method_name,
            input_data=input_data,
            output_data=output_data,
            started_at=input_data.timestamp,
            completed_at=output_data.timestamp,
            duration_seconds=output_data.duration_seconds,
        )
        
        self.call_records.append(record)
        
        # Add to session if exists
        session_id = session_id or self.current_session_id
        if session_id and session_id in self.sessions:
            self.sessions[session_id].add_call(record)
        
        return record
    
    def export_to_csv(
        self,
        filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Path:
        """
        Export tracking data to CSV.
        
        Args:
            filename: Output filename (auto-generates if None)
            session_id: Session to export (all records if None)
            
        Returns:
            Path to created CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Determine records to export
        if session_id:
            session = self.sessions.get(session_id)
            records = session.calls if session else []
        else:
            records = self.call_records
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_fieldnames())
            writer.writeheader()
            
            for record in records:
                row = self._record_to_csv_row(record)
                writer.writerow(row)
        
        return output_path
    
    def _get_csv_fieldnames(self) -> List[str]:
        """Get CSV column names."""
        return [
            'call_id',
            'agent_name',
            'method_name',
            'started_at',
            'completed_at',
            'duration_seconds',
            'success',
            'input_text',
            'output_text',
            'error',
            'evaluation_score',
            'evaluation_notes',
            # Metadata fields
            'model',
            'temperature',
            'max_tokens',
            'parallel_tool_calls',
            'tool_calls_count',
            'usage_prompt_tokens',
            'usage_completion_tokens',
            'usage_total_tokens',
        ]
    
    def _record_to_csv_row(self, record: MethodCallRecordSchema) -> Dict[str, Any]:
        """Convert a record to a CSV row."""
        # Extract metadata
        metadata = record.output_data.metadata or {}
        usage_obj = metadata.get('usage')
        
        # Extract usage data - handle both dict and object types
        usage_prompt = ''
        usage_completion = ''
        usage_total = ''
        if usage_obj:
            if isinstance(usage_obj, dict):
                usage_prompt = usage_obj.get('request_tokens', '') or usage_obj.get('prompt_tokens', '')
                usage_completion = usage_obj.get('response_tokens', '') or usage_obj.get('completion_tokens', '')
                usage_total = usage_obj.get('total_tokens', '')
            elif hasattr(usage_obj, 'request_tokens'):
                usage_prompt = getattr(usage_obj, 'request_tokens', '')
                usage_completion = getattr(usage_obj, 'response_tokens', '')
                usage_total = getattr(usage_obj, 'total_tokens', '')
        
        # Count tool calls if messages available
        tool_calls_count = 0
        messages = metadata.get('messages', [])
        if messages:
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'tool_name'):
                            tool_calls_count += 1
        
        # Safely extract model settings from kwargs
        kwargs = record.input_data.kwargs or {}
        model_settings = kwargs.get('model_settings', {}) or {}
        
        return {
            'call_id': record.call_id,
            'agent_name': record.agent_name,
            'method_name': record.method_name,
            'started_at': record.started_at.isoformat() if record.started_at else '',
            'completed_at': record.completed_at.isoformat() if record.completed_at else '',
            'duration_seconds': record.duration_seconds,
            'success': record.output_data.success,
            'input_text': record.input_data.input_text or '',
            'output_text': record.output_data.output_text or '',
            'error': record.output_data.error or '',
            'evaluation_score': record.evaluation_score or '',
            'evaluation_notes': record.evaluation_notes or '',
            # Metadata
            'model': model_settings.get('model', ''),
            'temperature': model_settings.get('temperature', ''),
            'max_tokens': model_settings.get('max_tokens', ''),
            'parallel_tool_calls': model_settings.get('parallel_tool_calls', ''),
            'tool_calls_count': tool_calls_count,
            'usage_prompt_tokens': usage_prompt,
            'usage_completion_tokens': usage_completion,
            'usage_total_tokens': usage_total,
        }
    
    def export_to_json(
        self,
        filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Path:
        """
        Export tracking data to JSON.
        
        Args:
            filename: Output filename (auto-generates if None)
            session_id: Session to export (all records if None)
            
        Returns:
            Path to created JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Determine records to export
        if session_id:
            session = self.sessions.get(session_id)
            if session:
                data = session.dict()
            else:
                data = {"error": "Session not found"}
        else:
            data = {
                "records": [record.dict() for record in self.call_records],
                "total_records": len(self.call_records),
            }
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Args:
            session_id: Session to analyze (all records if None)
            
        Returns:
            Statistics dictionary
        """
        if session_id:
            session = self.sessions.get(session_id)
            records = session.calls if session else []
        else:
            records = self.call_records
        
        if not records:
            return {"total_calls": 0}
        
        successful = sum(1 for r in records if r.output_data.success)
        failed = len(records) - successful
        total_duration = sum(r.duration_seconds for r in records)
        avg_duration = total_duration / len(records)
        
        return {
            "total_calls": len(records),
            "successful_calls": successful,
            "failed_calls": failed,
            "success_rate": successful / len(records) if records else 0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "min_duration_seconds": min(r.duration_seconds for r in records),
            "max_duration_seconds": max(r.duration_seconds for r in records),
        }


# Global tracking manager instance
_global_tracking_manager: Optional[TrackingManager] = None


def get_tracking_manager(output_dir: str = "./agent_tracking") -> TrackingManager:
    """Get or create the global tracking manager."""
    global _global_tracking_manager
    if _global_tracking_manager is None:
        _global_tracking_manager = TrackingManager(output_dir)
    return _global_tracking_manager


def track_method(
    method_name: Optional[str] = None,
    track_args: bool = True,
    track_kwargs: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to track method inputs and outputs.
    
    Args:
        method_name: Override method name (uses actual name if None)
        track_args: Whether to track positional arguments
        track_kwargs: Whether to track keyword arguments
        
    Returns:
        Decorated function
        
    Example:
        >>> @track_method()
        >>> async def arun(self, input_data, deps=None):
        >>>     return result
    """
    def decorator(func: F) -> F:
        actual_method_name = method_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Check if tracking is enabled
            if not getattr(self, '_tracking_enabled', False):
                return await func(self, *args, **kwargs)
            
            # Get tracking manager
            output_dir = getattr(self, '_tracking_output_dir', './agent_tracking')
            manager = get_tracking_manager(output_dir)
            
            # Generate call ID
            call_id = str(uuid.uuid4())
            
            # Extract input text if available
            input_text = None
            if args:
                if isinstance(args[0], str):
                    input_text = args[0]
                elif isinstance(args[0], AgentInput):
                    input_text = args[0].input_text
            
            # Create input record
            input_data = MethodInputSchema(
                timestamp=datetime.now(),
                method_name=actual_method_name,
                agent_name=self.name,
                input_text=input_text,
                args=list(args) if track_args else [],
                kwargs=dict(kwargs) if track_kwargs else {},
                call_id=call_id,
            )
            
            # Execute method
            start_time = time.time()
            success = True
            error = None
            result = None
            output_text = None
            metadata = {}
            
            try:
                result = await func(self, *args, **kwargs)
                
                # Extract output text if available
                if isinstance(result, AgentOutput):
                    output_text = result.output_text
                    metadata = result.metadata or {}
                elif isinstance(result, str):
                    output_text = result
                
            except Exception as e:
                success = False
                error = str(e)
                raise
            
            finally:
                duration = time.time() - start_time
                
                # Create output record
                output_data = MethodOutputSchema(
                    timestamp=datetime.now(),
                    method_name=actual_method_name,
                    agent_name=self.name,
                    output_text=output_text,
                    result=result if success else None,
                    duration_seconds=duration,
                    success=success,
                    error=error,
                    metadata=metadata,
                    call_id=call_id,
                )
                
                # Record the call
                manager.record_call(
                    agent_name=self.name,
                    method_name=actual_method_name,
                    input_data=input_data,
                    output_data=output_data,
                )
            
            return result
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Check if tracking is enabled
            if not getattr(self, '_tracking_enabled', False):
                return func(self, *args, **kwargs)
            
            # Get tracking manager
            output_dir = getattr(self, '_tracking_output_dir', './agent_tracking')
            manager = get_tracking_manager(output_dir)
            
            # Generate call ID
            call_id = str(uuid.uuid4())
            
            # Extract input text if available
            input_text = None
            if args:
                if isinstance(args[0], str):
                    input_text = args[0]
                elif isinstance(args[0], AgentInput):
                    input_text = args[0].input_text
            
            # Create input record
            input_data = MethodInputSchema(
                timestamp=datetime.now(),
                method_name=actual_method_name,
                agent_name=self.name,
                input_text=input_text,
                args=list(args) if track_args else [],
                kwargs=dict(kwargs) if track_kwargs else {},
                call_id=call_id,
            )
            
            # Execute method
            start_time = time.time()
            success = True
            error = None
            result = None
            output_text = None
            metadata = {}
            
            try:
                result = func(self, *args, **kwargs)
                
                # Extract output text if available
                if isinstance(result, AgentOutput):
                    output_text = result.output_text
                    metadata = result.metadata or {}
                elif isinstance(result, str):
                    output_text = result
                
            except Exception as e:
                success = False
                error = str(e)
                raise
            
            finally:
                duration = time.time() - start_time
                
                # Create output record
                output_data = MethodOutputSchema(
                    timestamp=datetime.now(),
                    method_name=actual_method_name,
                    agent_name=self.name,
                    output_text=output_text,
                    result=result if success else None,
                    duration_seconds=duration,
                    success=success,
                    error=error,
                    metadata=metadata,
                    call_id=call_id,
                )
                
                # Record the call
                manager.record_call(
                    agent_name=self.name,
                    method_name=actual_method_name,
                    input_data=input_data,
                    output_data=output_data,
                )
            
            return result
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


class TrackingMixin:
    """
    Mixin class to add tracking capabilities to agents.
    
    Usage:
        >>> class MyAgent(TrackingMixin, BaseAgent):
        >>>     pass
    """
    
    def __init__(self, *args, **kwargs):
        # Extract tracking-specific kwargs before passing to super
        self._tracking_enabled = kwargs.pop('enable_tracking', False)
        self._tracking_output_dir = kwargs.pop('tracking_output_dir', './agent_tracking')
        self._tracking_manager: Optional[TrackingManager] = None
        super().__init__(*args, **kwargs)
    
    def enable_tracking(self, output_dir: Optional[str] = None) -> None:
        """Enable tracking for this agent."""
        self._tracking_enabled = True
        if output_dir:
            self._tracking_output_dir = output_dir
    
    def disable_tracking(self) -> None:
        """Disable tracking for this agent."""
        self._tracking_enabled = False
    
    def get_tracking_manager(self) -> TrackingManager:
        """Get the tracking manager for this agent."""
        if self._tracking_manager is None:
            self._tracking_manager = get_tracking_manager(self._tracking_output_dir)
        return self._tracking_manager
    
    def export_tracking_data(
        self,
        format: str = 'csv',
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export tracking data.
        
        Args:
            format: 'csv' or 'json'
            filename: Output filename (auto-generates if None)
            
        Returns:
            Path to exported file
        """
        manager = self.get_tracking_manager()
        
        if format == 'csv':
            return manager.export_to_csv(filename)
        elif format == 'json':
            return manager.export_to_json(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics for this agent."""
        manager = self.get_tracking_manager()
        return manager.get_statistics()

