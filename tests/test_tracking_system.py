"""
Tests for the tracking system.

Tests cover:
- TrackingManager functionality
- Method tracking decorator
- TrackingMixin behavior
- CSV/JSON export
- Session management
- Statistics generation
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import csv
import json
from datetime import datetime

from ai_core.tracking import (
    TrackingManager,
    TrackingMixin,
    track_method,
    get_tracking_manager,
)
from ai_core.config_schema import (
    MethodInputSchema,
    MethodOutputSchema,
    MethodCallRecordSchema,
    TrackingSessionSchema,
)
from ai_core.interfaces.agent import AgentInput, AgentOutput


class TestTrackingSchemas:
    """Test tracking data schemas."""
    
    def test_method_input_schema(self):
        """Test MethodInputSchema."""
        input_data = MethodInputSchema(
            timestamp=datetime.now(),
            method_name="test_method",
            agent_name="test_agent",
            input_text="test input",
            args=[],
            kwargs={},
            call_id="test-id",
        )
        
        assert input_data.method_name == "test_method"
        assert input_data.agent_name == "test_agent"
        assert input_data.input_text == "test input"
        assert input_data.call_id == "test-id"
    
    def test_method_output_schema(self):
        """Test MethodOutputSchema."""
        output_data = MethodOutputSchema(
            timestamp=datetime.now(),
            method_name="test_method",
            agent_name="test_agent",
            output_text="test output",
            duration_seconds=1.5,
            success=True,
            call_id="test-id",
        )
        
        assert output_data.method_name == "test_method"
        assert output_data.success is True
        assert output_data.duration_seconds == 1.5
    
    def test_method_call_record_schema(self):
        """Test MethodCallRecordSchema."""
        now = datetime.now()
        
        input_data = MethodInputSchema(
            timestamp=now,
            method_name="test",
            agent_name="agent",
            call_id="id",
        )
        
        output_data = MethodOutputSchema(
            timestamp=now,
            method_name="test",
            agent_name="agent",
            duration_seconds=1.0,
            success=True,
            call_id="id",
        )
        
        record = MethodCallRecordSchema(
            call_id="id",
            agent_name="agent",
            method_name="test",
            input_data=input_data,
            output_data=output_data,
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )
        
        assert record.call_id == "id"
        assert record.agent_name == "agent"
        assert record.duration_seconds == 1.0
    
    def test_tracking_session_schema(self):
        """Test TrackingSessionSchema."""
        session = TrackingSessionSchema(
            session_id="test-session",
            agent_name="test-agent",
        )
        
        assert session.session_id == "test-session"
        assert session.total_calls == 0
        assert session.successful_calls == 0
        assert session.ended_at is None
    
    def test_tracking_session_add_call(self):
        """Test adding calls to session."""
        session = TrackingSessionSchema(
            session_id="test",
            agent_name="agent",
        )
        
        now = datetime.now()
        input_data = MethodInputSchema(
            timestamp=now,
            method_name="test",
            agent_name="agent",
            call_id="id1",
        )
        output_data = MethodOutputSchema(
            timestamp=now,
            method_name="test",
            agent_name="agent",
            duration_seconds=1.0,
            success=True,
            call_id="id1",
        )
        record = MethodCallRecordSchema(
            call_id="id1",
            agent_name="agent",
            method_name="test",
            input_data=input_data,
            output_data=output_data,
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )
        
        session.add_call(record)
        
        assert session.total_calls == 1
        assert session.successful_calls == 1
        assert session.failed_calls == 0
        assert session.total_duration == 1.0


class TestTrackingManager:
    """Test TrackingManager functionality."""
    
    def test_init(self):
        """Test TrackingManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            
            assert manager.output_dir == Path(tmpdir)
            assert len(manager.sessions) == 0
            assert len(manager.call_records) == 0
    
    def test_start_session(self):
        """Test starting a tracking session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            
            session_id = manager.start_session("test_agent")
            
            assert session_id is not None
            assert session_id in manager.sessions
            assert manager.current_session_id == session_id
    
    def test_end_session(self):
        """Test ending a tracking session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            
            session_id = manager.start_session("test_agent")
            manager.end_session(session_id)
            
            session = manager.get_session(session_id)
            assert session is not None
            assert session.ended_at is not None
    
    def test_record_call(self):
        """Test recording a method call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            session_id = manager.start_session("test_agent")
            
            now = datetime.now()
            input_data = MethodInputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                call_id="test-id",
            )
            output_data = MethodOutputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                duration_seconds=1.0,
                success=True,
                call_id="test-id",
            )
            
            record = manager.record_call(
                agent_name="test_agent",
                method_name="test_method",
                input_data=input_data,
                output_data=output_data,
                session_id=session_id,
            )
            
            assert record.call_id == "test-id"
            assert len(manager.call_records) == 1
            
            session = manager.get_session(session_id)
            assert session.total_calls == 1
    
    def test_export_to_csv(self):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            session_id = manager.start_session("test_agent")
            
            # Add a record
            now = datetime.now()
            input_data = MethodInputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                input_text="test input",
                call_id="test-id",
            )
            output_data = MethodOutputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                output_text="test output",
                duration_seconds=1.0,
                success=True,
                call_id="test-id",
            )
            
            manager.record_call(
                agent_name="test_agent",
                method_name="test_method",
                input_data=input_data,
                output_data=output_data,
                session_id=session_id,
            )
            
            # Export
            csv_path = manager.export_to_csv(filename="test.csv")
            
            assert csv_path.exists()
            
            # Read and verify
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 1
                assert rows[0]['call_id'] == 'test-id'
                assert rows[0]['agent_name'] == 'test_agent'
                assert rows[0]['method_name'] == 'test_method'
                assert rows[0]['input_text'] == 'test input'
                assert rows[0]['output_text'] == 'test output'
                assert rows[0]['success'] == 'True'
    
    def test_export_to_json(self):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            session_id = manager.start_session("test_agent")
            
            # Add a record
            now = datetime.now()
            input_data = MethodInputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                call_id="test-id",
            )
            output_data = MethodOutputSchema(
                timestamp=now,
                method_name="test_method",
                agent_name="test_agent",
                duration_seconds=1.0,
                success=True,
                call_id="test-id",
            )
            
            manager.record_call(
                agent_name="test_agent",
                method_name="test_method",
                input_data=input_data,
                output_data=output_data,
                session_id=session_id,
            )
            
            # Export
            json_path = manager.export_to_json(filename="test.json")
            
            assert json_path.exists()
            
            # Read and verify
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                assert data['session_id'] == session_id
                assert data['agent_name'] == 'test_agent'
                assert data['total_calls'] == 1
    
    def test_get_statistics(self):
        """Test statistics generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TrackingManager(output_dir=tmpdir)
            session_id = manager.start_session("test_agent")
            
            # Add successful record
            now = datetime.now()
            input_data = MethodInputSchema(
                timestamp=now,
                method_name="test",
                agent_name="test_agent",
                call_id="id1",
            )
            output_data = MethodOutputSchema(
                timestamp=now,
                method_name="test",
                agent_name="test_agent",
                duration_seconds=1.0,
                success=True,
                call_id="id1",
            )
            manager.record_call("test_agent", "test", input_data, output_data, session_id)
            
            # Add failed record
            input_data2 = MethodInputSchema(
                timestamp=now,
                method_name="test",
                agent_name="test_agent",
                call_id="id2",
            )
            output_data2 = MethodOutputSchema(
                timestamp=now,
                method_name="test",
                agent_name="test_agent",
                duration_seconds=2.0,
                success=False,
                error="Test error",
                call_id="id2",
            )
            manager.record_call("test_agent", "test", input_data2, output_data2, session_id)
            
            # Get statistics
            stats = manager.get_statistics()
            
            assert stats['total_calls'] == 2
            assert stats['successful_calls'] == 1
            assert stats['failed_calls'] == 1
            assert stats['success_rate'] == 0.5
            assert stats['total_duration_seconds'] == 3.0
            assert stats['average_duration_seconds'] == 1.5


class TestTrackMethodDecorator:
    """Test the @track_method decorator."""
    
    @pytest.mark.asyncio
    async def test_async_method_tracking(self):
        """Test tracking async methods."""
        
        class TestAgent:
            def __init__(self):
                self.name = "test_agent"
                self._tracking_enabled = True
                self._tracking_output_dir = tempfile.mkdtemp()
            
            def get_tracking_manager(self):
                return get_tracking_manager(self._tracking_output_dir)
            
            @track_method()
            async def test_method(self, input_text: str) -> AgentOutput:
                await asyncio.sleep(0.1)
                return AgentOutput(output_text=f"Response to: {input_text}")
        
        agent = TestAgent()
        result = await agent.test_method("test input")
        
        assert result.output_text == "Response to: test input"
        
        # Check tracking
        manager = agent.get_tracking_manager()
        stats = manager.get_statistics()
        
        assert stats['total_calls'] == 1
        assert stats['successful_calls'] == 1
    
    @pytest.mark.asyncio
    async def test_tracking_disabled(self):
        """Test that tracking respects enable flag."""
        
        class TestAgent:
            def __init__(self):
                self.name = "test_agent"
                self._tracking_enabled = False
                self._tracking_output_dir = tempfile.mkdtemp()
            
            def get_tracking_manager(self):
                return get_tracking_manager(self._tracking_output_dir)
            
            @track_method()
            async def test_method(self, input_text: str) -> str:
                return f"Response: {input_text}"
        
        agent = TestAgent()
        result = await agent.test_method("test")
        
        assert result == "Response: test"
        
        # Check no tracking occurred
        manager = agent.get_tracking_manager()
        stats = manager.get_statistics()
        
        assert stats['total_calls'] == 0
    
    @pytest.mark.asyncio
    async def test_tracking_error_handling(self):
        """Test tracking captures errors."""
        
        class TestAgent:
            def __init__(self):
                self.name = "test_agent"
                self._tracking_enabled = True
                self._tracking_output_dir = tempfile.mkdtemp()
            
            def get_tracking_manager(self):
                return get_tracking_manager(self._tracking_output_dir)
            
            @track_method()
            async def failing_method(self):
                raise ValueError("Test error")
        
        agent = TestAgent()
        
        with pytest.raises(ValueError):
            await agent.failing_method()
        
        # Check tracking recorded the failure
        manager = agent.get_tracking_manager()
        stats = manager.get_statistics()
        
        assert stats['total_calls'] == 1
        assert stats['failed_calls'] == 1


class TestTrackingMixin:
    """Test TrackingMixin functionality."""
    
    def test_mixin_initialization(self):
        """Test TrackingMixin initialization."""
        
        class TestAgent(TrackingMixin):
            def __init__(self):
                TrackingMixin.__init__(
                    self,
                    enable_tracking=True,
                    tracking_output_dir="./test_tracking",
                )
                self.name = "test_agent"
        
        agent = TestAgent()
        
        assert agent._tracking_enabled is True
        assert agent._tracking_output_dir == "./test_tracking"
    
    def test_enable_disable_tracking(self):
        """Test enabling and disabling tracking."""
        
        class TestAgent(TrackingMixin):
            def __init__(self):
                TrackingMixin.__init__(self, enable_tracking=False)
                self.name = "test_agent"
        
        agent = TestAgent()
        
        assert agent._tracking_enabled is False
        
        agent.enable_tracking("./new_dir")
        assert agent._tracking_enabled is True
        assert agent._tracking_output_dir == "./new_dir"
        
        agent.disable_tracking()
        assert agent._tracking_enabled is False
    
    def test_export_tracking_data(self):
        """Test export methods on TrackingMixin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            
            class TestAgent(TrackingMixin):
                def __init__(self):
                    TrackingMixin.__init__(
                        self,
                        enable_tracking=True,
                        tracking_output_dir=tmpdir,
                    )
                    self.name = "test_agent"
            
            agent = TestAgent()
            
            # Add a dummy record
            manager = agent.get_tracking_manager()
            session_id = manager.start_session(agent.name)
            
            now = datetime.now()
            input_data = MethodInputSchema(
                timestamp=now,
                method_name="test",
                agent_name=agent.name,
                call_id="test-id",
            )
            output_data = MethodOutputSchema(
                timestamp=now,
                method_name="test",
                agent_name=agent.name,
                duration_seconds=1.0,
                success=True,
                call_id="test-id",
            )
            manager.record_call(agent.name, "test", input_data, output_data, session_id)
            
            # Test CSV export
            csv_path = agent.export_tracking_data(format='csv')
            assert csv_path.exists()
            
            # Test JSON export
            json_path = agent.export_tracking_data(format='json')
            assert json_path.exists()
    
    def test_get_tracking_statistics(self):
        """Test getting statistics from TrackingMixin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            
            class TestAgent(TrackingMixin):
                def __init__(self):
                    TrackingMixin.__init__(
                        self,
                        enable_tracking=True,
                        tracking_output_dir=tmpdir,
                    )
                    self.name = "test_agent"
            
            agent = TestAgent()
            
            # Get stats (should be empty)
            stats = agent.get_tracking_statistics()
            
            assert stats['total_calls'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

