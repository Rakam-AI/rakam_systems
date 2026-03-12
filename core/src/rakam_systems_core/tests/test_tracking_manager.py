import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from rakam_systems_core.tracking import TrackingManager


class MockMethodInput:
    def __init__(self, call_id="1", input_text="hello", kwargs=None, timestamp=None):
        self.call_id = call_id
        self.input_text = input_text
        self.kwargs = kwargs or {}
        self.timestamp = timestamp or datetime.now()


class MockMethodOutput:
    def __init__(
        self,
        output_text="world",
        success=True,
        error=None,
        metadata=None,
        duration_seconds=0.5,
        timestamp=None,
    ):
        self.output_text = output_text
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.duration_seconds = duration_seconds
        self.timestamp = timestamp or datetime.now()


class MockCallRecord:
    def __init__(
        self,
        call_id,
        agent_name,
        method_name,
        input_data,
        output_data,
        started_at,
        completed_at,
        duration_seconds,
    ):
        self.call_id = call_id
        self.agent_name = agent_name
        self.method_name = method_name
        self.input_data = input_data
        self.output_data = output_data
        self.started_at = started_at
        self.completed_at = completed_at
        self.duration_seconds = duration_seconds
        self.evaluation_score = None
        self.evaluation_notes = None

    def dict(self):
        return {
            "call_id": self.call_id,
            "agent_name": self.agent_name,
        }


class MockSession:
    def __init__(self, session_id, agent_name, started_at):
        self.session_id = session_id
        self.agent_name = agent_name
        self.started_at = started_at
        self.calls = []
        self.ended = False

    def add_call(self, record):
        self.calls.append(record)

    def end_session(self):
        self.ended = True

    def dict(self):
        return {
            "session_id": self.session_id,
            "calls": len(self.calls),
        }



@pytest.fixture(autouse=True)
def patch_schemas(monkeypatch):
    import rakam_systems_core.tracking as track

    monkeypatch.setattr(track, "MethodInputSchema", MockMethodInput)
    monkeypatch.setattr(track, "MethodOutputSchema", MockMethodOutput)
    monkeypatch.setattr(track, "MethodCallRecordSchema", MockCallRecord)
    monkeypatch.setattr(track, "TrackingSessionSchema", MockSession)



def test_start_and_end_session():
    tm = TrackingManager(output_dir=tempfile.mkdtemp())

    session_id = tm.start_session("agent")

    assert session_id in tm.sessions
    assert tm.current_session_id == session_id

    tm.end_session()
    assert tm.sessions[session_id].ended is True


def test_record_call_and_session_link():
    tm = TrackingManager(output_dir=tempfile.mkdtemp())
    session_id = tm.start_session("agent")

    inp = MockMethodInput()
    out = MockMethodOutput()

    record = tm.record_call("agent", "run", inp, out)

    assert record in tm.call_records
    assert record in tm.sessions[session_id].calls


def test_get_session():
    tm = TrackingManager(output_dir=tempfile.mkdtemp())
    session_id = tm.start_session("agent")

    session = tm.get_session()
    assert session.session_id == session_id


def test_export_to_csv():
    tmpdir = tempfile.mkdtemp()
    tm = TrackingManager(output_dir=tmpdir)

    inp = MockMethodInput(kwargs={"model_settings": {"model": "gpt"}})
    out = MockMethodOutput()

    tm.record_call("agent", "run", inp, out)

    path = tm.export_to_csv("test.csv")
    assert path.exists()

    content = path.read_text()
    assert "agent_name" in content
    assert "run" in content


def test_export_to_json_all_records():
    tmpdir = tempfile.mkdtemp()
    tm = TrackingManager(output_dir=tmpdir)

    inp = MockMethodInput()
    out = MockMethodOutput()

    tm.record_call("agent", "run", inp, out)

    path = tm.export_to_json("data.json")
    data = json.loads(path.read_text())

    assert data["total_records"] == 1


def test_get_statistics():
    tm = TrackingManager(output_dir=tempfile.mkdtemp())

    inp = MockMethodInput()
    out1 = MockMethodOutput(success=True, duration_seconds=1.0)
    out2 = MockMethodOutput(success=False, duration_seconds=2.0)

    tm.record_call("agent", "run", inp, out1)
    tm.record_call("agent", "run", inp, out2)

    stats = tm.get_statistics()

    assert stats["total_calls"] == 2
    assert stats["successful_calls"] == 1
    assert stats["failed_calls"] == 1
    assert stats["success_rate"] == 0.5
    assert stats["average_duration_seconds"] == 1.5
    assert stats["min_duration_seconds"] == 1.0
    assert stats["max_duration_seconds"] == 2.0


def test_get_statistics_empty():
    tm = TrackingManager(output_dir=tempfile.mkdtemp())

    stats = tm.get_statistics()
    assert stats["total_calls"] == 0
