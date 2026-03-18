"""Unit tests for evaluation tracking backends and factory."""

import sys
from types import ModuleType
import builtins
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal mock SDK modules
# ---------------------------------------------------------------------------

def _make_langfuse_module():
    """Return a mock langfuse module with a Langfuse class."""
    mod = ModuleType("langfuse")
    lf_instance = MagicMock()
    lf_class = MagicMock(return_value=lf_instance)
    mod.Langfuse = lf_class

    # propagate_attributes is a context manager — mock it so `with propagate_attributes(...)`
    # works without hitting the real Langfuse OTel machinery.
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=None)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    mod.propagate_attributes = MagicMock(return_value=ctx_mgr)

    sys.modules["langfuse"] = mod
    return mod, lf_instance


def _make_mlflow_module():
    """Return a mock mlflow module."""
    mod = ModuleType("mlflow")
    mod.set_tracking_uri = MagicMock()
    mod.start_trace = MagicMock()
    mod.start_span = MagicMock()
    mod.set_tag = MagicMock()
    mod.update_current_trace = MagicMock()
    mod.search_traces = MagicMock()
    mod.get_trace = MagicMock()
    mod.log_feedback = MagicMock()
    mod.genai = MagicMock()
    sys.modules["mlflow"] = mod
    return mod


# ---------------------------------------------------------------------------
# LangfuseTracker tests
# ---------------------------------------------------------------------------

class TestLangfuseTracker:
    def setup_method(self):
        # Ensure fresh mock for each test
        self.lf_mod, self.lf_instance = _make_langfuse_module()
        # Import after patching sys.modules
        from rakam_systems_tools.evaluation.observability.langfuse import LangfuseTracker
        self.tracker = LangfuseTracker(
            public_key="pk-test", secret_key="sk-test", host="http://localhost"
        )
        self.client = self.lf_instance

    def test_log_trace_returns_id(self):
        # v4 API: start_observation returns a span with a .trace_id attribute
        fake_span = MagicMock()
        fake_span.trace_id = "trace-123"
        self.client.start_observation.return_value = fake_span

        result = self.tracker.log_trace(
            name="test",
            input={"q": "hello"},
            output={"a": "world"},
        )

        assert result == "trace-123"
        call_kwargs = self.client.start_observation.call_args[1]
        assert call_kwargs["name"] == "test"
        # No usage → as_type defaults to "span"
        assert call_kwargs["as_type"] == "span"
        assert call_kwargs["usage_details"] is None
        fake_span.end.assert_called_once()

    def test_log_trace_with_usage(self):
        fake_span = MagicMock()
        fake_span.trace_id = "trace-usage"
        self.client.start_observation.return_value = fake_span

        self.tracker.log_trace(
            name="test",
            input={},
            output={},
            usage={
                "model": "gpt-4o-mini",
                "input_tokens": 100,
                "output_tokens": 50,
                "input_cost": 0.000015,
                "output_cost": 0.00003,
            },
        )

        call_kwargs = self.client.start_observation.call_args[1]
        assert call_kwargs["as_type"] == "generation"
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["usage_details"] == {"input": 100, "output": 50, "total": 150}
        assert call_kwargs["cost_details"]["input"] == pytest.approx(0.000015)
        assert call_kwargs["cost_details"]["output"] == pytest.approx(0.00003)

    def test_fetch_traces(self):
        item = MagicMock()
        item.__dict__ = {"id": "t1", "name": "test"}
        response = MagicMock()
        response.data = [item]
        self.client.api.trace.list.return_value = response

        results = self.tracker.fetch_traces(name="test", limit=10)

        self.client.api.trace.list.assert_called_once_with(name="test", tags=None, limit=10, session_id=None, user_id=None)
        assert len(results) == 1

    def test_get_trace(self):
        from types import SimpleNamespace

        trace = SimpleNamespace(id="t1")
        self.client.api.trace.get.return_value = trace

        result = self.tracker.get_trace("t1")

        self.client.api.trace.get.assert_called_once_with("t1")
        assert result["id"] == "t1"

    def test_log_score(self):
        # v4 API: create_score replaces the old score() method
        self.tracker.log_score("t1", "quality", 0.9, comment="good")
        self.client.create_score.assert_called_once_with(
            trace_id="t1", name="quality", value=0.9, comment="good"
        )

    def test_create_dataset(self):
        ds = MagicMock()
        ds.name = "my-dataset"
        self.client.create_dataset.return_value = ds

        result = self.tracker.create_dataset("my-dataset", description="desc")

        assert result == "my-dataset"
        self.client.create_dataset.assert_called_once_with(
            name="my-dataset", description="desc", metadata=None
        )

    def test_add_dataset_item(self):
        item = MagicMock()
        item.id = "item-1"
        self.client.create_dataset_item.return_value = item

        result = self.tracker.add_dataset_item(
            "my-dataset", input={"q": "hi"}, expected_output={"a": "hello"}
        )

        assert result == "item-1"

    def test_get_dataset(self):
        from types import SimpleNamespace

        ds = SimpleNamespace(name="my-dataset")
        self.client.get_dataset.return_value = ds

        result = self.tracker.get_dataset("my-dataset")

        assert result["name"] == "my-dataset"

    def test_list_datasets(self):
        item = MagicMock()
        item.__dict__ = {"name": "ds1"}
        response = MagicMock()
        response.data = [item]
        self.client.api.datasets.list.return_value = response

        results = self.tracker.list_datasets()

        assert len(results) == 1

    def test_log_trace_with_session_id(self):
        fake_span = MagicMock()
        fake_span.trace_id = "trace-sess"
        self.client.start_observation.return_value = fake_span

        result = self.tracker.log_trace(
            name="test", input={}, output={}, session_id="session-1", user_id="user-42"
        )

        assert result == "trace-sess"
        # _update_trace_attributes is called internally; just verify no exception
        # and that span.end() was reached
        fake_span.end.assert_called_once()

    def test_get_session(self):
        from types import SimpleNamespace

        session = SimpleNamespace(session_id="s1", traces=[])
        self.client.api.sessions.get.return_value = session

        result = self.tracker.get_session("s1")

        self.client.api.sessions.get.assert_called_once_with("s1")

    def test_list_sessions(self):
        from types import SimpleNamespace

        item = SimpleNamespace(id="s1")
        response = MagicMock()
        response.data = [item]
        self.client.api.sessions.list.return_value = response

        results = self.tracker.list_sessions(limit=5)

        self.client.api.sessions.list.assert_called_once_with(limit=5)
        assert len(results) == 1

    def test_evaluate_traces_raises(self):
        with pytest.raises(NotImplementedError):
            self.tracker.evaluate_traces(["t1"], scorers=[])


# ---------------------------------------------------------------------------
# MLflowTracker tests
# ---------------------------------------------------------------------------

class TestMLflowTracker:
    def setup_method(self):
        self.mlflow_mod = _make_mlflow_module()
        from rakam_systems_tools.evaluation.observability.mlflow import MLflowTracker
        self.tracker = MLflowTracker(
            tracking_uri="http://localhost:5000", experiment_id="exp-1"
        )

    def test_log_trace_returns_id(self):
        trace_ctx = MagicMock()
        trace_ctx.info.request_id = "run-abc"
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        trace_ctx.__enter__ = MagicMock(return_value=trace_ctx)
        trace_ctx.__exit__ = MagicMock(return_value=False)
        self.mlflow_mod.start_trace.return_value = trace_ctx
        self.mlflow_mod.start_span.return_value = span_ctx

        result = self.tracker.log_trace(
            name="run1", input={"q": "hi"}, output={"a": "bye"}
        )

        assert result == "run-abc"

    def test_fetch_traces(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = [{"request_id": "t1"}]
        self.mlflow_mod.search_traces.return_value = fake_df

        results = self.tracker.fetch_traces(name="run1", limit=5)

        self.mlflow_mod.search_traces.assert_called_once_with(
            experiment_ids=["exp-1"],
            filter_string="name = 'run1'",
            max_results=5,
        )

    def test_fetch_traces_no_filter(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = []
        self.mlflow_mod.search_traces.return_value = fake_df

        self.tracker.fetch_traces(limit=10)

        self.mlflow_mod.search_traces.assert_called_once_with(
            experiment_ids=["exp-1"],
            filter_string=None,
            max_results=10,
        )

    def test_get_trace(self):
        from types import SimpleNamespace

        trace = SimpleNamespace(request_id="t1")
        self.mlflow_mod.get_trace.return_value = trace

        result = self.tracker.get_trace("t1")

        self.mlflow_mod.get_trace.assert_called_once_with("t1")
        assert result["request_id"] == "t1"

    def test_log_score(self):
        self.tracker.log_score("t1", "quality", 0.8, comment="ok", source_type="LLM_JUDGE")
        self.mlflow_mod.log_feedback.assert_called_once_with(
            trace_id="t1",
            name="quality",
            value=0.8,
            rationale="ok",
            source="llm_judge",
        )

    def test_log_trace_with_usage(self):
        trace_ctx = MagicMock()
        trace_ctx.info.request_id = "run-usage"
        span_mock = MagicMock()
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=span_mock)
        span_ctx.__exit__ = MagicMock(return_value=False)
        trace_ctx.__enter__ = MagicMock(return_value=trace_ctx)
        trace_ctx.__exit__ = MagicMock(return_value=False)
        self.mlflow_mod.start_trace.return_value = trace_ctx
        self.mlflow_mod.start_span.return_value = span_ctx

        self.tracker.log_trace(
            name="run1", input={}, output={},
            usage={"model": "gpt-4o", "input_tokens": 80, "output_tokens": 40, "input_cost": 0.001, "output_cost": 0.002},
        )

        # set_attribute (singular) called with correct keys
        calls = {c.args[0]: c.args[1] for c in span_mock.set_attribute.call_args_list}
        assert "mlflow.chat.tokenUsage" in calls
        assert calls["mlflow.chat.tokenUsage"]["input_tokens"] == 80
        assert calls["mlflow.chat.tokenUsage"]["output_tokens"] == 40
        assert calls["mlflow.chat.tokenUsage"]["total_tokens"] == 120
        assert calls.get("mlflow.llm.model") == "gpt-4o"

    def test_log_trace_with_session_id(self):
        trace_ctx = MagicMock()
        trace_ctx.info.request_id = "run-sess"
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        trace_ctx.__enter__ = MagicMock(return_value=trace_ctx)
        trace_ctx.__exit__ = MagicMock(return_value=False)
        self.mlflow_mod.start_trace.return_value = trace_ctx
        self.mlflow_mod.start_span.return_value = span_ctx

        result = self.tracker.log_trace(
            name="run1", input={}, output={}, session_id="session-1", user_id="user-42"
        )

        assert result == "run-sess"
        # session_id and user_id must be set via metadata, not tags
        call_kwargs = self.mlflow_mod.update_current_trace.call_args[1]
        meta = call_kwargs.get("metadata", {})
        assert meta.get("mlflow.trace.session") == "session-1"
        assert meta.get("mlflow.trace.user") == "user-42"

    def test_log_trace_with_user_id_only(self):
        trace_ctx = MagicMock()
        trace_ctx.info.request_id = "run-user"
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        trace_ctx.__enter__ = MagicMock(return_value=trace_ctx)
        trace_ctx.__exit__ = MagicMock(return_value=False)
        self.mlflow_mod.start_trace.return_value = trace_ctx
        self.mlflow_mod.start_span.return_value = span_ctx

        self.tracker.log_trace(name="run1", input={}, output={}, user_id="user-99")

        call_kwargs = self.mlflow_mod.update_current_trace.call_args[1]
        meta = call_kwargs.get("metadata", {})
        assert meta.get("mlflow.trace.user") == "user-99"
        assert "mlflow.trace.session" not in meta

    def test_fetch_traces_with_session_id(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = []
        self.mlflow_mod.search_traces.return_value = fake_df

        self.tracker.fetch_traces(session_id="s1", limit=5)

        call_kwargs = self.mlflow_mod.search_traces.call_args[1]
        fstr = call_kwargs.get("filter_string") or ""
        assert "mlflow.trace.session" in fstr
        assert "s1" in fstr

    def test_fetch_traces_with_user_id(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = []
        self.mlflow_mod.search_traces.return_value = fake_df

        self.tracker.fetch_traces(user_id="user-42", limit=5)

        call_kwargs = self.mlflow_mod.search_traces.call_args[1]
        fstr = call_kwargs.get("filter_string") or ""
        assert "mlflow.trace.user" in fstr
        assert "user-42" in fstr

    def test_get_session(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = [{"request_id": "t1"}]
        self.mlflow_mod.search_traces.return_value = fake_df

        result = self.tracker.get_session("s1")

        assert result["session_id"] == "s1"
        assert "traces" in result

    def test_create_dataset_raises(self):
        with pytest.raises(NotImplementedError):
            self.tracker.create_dataset("ds")

    def test_add_dataset_item_raises(self):
        with pytest.raises(NotImplementedError):
            self.tracker.add_dataset_item("ds", input={})

    def test_get_dataset_raises(self):
        with pytest.raises(NotImplementedError):
            self.tracker.get_dataset("ds")

    def test_list_datasets_raises(self):
        with pytest.raises(NotImplementedError):
            self.tracker.list_datasets()

    def test_evaluate_traces(self):
        results_mock = MagicMock()
        results_mock.to_dict.return_value = [{"trace_id": "t1", "score": 0.9}]
        self.mlflow_mod.genai.evaluate.return_value = results_mock

        results = self.tracker.evaluate_traces(["t1"], scorers=["scorer1"])

        self.mlflow_mod.genai.evaluate.assert_called_once_with(
            data=["t1"], scorers=["scorer1"]
        )
        assert results == [{"trace_id": "t1", "score": 0.9}]


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateTracker:
    def test_create_langfuse_tracker(self):
        _make_langfuse_module()
        from rakam_systems_tools.evaluation.observability.factory import create_tracker
        from rakam_systems_tools.evaluation.observability.langfuse import LangfuseTracker

        tracker = create_tracker("langfuse", public_key="pk", secret_key="sk")
        assert isinstance(tracker, LangfuseTracker)

    def test_create_mlflow_tracker(self):
        _make_mlflow_module()
        from rakam_systems_tools.evaluation.observability.factory import create_tracker
        from rakam_systems_tools.evaluation.observability.mlflow import MLflowTracker

        tracker = create_tracker("mlflow", experiment_id="exp-1")
        assert isinstance(tracker, MLflowTracker)

    def test_unknown_backend_raises(self):
        from rakam_systems_tools.evaluation.observability.factory import create_tracker

        with pytest.raises(ValueError, match="Unknown backend"):
            create_tracker("unknown")  # type: ignore

    def test_missing_langfuse_raises_import_error(self):
        # Simulate langfuse not being installed by making the import fail
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langfuse":
                raise ImportError("No module named 'langfuse'")
            return real_import(name, *args, **kwargs)

        import importlib
        import rakam_systems_tools.evaluation.observability.langfuse as lf_mod

        with patch.object(builtins, "__import__", side_effect=mock_import):
            importlib.reload(lf_mod)
            with pytest.raises(ImportError, match="langfuse is required"):
                lf_mod.LangfuseTracker()
