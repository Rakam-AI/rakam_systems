"""Tests for the AI gateway usage-metering hook contract (metering.py)."""
from rakam_systems_core.metering import NoopUsageHook, UsageHook


def test_noop_usage_hook_conforms_and_is_a_noop():
    hook = NoopUsageHook()
    # UsageHook is @runtime_checkable, so structural conformance is verifiable.
    assert isinstance(hook, UsageHook)
    assert (
        hook.record(ref="openai:gpt-4o", kind="chat", usage=None, latency_ms=1.0)
        is None
    )


def test_arbitrary_object_with_record_conforms():
    # The Protocol is structural: anything exposing a matching ``record`` passes.
    class _Custom:
        def record(self, *, ref, kind, usage, latency_ms):
            return None

    assert isinstance(_Custom(), UsageHook)


def test_object_without_record_does_not_conform():
    class _NotAHook:
        pass

    assert not isinstance(_NotAHook(), UsageHook)
