"""Tests for the default async fallback on the EmbeddingModel interface."""
import asyncio

from rakam_systems_core.interfaces.embedding_model import EmbeddingModel


class _SyncOnly(EmbeddingModel):
    """A sync-only embedder (like the local sentence_transformer backend)."""

    def __init__(self):
        super().__init__(name="sync_only")
        self.thread_ids = []

    def run(self, texts):
        import threading

        self.thread_ids.append(threading.get_ident())
        return [[float(len(t))] for t in texts]


def test_default_arun_offloads_sync_run_and_matches():
    m = _SyncOnly()
    texts = ["a", "bb", "ccc"]

    async def go():
        # arun must work while an event loop is running — this is the point:
        # a sync-only backend used from an async service handler.
        return await m.arun(texts)

    out = asyncio.run(go())
    assert out == [[1.0], [2.0], [3.0]]
    assert out == m.run(texts)  # same result as the sync path


def test_default_arun_runs_in_a_worker_thread():
    # The default offloads to an executor thread so it never blocks the loop.
    import threading

    m = _SyncOnly()
    main = threading.get_ident()

    asyncio.run(m.arun(["x"]))
    assert m.thread_ids and m.thread_ids[0] != main
