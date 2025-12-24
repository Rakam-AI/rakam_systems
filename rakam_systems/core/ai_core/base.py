from __future__ import annotations

import contextlib
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple


class BaseComponent(ABC):
    """Minimal, dependency-free lifecycle + evaluation mixin for all components.

    Responsibilities
    - hold a name + config
    - provide setup()/shutdown() lifecycle
    - provide __call__ that auto-setup then delegates to run()
    - provide evaluate() helper for quick smoke tests (no external deps)
    - context manager support to ensure shutdown() on exit
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = dict(config or {})
        self.initialized: bool = False

    # ---------- lifecycle ----------
    def setup(self) -> None:
        """Initialize heavy resources (override in subclasses)."""
        self.initialized = True

    def shutdown(self) -> None:
        """Release resources (override in subclasses)."""
        self.initialized = False

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the primary operation for the component."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.initialized:
            self.setup()
        return self.run(*args, **kwargs)

    # ---------- context manager ----------
    def __enter__(self) -> "BaseComponent":
        if not self.initialized:
            self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always attempt to shutdown, even if exceptions happened
        with contextlib.suppress(Exception):
            self.shutdown()

    # ---------- utility: timed call wrapper ----------
    def timed(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Tuple[Any, float]:
        start = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - start

    # ---------- micro evaluation harness ----------
    def evaluate(
        self,
        methods: Optional[List[str]] = None,
        test_cases: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        metric_fn: Optional[Callable[[Any, Any], float]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate one or more public methods using a tiny harness.

        test_cases format:
        {
          "run": [
             {"args": [...], "kwargs": {...}, "expected": <any>},
             ...
          ],
          "other_method": [...]
        }
        """
        methods = methods or ["run"]
        results: Dict[str, Any] = {}

        for method_name in methods:
            if not hasattr(self, method_name):
                raise AttributeError(
                    f"{self.__class__.__name__} has no method '{method_name}'"
                )
            method = getattr(self, method_name)
            if not callable(method):
                raise TypeError(f"{method_name} is not callable")

            cases = (test_cases or {}).get(method_name, [])
            mres: List[Dict[str, Any]] = []
            if verbose:
                print(
                    f"🔍 Evaluating {self.name}.{method_name} on {len(cases)} case(s)..."
                )

            for i, case in enumerate(cases):
                args = list(case.get("args", []))
                kwargs = dict(case.get("kwargs", {}))
                expected = case.get("expected", None)

                t0 = time.time()
                try:
                    out = method(*args, **kwargs)
                    dt = time.time() - t0
                    score = (
                        metric_fn(out, expected)
                        if (metric_fn and expected is not None)
                        else None
                    )
                    mres.append(
                        {
                            "case": i,
                            "ok": True,  # kept for backwards-compat
                            "success": True,  # <- added for your tests
                            "time": dt,
                            "input": {"args": args, "kwargs": kwargs},
                            "output": out,
                            "expected": expected,
                            "score": score,
                        }
                    )
                except Exception as e:
                    mres.append(
                        {
                            "case": i,
                            "ok": False,  # kept for backwards-compat
                            "success": False,  # <- added for your tests
                            "error": f"{e}",
                            "traceback": traceback.format_exc(),
                        }
                    )
            results[method_name] = mres

        if verbose:
            print(f"✅ Evaluation complete for {self.name}.")
        return results
