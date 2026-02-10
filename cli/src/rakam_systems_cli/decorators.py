import functools
import os
import sys
import time
from typing import Callable, Dict, Optional, TypeVar, Union, overload

import psutil

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec
P = ParamSpec("P")
R = TypeVar("R")


@overload
def eval_run(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def eval_run(
    func: None = None,
    **decorator_kwargs: Dict[str, object],
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def eval_run(
    func: Optional[Callable[P, R]] = None,
    **decorator_kwargs: Dict[str, object],
) -> Union[
    Callable[P, R],
    Callable[[Callable[P, R]], Callable[P, R]],
]:
    # used as @eval_run
    if callable(func):
        return _wrap(func)

    # used as @eval_run(...)
    def decorator(real_func: Callable[P, R]) -> Callable[P, R]:
        return _wrap(real_func)

    return decorator


def _wrap(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        process = psutil.Process(os.getpid())

        start_time = time.perf_counter()
        start_cpu = process.cpu_times()
        start_mem = process.memory_info().rss

        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            end_cpu = process.cpu_times()
            end_mem = process.memory_info().rss

            elapsed = end_time - start_time
            cpu_used = (end_cpu.user + end_cpu.system) - (
                start_cpu.user + start_cpu.system
            )
            mem_delta_mb = (end_mem - start_mem) / (1024 * 1024)

            print(
                f"[eval_run] {func.__module__}.{func.__name__} | "
                f"time={elapsed:.4f}s | "
                f"cpu={cpu_used:.4f}s | "
                f"mem_delta={mem_delta_mb:.2f}MB"
            )

    return inner
