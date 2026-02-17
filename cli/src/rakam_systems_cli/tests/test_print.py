import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import typer

from rakam_systems_cli.utils.print import (
    _print_and_save,
    git_diff,
    metric_direction,
    pct_change,
    pretty_print_comparison,
    print_metric_diff,
    print_summary,
    serialize_for_diff,
    summarize,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 20, "+100.00%"),
        (20, 10, "-50.00%"),
        (10, 10, "+0.00%"),
        (None, 10, None),
        (10, None, None),
        (0, 10, None),
    ],
)
def test_pct_change(
    a: Optional[int], b: Optional[int], expected: Optional[str]
) -> None:
    assert pct_change(a, b) == expected


# -------------------------
# metric_direction
# -------------------------


@pytest.mark.parametrize(
    "delta,expected",
    [
        (1.0, "improved"),
        (-1.0, "regressed"),
        (0.0, "unchanged"),
        (None, "unchanged"),
    ],
)
def test_metric_direction(delta: Optional[float], expected: str) -> None:
    assert metric_direction(delta) == expected


def test_print_and_save_no_output_path(capsys: pytest.CaptureFixture[str]) -> None:
    resp: Dict[str, int] = {"a": 1}

    _print_and_save(resp, pretty=False, out=None, overwrite=False)

    captured = capsys.readouterr()
    assert str(resp) in captured.out


def test_print_and_save_creates_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out: Path = tmp_path / "result.json"
    resp: Dict[str, str] = {"hello": "world"}

    _print_and_save(resp, pretty=False, out=out, overwrite=False)

    assert out.exists()
    assert json.loads(out.read_text()) == resp

    captured = capsys.readouterr()
    assert "Result saved" in captured.out


def test_print_and_save_refuses_overwrite(tmp_path: Path) -> None:
    out: Path = tmp_path / "result.json"
    out.write_text("{}")

    with pytest.raises(typer.Exit):
        _print_and_save({"x": 1}, pretty=False, out=out, overwrite=False)


def test_print_and_save_overwrites(tmp_path: Path) -> None:
    out: Path = tmp_path / "result.json"
    out.write_text("{}")

    _print_and_save({"x": 2}, pretty=False, out=out, overwrite=True)

    assert json.loads(out.read_text()) == {"x": 2}


def make_metric(
    *,
    metric: str,
    status: str = "changed",
    delta: Optional[float] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metric=metric,
        status=status,
        delta=delta,
    )


def test_summarize_groups_all_states() -> None:
    metrics: List[SimpleNamespace] = [
        make_metric(metric="added_metric", status="added"),
        make_metric(metric="removed_metric", status="removed"),
        make_metric(metric="improved_metric", delta=1.0),
        make_metric(metric="regressed_metric", delta=-1.0),
        make_metric(metric="unchanged_zero", delta=0.0),
        make_metric(metric="unchanged_none", delta=None),
    ]

    summary: Dict[str, List[str]] = summarize(
        metrics=metrics)  # type: ignore[arg-type]

    assert summary == {
        "improved": ["improved_metric"],
        "regressed": ["regressed_metric"],
        "unchanged": ["unchanged_zero", "unchanged_none"],
        "added": ["added_metric"],
        "removed": ["removed_metric"],
    }


def test_summarize_empty_metrics() -> None:
    assert summarize(metrics=[]) == {
        "improved": [],
        "regressed": [],
        "unchanged": [],
        "added": [],
        "removed": [],
    }


def test_pretty_print_comparison_summary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics: List[SimpleNamespace] = [
        make_metric(metric="added_metric", status="added"),
        make_metric(metric="improved_metric", delta=1.0),
    ]
    resp: SimpleNamespace = SimpleNamespace(metrics=metrics)

    called: Dict[str, Any] = {}

    def fake_print_summary(arg: Any) -> None:
        called["metrics"] = arg

    import rakam_systems_cli.utils.print as mod

    monkeypatch.setattr(mod, "print_summary", fake_print_summary)

    pretty_print_comparison(resp, summary_only=True)

    assert called["metrics"] == metrics


def test_pretty_print_comparison_full(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics: List[SimpleNamespace] = [
        make_metric(metric="added_metric", status="added"),
        make_metric(metric="improved_metric", delta=1.0),
    ]
    resp: SimpleNamespace = SimpleNamespace(metrics=metrics)

    calls: List[Any] = []

    def fake_print_metric_diff(metric: Any) -> None:
        calls.append(metric)

    import rakam_systems_cli.utils.print as mod

    monkeypatch.setattr(mod, "print_metric_diff", fake_print_metric_diff)

    pretty_print_comparison(resp, summary_only=False)

    assert calls == metrics


# Helper to capture secho calls
def capture_secho(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, dict]]:
    calls: List[Tuple[str, dict]] = []

    def fake_secho(message: str, **kwargs: Any) -> None:
        calls.append((message, kwargs))

    monkeypatch.setattr(
        "rakam_systems_cli.utils.print.secho",
        fake_secho,
    )
    return calls


def test_print_metric_diff_added(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_secho(monkeypatch)

    diff = SimpleNamespace(
        metric="accuracy",
        status="added",
        score_b=0.9,
        threshold_b=0.8,
        success_b=True,
    )

    print_metric_diff(diff)  # type: ignore[arg-type]

    messages = [msg for msg, _ in calls]

    assert any("Metric: accuracy" in m for m in messages)
    assert any("+ score: 0.9" in m for m in messages)
    assert any("+ threshold: 0.8" in m for m in messages)
    assert any("+ success: True" in m for m in messages)


def test_print_metric_diff_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_secho(monkeypatch)

    diff = SimpleNamespace(
        metric="latency",
        status="removed",
        score_a=120,
        threshold_a=100,
        success_a=False,
    )

    print_metric_diff(diff)  # type: ignore[arg-type]

    messages = [msg for msg, _ in calls]

    assert any("Metric: latency" in m for m in messages)
    assert any("- score: 120" in m for m in messages)
    assert any("- threshold: 100" in m for m in messages)
    assert any("- success: False" in m for m in messages)


def test_print_metric_diff_changed_improved(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_secho(monkeypatch)

    diff = SimpleNamespace(
        metric="f1",
        status="changed",
        score_a=0.5,
        score_b=0.8,
        threshold_a=0.7,
        threshold_b=0.7,
        success_a=False,
        success_b=True,
        delta=0.3,
    )

    print_metric_diff(diff)  # type: ignore[arg-type]

    messages = [msg for msg, _ in calls]

    assert any("- score: 0.5" in m for m in messages)
    assert any("+ score: 0.8" in m for m in messages)
    assert any("(+60.00%)" in m for m in messages)
    assert any("threshold: 0.7" in m for m in messages)
    assert any("- success: False" in m for m in messages)
    assert any("+ success: True" in m for m in messages)


def test_print_metric_diff_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_secho(monkeypatch)

    diff = SimpleNamespace(
        metric="precision",
        status="unchanged",
        score_a=0.9,
        score_b=0.9,
        threshold_a=0.8,
        threshold_b=0.8,
        success_a=True,
        success_b=True,
        delta=0.0,
    )

    print_metric_diff(diff)  # type: ignore[arg-type]

    messages = [msg for msg, _ in calls]

    assert any("score: 0.9" in m for m in messages)
    assert any("threshold: 0.8" in m for m in messages)
    assert any("success: True" in m for m in messages)


def test_serialize_for_diff_stable_sorted_output() -> None:
    obj = {
        "b": 1,
        "a": 2,
        "nested": {
            "z": 3,
            "y": 4,
        },
    }

    result: str = serialize_for_diff(obj)

    # must be valid JSON
    parsed = json.loads(result)
    assert parsed == obj

    # sorted keys (a before b)
    assert result.index('"a"') < result.index('"b"')

    # newline at EOF (git-friendly)
    assert result.endswith("\n")


def test_serialize_for_diff_unicode_preserved() -> None:
    obj = {"text": "café 🚀"}

    result: str = serialize_for_diff(obj)

    assert "café" in result
    assert "🚀" in result
    assert "\\u" not in result


def test_git_diff_side_by_side_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("typer.secho", lambda msg, **_: calls.append(msg))

    git_diff(
        "a",
        "b",
        label_a="a.txt",
        label_b="b.txt",
        side_by_side=True,
    )

    assert any("Git is required" in msg for msg in calls)


def test_git_diff_side_by_side_requires_vimdiff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def fake_which(cmd: str) -> Optional[str]:
        return "/usr/bin/git" if cmd == "git" else None

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("typer.secho", lambda msg, **_: calls.append(msg))

    git_diff(
        "a",
        "b",
        label_a="a.txt",
        label_b="b.txt",
        side_by_side=True,
    )

    assert any("Vimdiff is not installed" in msg for msg in calls)


def test_git_diff_side_by_side_runs_difftool(monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: List[List[str]] = []

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, check=False: run_calls.append(cmd),
    )

    git_diff(
        "foo",
        "bar",
        label_a="a.txt",
        label_b="b.txt",
        side_by_side=True,
    )

    assert run_calls
    assert "difftool" in run_calls[0]
    assert "--tool=vimdiff" in run_calls[0]


def test_git_diff_uses_git_diff_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: List[List[str]] = []

    def fake_which(cmd: str) -> Optional[str]:
        return "/usr/bin/git" if cmd == "git" else None

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, check=False: run_calls.append(cmd),
    )

    git_diff(
        "a",
        "b",
        label_a="old.txt",
        label_b="new.txt",
        side_by_side=False,
    )

    assert run_calls
    assert run_calls[0][1] == "diff"
    assert "--no-index" in run_calls[0]
    assert "--color=always" in run_calls[0]


def test_git_diff_fallback_to_difflib(monkeypatch: pytest.MonkeyPatch) -> None:
    output: List[str] = []

    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("typer.echo", lambda line: output.append(line))

    git_diff(
        "line1\nline2",
        "line1\nline3",
        label_a="a.txt",
        label_b="b.txt",
        side_by_side=False,
    )

    assert any("--- a.txt" in line for line in output)
    assert any("+++ b.txt" in line for line in output)
    assert any("-line2" in line for line in output)
    assert any("+line3" in line for line in output)


# Adjust this import to your actual module


@pytest.fixture
def fake_metrics() -> List[SimpleNamespace]:
    return [
        SimpleNamespace(name="accuracy"),
        SimpleNamespace(name="f1"),
    ]


@pytest.fixture
def summary_result() -> Dict:
    return {
        "improved": ["accuracy"],
        "regressed": ["latency"],
        "unchanged": [],
        "added": ["precision", "recall"],
        "removed": ["auc"],
    }


def test_print_summary_outputs_expected_lines(
    monkeypatch: pytest.MonkeyPatch,
    fake_metrics: List,
    summary_result: Dict,
) -> None:
    secho_calls = []

    def fake_secho(*args: Tuple, **kwargs: Dict) -> None:
        secho_calls.append((args, kwargs))

    monkeypatch.setattr(
        "rakam_systems_cli.utils.print.summarize",
        lambda _: summary_result,
    )
    monkeypatch.setattr(
        "rakam_systems_cli.utils.print._fmt",
        lambda metrics: ", ".join(metrics) if metrics else "-",
    )
    monkeypatch.setattr(
        "rakam_systems_cli.utils.print.secho",
        fake_secho,
    )

    print_summary(fake_metrics)

    # --- Header ---
    assert secho_calls[0][0][0] == "\nSummary:"
    assert secho_calls[0][1]["bold"] is True

    assert "Status" in secho_calls[1][0][0]
    assert "Metrics" in secho_calls[1][0][0]

    # --- Rows ---
    expected_rows = [
        ("↑ Improved", 1, "green"),
        ("↓ Regressed", 1, "red"),
        ("± Unchanged", 0, None),
        ("+ Added.", 2, "green"),
        ("- Removed.", 1, "red"),
    ]

    row_calls = secho_calls[3:]  # skip header lines

    for (label, count, color), (args, kwargs) in zip(expected_rows, row_calls):
        line = args[0]

        assert label in line
        assert f"| {count} |" in line

        if color:
            assert kwargs.get("fg") == color
        else:
            assert kwargs.get("dim") is True


def test_print_summary_calls_summarize(
    monkeypatch: pytest.MonkeyPatch, fake_metrics: List
) -> None:
    called = False

    def fake_summarize(metrics: List) -> Dict:
        nonlocal called
        called = True
        return {
            "improved": [],
            "regressed": [],
            "unchanged": [],
            "added": [],
            "removed": [],
        }

    monkeypatch.setattr(
        "rakam_systems_cli.utils.print.summarize",
        fake_summarize,
    )
    monkeypatch.setattr(
        "rakam_systems_cli.utils.print._fmt",
        lambda _: "-",
    )
    monkeypatch.setattr(
        "rakam_systems_cli.utils.print.secho",
        lambda *_, **__: None,
    )

    print_summary(fake_metrics)

    assert called is True
