from typing import List
from rakam_systems_cli.cli import app
from typer.testing import CliRunner
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from pytest import CaptureFixture

from rakam_systems_cli.cli import metrics


def test_metrics_no_metrics_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "eval.py"
    file.write_text("def foo(): pass")

    monkeypatch.setattr(
        "rakam_systems_cli.cli.find_decorated_functions",
        lambda *_: [],
    )

    with pytest.raises(typer.Exit) as exc:
        metrics(directory=tmp_path, recursive=False)

    assert exc.value.exit_code == 0


def test_metrics_finds_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    file = tmp_path / "eval.py"
    file.write_text("")

    monkeypatch.setattr(
        "rakam_systems_cli.cli.find_decorated_functions",
        lambda *_: ["run_eval"],
    )

    fake_module = SimpleNamespace(
        run_eval=lambda: SimpleNamespace(
            metrics=[
                SimpleNamespace(type="accuracy", name=None),
                SimpleNamespace(type="f1", name="f1_v2"),
            ]
        )
    )

    monkeypatch.setattr(
        "rakam_systems_cli.cli.load_module_from_path",
        lambda _: fake_module,
    )

    metrics(directory=tmp_path, recursive=False)

    out = capsys.readouterr().out

    assert "accuracy" in out
    assert "f1" in out
    assert "unique metrics found" in out


runner = CliRunner()


def test_compare_summary_and_side_by_side_conflict() -> None:
    result = runner.invoke(
        app,
        ["compare", "--summary", "--side-by-side", "--id", "1", "--id", "2"],
    )

    assert result.exit_code == 1
    assert "--summary and --side-by-side cannot be used together" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["compare", "--id", "1"],  # one target
        ["compare", "--id", "1", "--id", "2", "--id", "3"],  # three targets
        ["compare"],  # none
    ],
)
def test_compare_requires_exactly_two_targets(args: List[str]) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "Provide exactly two targets" in result.output


def test_compare_summary_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = Mock()
    fake_client.compare_testcases.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        "rakam_systems_cli.cli.DeepEvalClient",
        lambda: fake_client,
    )

    result = runner.invoke(
        app,
        ["compare", "--summary", "--id", "1", "--id", "2"],
    )

    assert result.exit_code == 1
    assert "Request failed" in result.output


def test_compare_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = Mock()

    monkeypatch.setattr(
        "rakam_systems_cli.cli.DeepEvalClient",
        lambda: fake_client,
    )

    monkeypatch.setattr(
        "rakam_systems_cli.cli.fetch_run",
        lambda *_, run_id=None, tag=None, **__: (
            {"foo": "a"} if run_id == 1 else {"foo": "b"},
            run_id or tag,
        ),
    )

    serialize_mock = Mock(side_effect=lambda x: f"{x}")
    monkeypatch.setattr(
        "rakam_systems_cli.cli.serialize_for_diff",
        serialize_mock,
    )

    git_diff_mock = Mock()
    monkeypatch.setattr(
        "rakam_systems_cli.cli.git_diff",
        git_diff_mock,
    )

    result = runner.invoke(
        app,
        ["compare", "--id", "1", "--id", "2"],
    )

    assert result.exit_code == 0

    git_diff_mock.assert_called_once()


def test_compare_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rakam_systems_cli.cli.fetch_run",
        Mock(side_effect=RuntimeError("fetch failed")),
    )

    monkeypatch.setattr(
        "rakam_systems_cli.cli.DeepEvalClient",
        lambda: Mock(),
    )

    result = runner.invoke(
        app,
        ["compare", "--id", "1", "--id", "2"],
    )

    assert result.exit_code == 1
    assert "Fetch failed" in result.output


def test_compare_summary_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_metrics = [SimpleNamespace(
        metric="accuracy", status="added", delta=None)]
    fake_resp = {"metrics": fake_metrics}

    fake_client = Mock()
    fake_client.compare_testcases.return_value = fake_resp

    monkeypatch.setattr(
        "rakam_systems_cli.cli.DeepEvalClient",
        lambda: fake_client,
    )

    # 🔑 Bypass Pydantic validation entirely
    fake_comparison = SimpleNamespace(metrics=fake_metrics)
    monkeypatch.setattr(
        "rakam_systems_cli.cli.TestCaseComparison",
        lambda **_: fake_comparison,
    )

    pretty_mock = Mock()
    monkeypatch.setattr(
        "rakam_systems_cli.cli.pretty_print_comparison",
        pretty_mock,
    )

    result = runner.invoke(
        app,
        ["compare", "--summary", "--id", "1", "--tag", "baseline"],
    )

    assert result.exit_code == 0

    fake_client.compare_testcases.assert_called_once_with(
        testcase_a_id=1,
        testcase_b_tag="baseline",
        raise_exception=True,
    )

    pretty_mock.assert_called_once_with(
        fake_comparison,
        summary_only=True,
    )
