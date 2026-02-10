# cli.py
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import typer
from dotenv import load_dotenv
from rakam_eval_sdk.client import DeepEvalClient
from rakam_eval_sdk.schema import TestCaseComparison
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from rakam_systems_cli.decorators import eval_run
from rakam_systems_cli.utils.decorator_utils import (
    find_decorated_functions,
    load_module_from_path,
)
from rakam_systems_cli.utils.print import (
    _print_and_save,
    git_diff,
    pretty_print_comparison,
    serialize_for_diff,
)

load_dotenv()
app = typer.Typer(help="CLI tools for evaluation utilities")
console = Console()

# add root of the project to sys.path
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
list_app = typer.Typer(help="List evaluations or runs")
metrics_app = typer.Typer(help="Metrics utilities")

# Sub-apps are registered at the end to control command order


def extract_metric_names(config: Any) -> List[Tuple[str, Optional[str]]]:
    """
    Returns [(type, name)] from EvalConfig / SchemaEvalConfig
    """
    if not hasattr(config, "metrics"):
        return []

    results: List[Tuple[str, Optional[str]]] = []

    for metric in config.metrics or []:
        metric_type = getattr(metric, "type", None)
        metric_name = getattr(metric, "name", None)
        if metric_type:
            results.append((metric_type, metric_name))

    return results


@metrics_app.command("list")
def metrics(
    directory: Path = typer.Argument(
        Path("./eval"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan (default: ./eval)",
    ),
    recursive: bool = typer.Option(
        False,
        "-r",
        "--recursive",
        help="Recursively search for Python files",
    ),
) -> None:
    """
    List all metric types used by loaded eval configs.
    """
    files = directory.rglob("*.py") if recursive else directory.glob("*.py")
    TARGET_DECORATOR = eval_run.__name__

    all_metrics: Set[Tuple[str, Optional[str]]] = set()
    found_any = False

    for file in sorted(files):
        functions = find_decorated_functions(file, TARGET_DECORATOR)
        if not functions:
            continue

        typer.echo(f"\n📄 {file}")

        try:
            module = load_module_from_path(file)
        except Exception as e:
            typer.echo(f"  ❌ Failed to import module: {e}")
            continue

        for fn_name in functions:
            try:
                func = getattr(module, fn_name)
                result = func()

                metrics = extract_metric_names(result)
                if not metrics:
                    continue

                found_any = True
                for metric_type, metric_name in metrics:
                    all_metrics.add((metric_type, metric_name))

                    if metric_name:
                        typer.echo(f"  • {metric_type} (alias: {metric_name})")
                    else:
                        typer.echo(f"  • {metric_type}")

            except Exception as e:
                typer.echo(f"  ❌ Failed to inspect {fn_name}: {e}")

    if not found_any:
        typer.echo("\nNo metrics found.")
        raise typer.Exit(code=0)

    typer.echo(f"\n✅ {len(all_metrics)} unique metrics found")


@list_app.command("evals")
def list_evals(
    directory: Path = typer.Argument(
        Path("./eval"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan (default: ./eval)",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively search for Python files",
    ),
) -> None:
    """
    List evaluations (functions decorated with @eval_run).
    """
    TARGET_DECORATOR = eval_run.__name__
    files = directory.rglob("*.py") if recursive else directory.glob("*.py")

    found = False

    for file in sorted(files):
        functions = find_decorated_functions(file, TARGET_DECORATOR)
        for fn in functions:
            found = True
            typer.echo(f"{file}:{fn}")

    if not found:
        typer.echo("No evaluations found.")


@list_app.command("runs")
def list_runs(
    limit: int = typer.Option(20, "-l", "--limit", help="Max number of runs"),
    offset: int = typer.Option(0, help="Pagination offset"),
) -> None:
    """
    List runs (newest first).
    """
    client = DeepEvalClient()

    response = client.list_evaluation_testcases(
        limit=limit,
        offset=offset,
        raise_exception=True,
    )
    assert response is not None
    items = response.get("items", [])
    total = response.get("total", 0)

    if not items:
        typer.echo("No runs found.")
        return

    typer.echo(f"[id] {'tag':<20}{'label':<20}created_at")

    for run in items:
        run_id = run.get("id")
        label = run.get("label") or "-"
        uid = run.get("tag") or "-"
        created_at = run.get("created_at")

        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass

        typer.echo(f"[{run_id}] {uid:<20} {label:<20} {created_at}")

    shown = offset + len(items)
    if shown < total:
        typer.echo()
        typer.echo(
            f"Showing {shown} of {total} runs. Use --limit to see more.")


@app.command()
def show(
    run_id: Optional[int] = typer.Option(
        None,
        "-i",
        "--id",
        help="Run ID",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "-t",
        "--tag",
        help="Run tag",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Print raw JSON instead of formatted output",
    ),
) -> None:
    """
    Show a run by ID or tag. Without arguments, shows the most recent run.
    """

    if run_id and tag:
        raise typer.BadParameter("Provide only one of --id or --tag")

    client = DeepEvalClient()

    # If no arguments, fetch the most recent run
    if not run_id and not tag:
        response = client.list_evaluation_testcases(
            limit=1, offset=0, raise_exception=True
        )
        if not response or not response.get("items"):
            console.print(
                Panel(
                    "No runs found",
                    title="Error",
                    style="red",
                )
            )
            raise typer.Exit(code=1)
        run_id = response["items"][0]["id"]
        identifier = f"run_id={run_id} (latest)"
        assert isinstance(run_id, int)
        result = client.get_evaluation_testcase_by_id(run_id)
    elif run_id:
        result = client.get_evaluation_testcase_by_id(run_id)
        identifier = f"run_id={run_id}"
    else:
        assert tag is not None
        result = client.get_evaluation_testcase_by_tag(tag)
        identifier = f"tag={tag}"

    if not result:
        console.print(
            Panel(
                f"No response received for {identifier}",
                title="Error",
                style="red",
            )
        )
        raise typer.Exit(code=1)

    if isinstance(result, dict) and result.get("error"):
        console.print(
            Panel(
                result["error"],
                title="Error",
                style="red",
            )
        )
        raise typer.Exit(code=1)

    if raw:
        console.print(Pretty(result))
        raise typer.Exit()

    console.print(
        Panel.fit(
            Pretty(result),
            title="Run",
            subtitle=identifier,
        )
    )


def validate_eval_result(result: Any, fn_name: str) -> str:
    eval_config = getattr(result, "__eval_config__", None)

    if not isinstance(eval_config, str):
        expected = "EvalConfig or SchemaEvalConfig"
        actual = type(result).__name__

        typer.echo(
            f"    ❌ Invalid return type from `{fn_name}`\n"
            f"       Expected: {expected}\n"
            f"       Got: {actual}"
        )
        return ""

    return eval_config


@app.command()
def run(
    directory: Path = typer.Argument(
        Path("./eval"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan (default: ./eval)",
    ),
    recursive: bool = typer.Option(
        False,
        "-r",
        "--recursive",
        help="Recursively search for Python files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only list functions without executing them",
    ),
    save_runs: bool = typer.Option(
        False,
        "--save-runs",
        help="Save each run result to a JSON file",
    ),
    output_dir: Path = typer.Option(
        Path("./eval_runs"),
        "--output-dir",
        help="Directory where run results are saved",
    ),
) -> None:
    """
    Execute evaluations (functions decorated with @eval_run).
    """
    files = directory.rglob("*.py") if recursive else directory.glob("*.py")
    TARGET_DECORATOR = eval_run.__name__

    executed_any = False

    if save_runs and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(files):
        functions = find_decorated_functions(file, TARGET_DECORATOR)
        if not functions:
            continue

        typer.echo(f"\n📄 {file}")

        module = None
        if not dry_run:
            try:
                module = load_module_from_path(file)
            except Exception as e:
                typer.echo(f"  ❌ Failed to import module: {e}")
                continue

        for fn_name in functions:
            typer.echo(f"  ▶ {fn_name}")

            if dry_run:
                continue

            try:
                func = getattr(module, fn_name)
                result = func()

                eval_type = validate_eval_result(result, fn_name)
                if not eval_type:
                    continue

                client = DeepEvalClient()

                if eval_type == "text_eval":
                    resp = client.text_eval(config=result)
                else:
                    resp = client.schema_eval(config=result)

                typer.echo(f"{resp}")
                executed_any = True
                typer.echo(f"    ✅ Returned {type(result).__name__}")

                if save_runs:
                    run_id = (
                        resp["id"]
                        if resp is not None and "id" in resp
                        else uuid.uuid4().hex[:8]
                    )

                    output_path = output_dir / f"run_{fn_name}_{run_id}.json"

                    def to_json_safe(obj: Any) -> Any:
                        if hasattr(obj, "model_dump"):
                            return obj.model_dump()
                        if hasattr(obj, "dict"):
                            return obj.dict()
                        return obj

                    with output_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            to_json_safe(resp),
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                    typer.echo(f"    💾 Saved run → {output_path}")

            except Exception as e:
                typer.echo(f"    ❌ Execution failed: {e}")

    if not executed_any and not dry_run:
        typer.echo("\nNo evaluations executed.")


def fetch_run(
    client: DeepEvalClient,
    *,
    run_id: Optional[int],
    tag: Optional[str],
) -> Tuple[dict, str]:
    """
    Fetch a single run by id or tag.
    Returns (payload, identifier)
    """
    if run_id is not None:
        result = client.get_evaluation_testcase_by_id(run_id)
        identifier = f"run_id={run_id}"
    else:
        assert tag is not None
        result = client.get_evaluation_testcase_by_tag(tag)
        identifier = f"tag={tag}"

    if not result:
        raise RuntimeError(f"No data returned for {identifier}")

    return result, identifier


@app.command()
def compare(
    tag: List[str] = typer.Option(
        [],
        "-t",
        "--tag",
        help="Run tag",
    ),
    run_id: List[int] = typer.Option(
        [],
        "-i",
        "--id",
        help="Run ID",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Show summary diff only",
    ),
    side_by_side: bool = typer.Option(
        False,
        "--side-by-side",
        help="Show side-by-side diff (git)",
    ),
) -> None:
    """
    Compare two evaluation runs.

    Default: unified git diff
    """

    if summary and side_by_side:
        typer.secho(
            "❌ --summary and --side-by-side cannot be used together",
            fg="red",
            bold=True,
        )
        raise typer.Exit(code=1)

    targets: List[Tuple[str, Union[str, int]]] = []

    for r in run_id:
        targets.append(("run", r))
    for t in tag:
        targets.append(("tag", t))

    if len(targets) != 2:
        typer.secho(
            "❌ Provide exactly two targets using --id and/or --tag",
            fg="red",
            bold=True,
        )
        raise typer.Exit(code=1)

    client = DeepEvalClient()
    # Summary mode (reduced payload)
    (type_a, value_a), (type_b, value_b) = targets
    if summary:
        kwargs: Dict[str, Any] = {"raise_exception": True}
        if type_a == "run":
            kwargs["testcase_a_id"] = value_a
        else:
            kwargs["testcase_a_tag"] = value_a

        if type_b == "run":
            kwargs["testcase_b_id"] = value_b
        else:
            kwargs["testcase_b_tag"] = value_b
        try:
            resp = client.compare_testcases(**kwargs)
        except Exception as e:
            typer.secho(f"❌ Request failed: {e}", fg="red")
            raise typer.Exit(code=1)

        if not resp:
            typer.secho("⚠️ No response received", fg="yellow")
            raise typer.Exit(code=1)
        comparison = TestCaseComparison(**resp)
        pretty_print_comparison(
            comparison,
            summary_only=summary,
        )
        return

    try:
        run_a, id_a = fetch_run(
            client,
            run_id=int(value_a) if type_a == "run" else None,
            tag=str(value_a) if type_a == "tag" else None,
        )
        run_b, id_b = fetch_run(
            client,
            run_id=int(value_b) if type_b == "run" else None,
            tag=str(value_b) if type_b == "tag" else None,
        )
    except Exception as e:
        typer.secho(f"❌ Fetch failed: {e}", fg="red")
        raise typer.Exit(code=1)

    a_text = serialize_for_diff(run_a)
    b_text = serialize_for_diff(run_b)

    git_diff(
        a_text,
        b_text,
        label_a=f"{id_a}.full.json",
        label_b=f"{id_b}.full.json",
        side_by_side=side_by_side,
    )


@app.command(hidden=True)
def compare_label_latest(
    label_a: str = typer.Argument(
        ...,
        help="First label (latest run will be used)",
    ),
    label_b: str = typer.Argument(
        ...,
        help="Second label (latest run will be used)",
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--raw",
        help="Pretty-print the response",
    ),
    raise_exception: bool = typer.Option(
        False,
        "--raise",
        help="Raise HTTP exceptions instead of swallowing them",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out",
        help="Optional file path to save the result as JSON",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite output file if it already exists",
    ),
) -> None:
    """
    Compare the latest runs for two labels.
    """
    client = DeepEvalClient()

    typer.echo(f"🔍 Comparing latest runs: '{label_a}' ↔ '{label_b}'")

    try:
        resp = client.compare_latest_by_labels(
            label_a=label_a,
            label_b=label_b,
            raise_exception=raise_exception,
        )
    except Exception as e:
        typer.echo(f"❌ Request failed: {e}")
        raise typer.Exit(code=1)

    if not resp:
        typer.echo("⚠️ No response received")
        raise typer.Exit(code=1)

    _print_and_save(resp, pretty, out, overwrite)


@app.command(hidden=True)
def compare_last(
    label: str = typer.Argument(
        ...,
        help="Label whose last two runs will be compared",
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--raw",
        help="Pretty-print the response",
    ),
    raise_exception: bool = typer.Option(
        False,
        "--raise",
        help="Raise HTTP exceptions instead of swallowing them",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out",
        help="Optional file path to save the result as JSON",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite output file if it already exists",
    ),
) -> None:
    """
    Compare the last two evaluation runs of a label.
    """
    client = DeepEvalClient()

    typer.echo(f"🔍 Comparing last two runs for label '{label}'")

    try:
        resp = client.compare_last_two_by_label(
            label=label,
            raise_exception=raise_exception,
        )
    except Exception as e:
        typer.echo(f"❌ Request failed: {e}")
        raise typer.Exit(code=1)

    if not resp:
        typer.echo("⚠️ No response received")
        raise typer.Exit(code=1)

    _print_and_save(resp, pretty, out, overwrite)


@app.command("tag")
def tag_command(
    run_id: Optional[int] = typer.Option(
        None,
        "-i",
        "--id",
        help="Run ID",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "-t",
        "--tag",
        help="Tag to assign to the run",
    ),
    delete: Optional[str] = typer.Option(
        None,
        "--delete",
        help="Delete a tag",
    ),
) -> None:
    """
    Assign a tag to a run or delete a tag.
    """

    # --- validation ---
    if delete:
        if run_id or tag:
            typer.echo("❌ --delete cannot be used with --id or --tag")
            raise typer.Exit(code=1)
    else:
        if not run_id or not tag:
            typer.echo("❌ Use --id and --tag together, or --delete")
            raise typer.Exit(code=1)

    client = DeepEvalClient()

    if delete:
        assert run_id is not None

        result = client.update_evaluation_testcase_tag(
            testcase_id=run_id,
            tag=delete,
            raise_exception=True,
        )
        typer.echo("🗑️ Tag deleted successfully")
        typer.echo(f"Tag: {delete}")
        return
    assert run_id is not None
    assert tag is not None
    result = client.update_evaluation_testcase_tag(
        testcase_id=run_id,
        tag=tag,
        raise_exception=True,
    )
    assert result is not None

    typer.echo("✅ Tag assigned successfully")
    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Tag: {result.get('tag')}")


# Register sub-apps in user journey order (after regular commands)
app.add_typer(list_app, name="list")
app.add_typer(metrics_app, name="metrics")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
