import json
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

from click import Context
from click.formatting import HelpFormatter
from rakam_systems_tools.evaluation.schema import MetricDiff
from rich.panel import Panel
from rich.text import Text
from typer import rich_utils, secho, echo, style, Exit
from typer.core import TyperGroup


def _print_and_save(
    resp: dict,
    pretty: bool,
    out: Optional[Path],
    overwrite: bool,
) -> None:
    if pretty:
        echo(style("📊 Result:", bold=True))
        pprint(resp)
    else:
        echo(resp)

    if out is None:
        return

    if out.exists() and not overwrite:
        echo(
            f"❌ File already exists: {out} (use --overwrite to replace)")
        raise Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2, ensure_ascii=False)

    echo(f"💾 Result saved to {out}")


def pct_change(a: Optional[float], b: Optional[float]) -> Optional[str]:
    if a is None or b is None or a == 0:
        return None
    return f"{((b - a) / a) * 100:+.2f}%"


def metric_direction(delta: Optional[float]) -> str:
    if delta is None:
        return "unchanged"
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "unchanged"


def print_metric_diff(diff: MetricDiff) -> None:
    secho(f"\nMetric: {diff.metric}", bold=True)

    if diff.status == "added":
        secho(f"+ score: {diff.score_b}", fg="green")
        secho(f"+ threshold: {diff.threshold_b}", fg="green")
        secho(f"+ success: {diff.success_b}", fg="green")
        return

    if diff.status == "removed":
        secho(f"- score: {diff.score_a}", fg="red")
        secho(f"- threshold: {diff.threshold_a}", fg="red")
        secho(f"- success: {diff.success_a}", fg="red")
        return

    # unchanged / changed
    if diff.score_a != diff.score_b:
        direction = metric_direction(diff.delta)
        color = "green" if direction == "improved" else "red"
        pct = pct_change(diff.score_a, diff.score_b)

        secho(f"- score: {diff.score_a}", fg="red")
        secho(
            f"+ score: {diff.score_b}" + (f"   ({pct})" if pct else ""),
            fg=color,
        )
    else:
        secho(f"  score: {diff.score_a}", dim=True)

    if diff.threshold_a != diff.threshold_b:
        secho(f"- threshold: {diff.threshold_a}", fg="red")
        secho(f"+ threshold: {diff.threshold_b}", fg="green")
    else:
        secho(f"  threshold: {diff.threshold_a}", dim=True)

    if diff.success_a != diff.success_b:
        secho(f"- success: {diff.success_a}", fg="red")
        secho(f"+ success: {diff.success_b}", fg="green")
    else:
        secho(f"  success: {diff.success_a}", dim=True)


def summarize(metrics: List[MetricDiff]) -> Dict[str, List[str]]:
    """
    Returns metric names grouped by category.
    """
    summary: Dict[str, List[str]] = {
        "improved": [],
        "regressed": [],
        "unchanged": [],
        "added": [],
        "removed": [],
    }

    for m in metrics:
        if m.status == "added":
            summary["added"].append(m.metric)
            continue

        if m.status == "removed":
            summary["removed"].append(m.metric)
            continue

        direction = metric_direction(m.delta)
        summary[direction].append(m.metric)

    return summary


def _fmt(metrics: List[str]) -> str:
    return ", ".join(metrics) if metrics else "-"


def print_summary(metrics: List[MetricDiff]) -> None:
    summary = summarize(metrics)

    rows = [
        ("↑ Improved", "improved", "green"),
        ("↓ Regressed", "regressed", "red"),
        ("± Unchanged", "unchanged", None),
        ("+ Added.", "added", "green"),
        ("- Removed.", "removed", "red"),
    ]

    secho("\nSummary:", bold=True)
    secho(
        "  | Status       | # | Metrics                |",
        dim=True,
    )
    secho(
        "  |--------------|---|------------------------|",
        dim=True,
    )

    for label, key, color in rows:
        count = len(summary[key])
        metrics_str = _fmt(summary[key])

        line = f"  | {label:<12} | {count:<1} | {metrics_str:<22} |"

        if color:
            secho(line, fg=color)
        else:
            secho(line, dim=True)


def pretty_print_comparison(resp: Any, summary_only: bool = False) -> None:
    if not summary_only:
        for metric in resp.metrics:
            print_metric_diff(metric)
        return

    print_summary(resp.metrics)


def serialize_for_diff(obj: dict) -> str:
    """
    Stable, git-friendly JSON representation
    """
    return (
        json.dumps(
            obj,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def git_diff(
    a_text: str,
    b_text: str,
    *,
    label_a: str,
    label_b: str,
    side_by_side: bool = False,
) -> None:
    """
    Show diff between two text blobs.

    - side_by_side: uses git difftool with vimdiff (interactive)
    - fallback to git diff -U3 or difflib if git is not available
    """
    import shutil
    import subprocess
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import typer

    git = shutil.which("git")
    vimdiff = shutil.which("vimdiff")

    with TemporaryDirectory() as tmp:
        a = Path(tmp) / label_a
        b = Path(tmp) / label_b

        a.write_text(a_text)
        b.write_text(b_text)

        # --- Side-by-side with vimdiff ---
        if side_by_side:
            if not git:
                typer.secho(
                    "❌ Git is required for side-by-side diffs", fg="red", bold=True
                )
                return
            if not vimdiff:
                typer.secho(
                    "❌ Vimdiff is not installed. Please install vim or vimdiff to use side-by-side mode.",
                    fg="red",
                    bold=True,
                )
                return

            cmd = [
                git,
                "difftool",
                "--no-index",
                "--tool=vimdiff",
                "--no-prompt",  # skip Y/n prompt
                str(a),
                str(b),
            ]

            subprocess.run(cmd, check=False)
            return

        # (default)
        if git:
            cmd = [
                git,
                "diff",
                "--no-index",
                "--color=always",
                "-U3",
                str(a),
                str(b),
            ]
            subprocess.run(cmd, check=False)
            return

        #  Fallback
        import difflib

        diff = difflib.unified_diff(
            a_text.splitlines(),
            b_text.splitlines(),
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
        )
        for line in diff:
            typer.echo(line)


class OrderedHelpGroup(TyperGroup):
    def format_help(self, ctx: Context, formatter) -> None:

        console = rich_utils._get_rich_console()
        term_width = console.width

        usage = ctx.command.get_usage(ctx)

        sections = []

        sections.append(Panel(
            Text(usage, style="bold"),
            title="Usage",
            border_style="cyan",
            width=term_width,
        ))

        cmd_formatter = HelpFormatter(width=term_width, max_width=term_width)
        self.format_commands(ctx, cmd_formatter)
        commands_output = cmd_formatter.getvalue().strip()

        if commands_output:
            lines = [l.strip() for l in commands_output.replace(
                "Commands:", "").strip().split("\n") if l.strip()]
            commands_output = "\n".join(lines)
            sections.append(Panel(
                Text(commands_output),
                title="Commands",
                border_style="green",
                width=term_width,
            ))

        opt_formatter = HelpFormatter(width=term_width, max_width=term_width)
        TyperGroup.format_options(self, ctx, opt_formatter)
        options_output = opt_formatter.getvalue().strip()

        if options_output:
            lines = []
            inside_option = False
            for line in options_output.replace("Options:", "").strip().split("\n"):
                stripped = line.strip()
                if not stripped:
                    inside_option = False
                    continue
                if stripped.startswith("--") or stripped.startswith("-"):
                    inside_option = True
                    lines.append(stripped)
                elif inside_option:
                    lines.append(stripped)
            options_output = "\n".join(lines)
            if options_output:
                sections.append(Panel(
                    Text(options_output),
                    title="Options",
                    border_style="yellow",
                    width=term_width,
                ))

        for section in sections:
            console.print(section)
