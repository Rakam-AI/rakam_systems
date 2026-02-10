#!/usr/bin/env python3
"""Pre-commit hook: Check CLI.md command reference is in sync with actual CLI.

Usage:
    python scripts/check_cli_docs.py          # Check and offer to fix
    python scripts/check_cli_docs.py --fix    # Auto-fix without prompting
    python scripts/check_cli_docs.py --check  # Check only, no prompts (for CI)
"""

import os
import subprocess
import sys
from pathlib import Path

# Path relative to repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CLI_MD_PATH = REPO_ROOT / "CLI.md"

# Force consistent terminal width for reproducible output
TERMINAL_WIDTH = "80"

COMMANDS = [
    "list evals",
    "list runs",
    "run",
    "show",
    "compare",
    "tag",
    "metrics list",
]

START_MARKER = "## Command Reference"
END_MARKER = "</details>"


def get_help_output(command: str) -> str:
    """Run rakam_eval <command> --help and capture output.

    Forces 80-column width for consistent output across different terminals.
    """
    env = os.environ.copy()
    env["COLUMNS"] = TERMINAL_WIDTH

    result = subprocess.run(
        ["rakam_eval"] + command.split() + ["--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"Warning: 'rakam_eval {command} --help' failed", file=sys.stderr)
        return f"Error: Could not get help for '{command}'"
    return result.stdout.strip()


def generate_reference() -> str:
    """Generate markdown for all commands."""
    lines = [
        "## Command Reference",
        "",
        "<details>",
        "<summary>Full command reference (click to expand)</summary>",
        "",
    ]

    for command in COMMANDS:
        cmd_name = f"rakam_eval {command}"
        help_text = get_help_output(command)

        lines.append(f"### `{cmd_name}`")
        lines.append("")
        lines.append("```")
        lines.append(help_text)
        lines.append("```")
        lines.append("")

    lines.append("</details>")

    return "\n".join(lines)


def extract_current_reference(content: str) -> str:
    """Extract current command reference section from CLI.md."""
    start = content.find(START_MARKER)
    if start == -1:
        return ""

    end = content.find(END_MARKER, start)
    if end == -1:
        return ""

    return content[start : end + len(END_MARKER)]


def update_cli_md(content: str, new_reference: str) -> str:
    """Replace command reference section in CLI.md content."""
    start = content.find(START_MARKER)
    end = content.find(END_MARKER, start) + len(END_MARKER)

    return content[:start] + new_reference + content[end:]


def normalize(text: str) -> str:
    """Normalize text for comparison (strip trailing whitespace per line)."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def main() -> int:
    fix_mode = "--fix" in sys.argv
    check_mode = "--check" in sys.argv

    if not CLI_MD_PATH.exists():
        print(f"Error: CLI.md not found at {CLI_MD_PATH}")
        return 1

    content = CLI_MD_PATH.read_text()
    current_ref = extract_current_reference(content)

    if not current_ref:
        print("Error: Could not find Command Reference section in CLI.md")
        print(f"Looking for '{START_MARKER}' ... '{END_MARKER}'")
        return 1

    generated_ref = generate_reference()

    # Compare normalized versions
    if normalize(current_ref) == normalize(generated_ref):
        print("CLI.md command reference is in sync")
        return 0

    print("CLI.md command reference is OUT OF SYNC with actual CLI")
    print()

    if fix_mode:
        new_content = update_cli_md(content, generated_ref)
        CLI_MD_PATH.write_text(new_content)
        print(f"Updated {CLI_MD_PATH}")
        return 0

    if check_mode:
        print("Run 'python scripts/check_cli_docs.py --fix' to update.")
        return 1

    # Interactive mode
    print("The documented commands don't match the actual --help output.")
    print()

    try:
        response = input("Show diff? [y/N] ").strip().lower()
        if response == "y":
            import difflib

            diff = difflib.unified_diff(
                current_ref.splitlines(keepends=True),
                generated_ref.splitlines(keepends=True),
                fromfile="CLI.md (current)",
                tofile="CLI.md (expected)",
            )
            print("".join(diff))

        print()
        response = input("Update CLI.md automatically? [y/N] ").strip().lower()
        if response == "y":
            new_content = update_cli_md(content, generated_ref)
            CLI_MD_PATH.write_text(new_content)
            print(f"Updated {CLI_MD_PATH}")
            print()
            print("Run 'git add sdk/CLI.md' and commit again.")
            return 1  # Still fail so user reviews and re-commits

    except (EOFError, KeyboardInterrupt):
        print()

    print()
    print("Commit aborted. To fix, run:")
    print("  python scripts/check_cli_docs.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
