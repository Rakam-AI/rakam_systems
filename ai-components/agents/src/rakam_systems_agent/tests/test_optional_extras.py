"""Tests that the optional extras are genuinely optional.

``psycopg2`` ([postgres]) and the provider SDKs ([llm-providers]) used to be
imported eagerly by the package ``__init__`` chain, so ``import
rakam_systems_agent`` raised ``ModuleNotFoundError`` on a correct minimal
install -- including for consumers that only want ``ModelGateway``.

The extras ARE installed in CI (``uv sync --all-extras``), so a minimal install
is simulated in a subprocess whose ``sys.meta_path`` refuses the optional
dependencies. That also makes the test independent of the local environment.
"""
import subprocess
import sys
import textwrap

import pytest

# Third-party packages that only the optional extras install.
BLOCKED = ("psycopg2", "openai", "mistralai", "tiktoken")

_BLOCKER = f"""
import importlib.abc
import sys

BLOCKED = {BLOCKED!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{root!r}}", name=root)
        return None


sys.meta_path.insert(0, _Blocker())
"""


def run_without_extras(body: str) -> str:
    """Run ``body`` in a subprocess where the optional dependencies are absent."""
    script = _BLOCKER + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_package_imports_without_the_optional_extras():
    out = run_without_extras(
        """
        import rakam_systems_agent
        from rakam_systems_agent import BaseAgent, JSONChatHistory, SQLChatHistory
        print("imported")
        """
    )
    assert "imported" in out


def test_model_gateway_is_usable_without_the_optional_extras():
    # The reason the eager imports mattered: ModelGateway needs pydantic-ai
    # only, but was unreachable behind a psycopg2 ImportError.
    out = run_without_extras(
        """
        from rakam_systems_agent.components.model_gateway import ModelGateway
        print(type(ModelGateway()).__name__)
        """
    )
    assert "ModelGateway" in out


@pytest.mark.parametrize(
    "symbol, extra",
    [
        ("PostgresChatHistory", "postgres"),
        ("OpenAIGateway", "llm-providers"),
        ("MistralGateway", "llm-providers"),
        ("LLMGatewayFactory", "llm-providers"),
        ("get_llm_gateway", "llm-providers"),
    ],
)
def test_optional_symbol_errors_name_the_extra(symbol, extra):
    out = run_without_extras(
        f"""
        import rakam_systems_agent
        try:
            rakam_systems_agent.{symbol}
        except ImportError as exc:
            print(exc)
        """
    )
    assert symbol in out
    assert f"rakam-systems-agent[{extra}]" in out


def test_unknown_attribute_still_raises_attribute_error():
    out = run_without_extras(
        """
        import rakam_systems_agent
        try:
            rakam_systems_agent.NoSuchThing
        except AttributeError as exc:
            print(exc)
        """
    )
    assert "NoSuchThing" in out


def test_importing_the_package_does_not_load_the_optional_dependencies():
    # Deferred, not merely tolerated: a re-added eager import would still pass
    # the tests above whenever the extras happen to be installed.
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rakam_systems_agent; "
            f"print([m for m in {BLOCKED!r} if m in sys.modules])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "[]"


def test_postgres_chat_history_keeps_its_public_import_paths():
    psycopg2 = pytest.importorskip(
        "psycopg2", reason="[postgres] extra not installed"
    )
    assert psycopg2 is not None

    from rakam_systems_agent import PostgresChatHistory as from_package
    from rakam_systems_agent.components import PostgresChatHistory as from_components
    from rakam_systems_agent.components.chat_history import (
        PostgresChatHistory as from_chat_history,
    )
    from rakam_systems_agent.components.chat_history.postgres_chat_history import (
        PostgresChatHistory as from_module,
    )

    assert from_package is from_components is from_chat_history is from_module


def test_llm_gateway_symbols_keep_their_public_import_paths():
    pytest.importorskip("openai", reason="[llm-providers] extra not installed")

    from rakam_systems_agent import LLMGatewayFactory as factory_from_package
    from rakam_systems_agent import OpenAIGateway as from_package
    from rakam_systems_agent import get_llm_gateway as helper_from_package
    from rakam_systems_agent.components.llm_gateway import (
        LLMGatewayFactory as factory_from_subpackage,
    )
    from rakam_systems_agent.components.llm_gateway import (
        OpenAIGateway as from_subpackage,
    )
    from rakam_systems_agent.components.llm_gateway import (
        get_llm_gateway as helper_from_subpackage,
    )

    assert from_package is from_subpackage
    assert factory_from_package is factory_from_subpackage
    assert helper_from_package is helper_from_subpackage
