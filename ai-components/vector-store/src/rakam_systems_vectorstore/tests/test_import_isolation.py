"""Guard: importing the package must not drag in an optional provider SDK.

This package has no other minimal-install guard. CI installs `--all-extras`, so
every SDK is importable in the test environment and an accidental module-level
`from mistralai import Mistral` (or `import pydantic_ai`) somewhere in the import
chain would pass every other test while breaking a bare
`pip install rakam-systems-vectorstore` for everyone who wanted none of them.

The check runs in a SUBPROCESS because `sys.modules` in the pytest process is
already polluted by the other test modules.

This is also what makes `gateway_embeddings.py`'s "pydantic-ai imports are lazy so
importing this module never requires pydantic-ai" docstring an enforced promise
rather than a comment, and it is why `mistral_embedding_model.py` may import
mistralai eagerly at module level: nothing imports *it* until `build_embedder`
actually routes a `mistral:` ref.
"""
import subprocess
import sys

import pytest

# Optional-SDK distributions that no bare import may pull in. Anything added to a
# [project.optional-dependencies] group and imported from this package belongs
# here, under its IMPORT name.
OPTIONAL_SDKS = ("mistralai", "pydantic_ai", "openai", "cohere", "sentence_transformers")


@pytest.mark.parametrize("sdk", OPTIONAL_SDKS)
def test_importing_the_package_does_not_load(sdk):
    check = (
        "import sys; import rakam_systems_vectorstore; "
        f"assert {sdk!r} not in sys.modules, "
        f"'importing rakam_systems_vectorstore eagerly loaded {sdk}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_building_a_mistral_embedder_is_what_loads_mistralai():
    # The other half of the contract: the SDK must be reachable when it is
    # actually needed, so the laziness above is deferral and not a broken import.
    check = (
        "import os, sys; os.environ['MISTRAL_API_KEY'] = 'test-key'; "
        "from rakam_systems_core.config_schema import EmbeddingRef; "
        "from rakam_systems_vectorstore import build_embedder; "
        "build_embedder(EmbeddingRef(ref='mistral:mistral-embed', dim=1024)); "
        "assert 'mistralai' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
