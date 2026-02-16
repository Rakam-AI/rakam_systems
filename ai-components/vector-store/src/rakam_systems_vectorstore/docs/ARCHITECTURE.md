# AI Vectorstore - Architecture & Design Decisions

## Module Structure

The `ai_vectorstore` is designed as a **namespace submodule** within the `rakam_systems` framework. This document explains the architectural decisions and import structure.

## Package Hierarchy

```
rakam_systems/
├── ai_core/              # Core interfaces and base components
│   ├── interfaces/       # Abstract base classes
│   │   ├── vectorstore.py
│   │   ├── embedding_model.py
│   │   ├── chunker.py
│   │   └── ...
│   └── base.py          # BaseComponent
│
├── ai_vectorstore/       # Vector store implementations
│   ├── core.py          # Core data structures (Node, VSFile, etc.)
│   ├── config.py        # Configuration system
│   ├── components/      # Component implementations
│   │   ├── vectorstore/
│   │   ├── embedding_model/
│   │   ├── chunker/
│   │   └── ...
│   └── ...
│
└── ai_agents/           # Agent framework
    └── ...
```

## Import Structure

### Why Use `from rakam_systems_vectorstore.*`?

The module uses **namespace package** pattern for several important reasons:

#### 1. Shared Interfaces

`ai_vectorstore` implements interfaces defined in `ai_core`:

```python
# ai_vectorstore depends on ai_core interfaces
from rakam_system_core.ai_core.interfaces.vectorstore import VectorStore
from rakam_system_core.ai_core.interfaces.embedding_model import EmbeddingModel
from rakam_system_core.ai_core.base import BaseComponent
```

These interfaces ensure:

- Consistent API across all rakam_systems components
- Type safety and IDE support
- Interoperability with other rakam_systems modules (e.g., ai_agents)

#### 2. Standalone Installation Includes Dependencies

When installed standalone, the `setup.py` includes both modules:

```python
packages=find_packages(
    where="..",
    include=["rakam_systems_vectorstore*", "rakam_system_core*"]
)
```

This means:

- Users get all required interfaces automatically
- No code duplication
- Single source of truth for interfaces

#### 3. Consistent Import Pattern

Whether installed as:

- `pip install rakam-systems[ai-vectorstore]` (part of full framework)
- `pip install ./rakam_systems/ai_vectorstore` (standalone)

Users **always import the same way**:

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore
from rakam_systems_vectorstore import Node, NodeMetadata
from rakam_systems_vectorstore.config import VectorStoreConfig
```

## Internal vs External Imports

### Internal Imports (Within ai_vectorstore)

These are imports within the ai_vectorstore module:

```python
# ✅ Correct: Internal imports use full path
from rakam_systems_vectorstore.core import Node, NodeMetadata
from rakam_systems_vectorstore.config import VectorStoreConfig
from rakam_systems_vectorstore.components.vectorstore.pg_models import Collection
```

**Why not relative imports?**

- Clearer and more explicit
- Easier to refactor and move files
- Consistent with cross-module imports
- Better IDE support

### Cross-Module Imports (ai_vectorstore → ai_core)

These are imports from other rakam_systems modules:

```python
# ✅ Correct: Cross-module imports
from rakam_system_core.ai_core.interfaces.vectorstore import VectorStore
from rakam_system_core.ai_core.interfaces.embedding_model import EmbeddingModel
from rakam_system_core.ai_core.base import BaseComponent
```

**Why this pattern?**

- ai_core provides standard interfaces
- All components share the same base classes
- Enables plugin architecture
- Allows components to work together

## Alternative Approach (Not Recommended)

### Completely Independent Package

You could make ai_vectorstore fully independent:

```python
# Would require changing to:
from ai_vectorstore import ConfigurablePgVectorStore
from ai_vectorstore.core import Node
```

**Why NOT recommended:**

1. **Code Duplication**: Would need to copy ai_core interfaces into ai_vectorstore
2. **Maintenance Burden**: Two copies of interfaces to maintain
3. **Breaking Change**: Different import pattern confuses users
4. **Loss of Interoperability**: Can't work with other rakam_systems components
5. **Version Conflicts**: Risk of interface version mismatches

### Relative Imports

You could use relative imports internally:

```python
# Internal imports with relative paths
from .core import Node, NodeMetadata
from .config import VectorStoreConfig
from ..ai_core.interfaces.vectorstore import VectorStore  # Problematic!
```

**Why NOT recommended:**

1. **Harder to Read**: Less explicit about what's being imported
2. **Refactoring Issues**: Breaks when moving files between directories
3. **IDE Issues**: Some IDEs struggle with relative imports
4. **Cross-module Ambiguity**: `..ai_core` is less clear than `rakam_system_core.ai_core`

## Benefits of Current Structure

### ✅ For Users

1. **Consistent Imports**: Same import pattern whether standalone or full install
2. **Type Safety**: Full IDE autocomplete and type checking
3. **No Surprises**: Clear where components come from

### ✅ For Developers

1. **Clear Dependencies**: Explicit cross-module dependencies
2. **Easy Testing**: Can mock `rakam_system_core.ai_core` interfaces
3. **Maintainable**: Single source of truth for interfaces
4. **Extensible**: Easy to add new components

### ✅ For the Ecosystem

1. **Interoperability**: Components work together seamlessly
2. **Plugin Architecture**: Third parties can implement interfaces
3. **Consistent API**: All components follow same patterns

## Installation Modes

### Mode 1: Full Framework

```bash
cd app/rakam_systems
pip install -e ".[all]"
```

Installs entire rakam_systems with all components.

### Mode 2: Selective Components

```bash
cd app/rakam_systems
pip install -e ".[ai-vectorstore]"
```

Installs core + ai_vectorstore (includes ai_core interfaces).

### Mode 3: Standalone Submodule

```bash
cd app/rakam_systems/rakam_systems/ai_vectorstore
pip install -e ".[all]"
```

Installs ai_vectorstore + ai_core (minimal rakam_systems footprint).

**In all cases, imports are identical:**

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore
```

## Package Distribution

When distributed via PyPI (future):

```bash
# From PyPI (future)
pip install rakam-systems-ai-vectorstore

# Still import as:
from rakam_systems_vectorstore import ConfigurablePgVectorStore
```

The package name (`rakam-systems-ai-vectorstore`) is different from the import path (`rakam_systems_vectorstore`), which is standard Python practice.

## Design Principles

1. **Namespace Packages**: Use Python namespace packages for modular structure
2. **Explicit Imports**: Use full import paths for clarity
3. **Shared Interfaces**: Define interfaces in ai_core, implement in submodules
4. **Consistent API**: Same import pattern across all installation modes
5. **Minimal Dependencies**: Core dependencies separate from optional features

## Conclusion

The current import structure:

```python
from rakam_system_core.ai_core.interfaces.vectorstore import VectorStore
from rakam_systems_vectorstore.core import Node, NodeMetadata
from rakam_systems_vectorstore.config import VectorStoreConfig
```

Is the **correct and recommended approach** for a namespace submodule that:

- Can be installed standalone
- Shares interfaces with other components
- Maintains consistency across installation modes
- Provides a clear, maintainable architecture

## References

- [PEP 420 - Implicit Namespace Packages](https://peps.python.org/pep-0420/)
- [Python Packaging Guide - Namespace Packages](https://packaging.python.org/guides/packaging-namespace-packages/)
- [Setuptools - Package Discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)

---

**Last Updated**: 2025-11-18  
**Author**: Rakam Systems Team
