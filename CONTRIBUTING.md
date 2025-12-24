# 🤝 Contributing to Rakam Systems

Rakam Systems — an end-to-end AI Systems Lifecycle Framework.

Rakam Systems is built in layers to maximize reuse, maintainability, and clarity.
To contribute effectively, please follow the guidelines below.

# 🏗️ Contribution Levels

Rakam Systems has four contribution layers.
Each layer has its own review expectations, coding standards, and acceptance criteria.

| Level                    | Description                                                             | Who usually contributes |
| ------------------------ | ----------------------------------------------------------------------- | ----------------------- |
| **Level 1 — Syntax**     | Typing, formatting, docstrings, naming consistency                      | Everyone                |
| **Level 2 — Components** | Reusable blocks: tools, gateways, metrics, RAG primitives               | AI Team                 |
| **Level 3 — Pipelines**  | Higher-level flows: evaluation runners, monitoring handlers, data flows | Everyone                |
| **Level 4 — Templates**  | Project generators, system templates, agent templates                   | Everyone                |

Below is a detailed explanation of how to contribute at each level.

# 🔹 Level 1 — Syntax Contributions

(Formatting, typing, naming, documentation)

These contributions focus on code quality, clarity, and internal consistency.
They are essential for readability and long-term maintainability.

## ✔️ What counts as a Level-1 contribution

- Fixing typos

- Update DSL (Domain Specific Language)

  - Adding missing type hints

  - Renaming variables/functions for consistency

  - Improving documentations

  - Adding small unit tests

  - Fixing imports or unused code

  - Refactoring long functions into readable blocks (no behavior change)

  - Adding new syntax

## 🛠️ How to contribute

Ensure your changes do not modify behavior (unless it's a bugfix).

Follow the project conventions:

- Black for formatting

- Ruff for linting

- mypy for type checking

Run:

```
pre-commit run --all-files
pytest
```

Submit a PR with a clear title:

example `chores: Renamed run_evaluation function to evaluate`

### 📝 Review expectations:

- 1 reviewer required
- Changes should be safe, isolated, and reversible

# 🔹 Level 2 — Component Contributions

(Metrics, gateways, tools, utilities, RAG primitives, memory blocks, policies)

Components are the building blocks of Rakam Systems.
They should be generic, reusable, strongly typed, and easily composable. Contributions may involve improving an existing component or creating a new one.

## ✔️ What counts as a Level-2 contribution

- A new metric (e.g., retrieval_hit_rate, hallucination_score)

- A new LLM Gateway (Bedrock, Azure OpenAI, Together AI…)

- New tool types (search tool, web tool, SQL tool…)

## 🛠️ How to contribute

### In case of updating component

- develop new feature in the client's project
  - Identify the base compoonent from rakam_system to inherent from.
  - create new class that inheret from choosen base class
  - Implement new core logic
- switch to rakam_system repo:
  - checkout from main
  - copy and adapt new implementaion from the first step
  - push, open pr and request review

```python
# e.g. of new gateway implemented in the client's project
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Type, TypeVar
from pydantic import BaseModel
from rakam_systems.rakam_systems.ai_core.interfaces.llm_gateway import (
    LLMGateway,
    LLMRequest,
    LLMResponse,
)


T = TypeVar("T", bound=BaseModel)


class AnthropicGateway(LLMGateway):
    """
    ...
    """

    def __init__(
        self,
        name: str = "llm_gateway",
        ...
    ):
        ...

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: Standardized LLM request

        Returns:
            Standardized LLM response
        """
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Generate structured output conforming to a Pydantic schema.

        Args:
            request: Standardized LLM request
            schema: Pydantic model class to parse response into

        Returns:
            Instance of the schema class
        """
        raise NotImplementedError

    def stream(
        self,
        request: LLMRequest,
    ) -> Iterator[str]:
        """Stream token/segment responses.

        Args:
            request: Standardized LLM request

        Yields:
            String chunks from the LLM
        """
        ...

    @abstractmethod
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Model name to determine encoding

        Returns:
            Number of tokens
        """
        raise NotImplementedError

    # Legacy methods for backward compatibility
    def run(self, prompt: str, **kwargs: Any) -> str:
        """Legacy synchronous text completion."""
        ...

```

### In case of creating a new component

- in client's project
  - develop new component logic
  - write evaluation
  - adapt new logic to become rakam component
- switch to rakam_system repo:
  - create new module under rakam_system
  - copy and adapt code from first step ( try to follow examples of other existing components)
  - push, open pr and request review

# 🔹 Level 3 — Pipeline Contributions

(Evaluation pipelines, monitoring flows, data ingestion, orchestration)

Pipelines are end-to-end flows.
They orchestrate multiple components together.

## ✔️ What counts as a Level-3

- New evaluation pipeline
- Improvements to EvalService:
- run → test case → entry → metrics → reasons → logs
- Data preparation pipelines

## 🛠️ How to contribute

### In case of updating component

- develop new pipeline in the client's project
  - Identify the base compoonent from rakam_system to inherent from.
  - create new class that inheret from choosen base class
  - Implement new pipeline logic
- switch to rakam_system repo:
  - checkout from main
  - copy and adapt new implementaion from the first step
  - push, open pr and request review

# 🔹 Level 4 — Template Contributions

(System templates, agent templates, scaffolds, generators)

Templates define the starting point of entire AI systems.

## ✔️ What counts as a Level-4 contribution

- A full agent template:
  - Support Agent
  - RAG Agent
  - OCR Agent
  - ...
- A new service template (FastAPI, Django ...)
- A deployment template (Terraform/K8s)
- Improvements to project generation
- Integrate Monitoring features

## 🛠️ How to contribute

- update template (e.g. versions upgrade, update available rakam_system components, bug fixes...)
- push, open pr and request review
