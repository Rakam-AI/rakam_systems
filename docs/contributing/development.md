---
title: Development
---

# Development

## Branch conventions

- `main` — stable, always deployable
- `fix/<description>` — bug fixes
- `feat/<description>` — new features
- `chore/<description>` — maintenance, CI, tooling

## Commit messages

```
fix: Short description

What was done and why.
```

Use `fix:`, `feat:`, `chore:`, or `docs:` prefixes. Keep the first line under 72 characters.

## Pull requests

- One concern per PR. Small PRs are easier to review.
- Target `main` on `Rakam-AI/rakam_systems`.
- Include a summary of what changed and why.
- Link related issues.

## Testing

Run tests from the package directory:

```bash
cd core && pytest
cd ai-components/agents && pytest
cd ai-components/vector-store && pytest
cd tools && pytest
```
