---
title: Documentation
---

# Documentation

## Where docs live

Documentation source files are in `docs/` within the `rakam_systems` repository. The Docusaurus site (`rakam-systems-docs`) pulls from this directory.

## Conventions

- **Headings:** sentence case, unnumbered, imperative verb form ("Configure API keys", not "1. Configuring API Keys")
- **Sub-sections:** use `###`, not bold-numbered steps
- **No horizontal rules** (`---`) except YAML frontmatter delimiters
- **No emojis** in headings or prose (OK in code output like `print("✅ Done")`)
- **Language:** English, no commercial language (open source project)
- **No internal references** in public-facing content (URLs, credentials, internal tooling)

## File structure

Every `.md` file must start with YAML frontmatter:

```markdown
---
title: Page Title
---

# Page Title

Introductory sentence.

## First section
```

## Style

- One sentence per paragraph for introductions, then code example.
- Keep code examples minimal and runnable.
- Use Docusaurus admonitions for important notes:

```markdown
:::note
The evaluation service is configured separately.
:::
```

## Import paths

Always use the correct module names:

- `rakam_systems_core` (not `rakam_system_core`, not `ai_core`)
- `rakam_systems_agent`
- `rakam_systems_vectorstore`
- `rakam_systems_tools`
- `rakam_systems_cli`

## Preview locally

```bash
cd repos/rakam-systems-docs
npm install
npx docusaurus start --port 3001
```
