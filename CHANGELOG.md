# Changelog

## (unreleased)

Fix

```
- Clean up getting started guide (QSG-MINOR-01, 05, 06, 07) [Yann
  Rapaport]
- Rename docs/intro.md to docs/introduction.md (MINOR-03) [Yann
  Rapaport]
- Update Python version requirement to 3.10+ [Yann Rapaport]

  - Update pyproject.toml: requires-python >= 3.10
  - Update README.md to reflect Python 3.10+ requirement
  - Create shared prerequisites partial (Python 3.10+, pip, Docker)
  - Update getting-started.md and user-guide.md to use shared prerequisites
  - Keep existing prerequisites (Docker, credentials) unchanged for now
- Allow importation of all modules and object from core.
  [somebodyawesome-dev]
- Fix build links for docs. [somebodyawesome-dev]
- Regenerate quick_start.md. [somebodyawesome-dev]
- Fix typo in docs. [somebodyawesome-dev]
- Fix importations issues and bump versions. [somebodyawesome-dev]
- Fix importation issue accross all packages. [somebodyawesome-dev]
- Fix vs ci. [somebodyawesome-dev]
- Added tests setup/config to core. [somebodyawesome-dev]
- Added dev group to core ci. [somebodyawesome-dev]
- Fix core added yaml missing deps. [somebodyawesome-dev]
- Increase min python version for agent and vs. [somebodyawesome-dev]
- Update broken links between ddocs for ci in docs repo.
  [somebodyawesome-dev]
- Fix deps issues when installing rakam-systems. [somebodyawesome-dev]
- Tests for cli. [somebodyawesome-dev]
- Fix cli's package name. [somebodyawesome-dev]
- Docstring and return type. [Sofia Casadei]

Other
```

- Chores: remove versions specification in docs. [somebodyawesome-dev]
- Merge branch 'yannrapaport-pr/4-getting-started-cleanup'
  [somebodyawesome-dev]
- Merge branch 'pr/4-getting-started-cleanup' of
  github.com:yannrapaport/rakam_systems into yannrapaport-pr/4-getting-
  started-cleanup. [somebodyawesome-dev]
- Chores: removed empty help section in user-guide. [somebodyawesome-
  dev]
- Merge pull request #93 from Rakam-AI:chores/docs-remove-links.
  [Mohamed Bashar Touil]

  Chores/docs-remove-links

- Chores: remove any rc or uncessary version mentions. [somebodyawesome-
  dev]
- Chores: removed duplicate links. [somebodyawesome-dev]
- Chores: update docs to point to latest rc versions. [somebodyawesome-
  dev]
- Merge pull request #92 from yannrapaport/fix/minor-03-url-
  introduction. [Mohamed Bashar Touil]

  fix: rename docs/intro.md to docs/introduction.md (MINOR-03)

- Merge pull request #90 from yannrapaport/pr/3-getting-started.
  [Mohamed Bashar Touil]

  [BLOCKER-02/03/04/09/10] Rewrite Getting Started Guide for public release

- Docs: rewrite Getting Started Guide for public release. [Claude Sonnet
  4.6, Yann Rapaport]

  [BLOCKER-09] Rename title and heading to "Getting Started Guide"
  [BLOCKER-10] Consistent naming now matches User Guide, Developer Guide format
  [BLOCKER-02] Remove internal OVH container registry docker run command;
  replace with contact message for evaluation service setup
  [BLOCKER-03] Remove RC version from pip install command
  [BLOCKER-04] Recommend .env file only for API keys; remove With Streaming
  section; focus EvalConfig only (remove SchemaEvalConfig reference);
  remove compare -o flag; add filename recommendation for first agent

- Merge pull request #89 from yannrapaport/pr/2-python-version. [Mohamed
  Bashar Touil]

  [BLOCKER-01] Update Python version requirement to 3.10+

- Merge pull request #88 from yannrapaport/pr/1-introduction. [Mohamed
  Bashar Touil]

  [BLOCKER-07] Rewrite Introduction page

- Docs: Update Introduction with vision-aligned content. [Yann Rapaport]

  BLOCKER-07: Align Introduction content with vision
  - Complete rewrite combining best elements from Notion and current versions
  - Added Origins section (establishes Rakam's AI expertise)
  - Updated Target Users: AI teams focus (removed commercial language)
  - Completely reworked 'Why Rakam Systems' section:
    - State-of-the-Art Technology (with 15+ concrete library examples)
    - Production-First Framework (with real-world production focus)
    - Open Source (humble, inviting tone)
  - Updated Core Components with production-ready emphasis
  - Kept Supporting Services & Tools section

  MINOR-04: Remove emojis
  - Removed 🥵 from Problem Statement (section removed)
  - Removed ✨ from Key Features (section removed)

  Key improvements:
  - Concrete examples: FastAPI, Pydantic, FAISS, pgvector, PyTorch,
    Sentence Transformers, Django, BeautifulSoup4, PyMuPDF, pandas, etc.
  - Real-world credibility through production experience (not commercial)
  - Technical depth: type safety, data structures, performance
  - Vocabulary compliant (no 'Engine' - uses 'Continuous Updates')
  - Professional, humble tone throughout

  Validation:
  - No commercial language
  - Clear technical focus
  - Emphasizes quality through specifics, not claims

- Merge pull request #87 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  Chores/restructure

- Chores: update package versions. [somebodyawesome-dev]
- Merge pull request #86 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  ft: make --dry-run arg run checks of eval configuration

- Chores: update docs versions. [somebodyawesome-dev]
- Ft: make --dry-run arg run checks of eval configuration.
  [somebodyawesome-dev]
- Chores: specify evalframeworks env are required to use evaluation
  service. [somebodyawesome-dev]
- Merge pull request #85 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  fix: fix build links for docs

- Merge pull request #84 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  Chores/restructure

- Chores: fix docs. [somebodyawesome-dev]
- Chores: update versions in docs. [somebodyawesome-dev]
- Merge pull request #83 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  chores: update docs

- Chores: update docs. [somebodyawesome-dev]
- Merge pull request #82 from Rakam-AI/chores/restructure. [Mohamed
  Bashar Touil]

  Chores/restructure

- Chores: update docs. [somebodyawesome-dev]
- Chores: update docs. [somebodyawesome-dev]
- Chores: update docs. [somebodyawesome-dev]
- Chores: update main ci branch. [somebodyawesome-dev]
- Chores: bump version for all rakam's packages. [somebodyawesome-dev]
- Ft: added initial ci for building all packages. [somebodyawesome-dev]
- Chores: added getting started docs. [somebodyawesome-dev]
- Chores: bump version to latest rc. [somebodyawesome-dev]
- Ft: remove deps to rs_tools from rs_core. [somebodyawesome-dev]
- Chores: quick update to docs. [somebodyawesome-dev]
- Ft: fixed tests after rakam_eval cli renaming. [somebodyawesome-dev]
- Ft: rename rakam_eval cli to "rakam eval" [somebodyawesome-dev]
- Chores: upgraded vectorstore version and published it.
  [somebodyawesome-dev]
- Chores: add missing docs. [somebodyawesome-dev]
- Ft: added ai_utils module in core to tools. [somebodyawesome-dev]
- Ft: removed ai_utils modules and refactored ai_core module.
  [somebodyawesome-dev]
- Ft: added rakam_systems_cli package. [somebodyawesome-dev]
- Ft: added rakam_systems_tool. [somebodyawesome-dev]
- Ft: added rakam_systems_vectorstore. [somebodyawesome-dev]
- Ft: added rakam_systems_agents package. [somebodyawesome-dev]
- Ft: added docs from rakam-systems-docs repo. [somebodyawesome-dev]
- Ft: added rakam_systems_core package. [somebodyawesome-dev]
- Chores: clean up unused modules. [somebodyawesome-dev]
- Ft: added init isssue and pr template. [somebodyawesome-dev]
