# Rakam Systems Tools

Shared utilities for the Rakam Systems framework: evaluation client, S3 storage, logging, metrics, and tracing.

## Installation

```bash
pip install rakam-systems-tools
```

## Modules

### Evaluation Client

A client for submitting evaluation runs to the Rakam evaluation server. Used by `rakam-systems-cli` under the hood.

```python
from rakam_systems_tools.evaluation.client import EvalClient
from rakam_systems_tools.evaluation.schema import EvalConfig, TextInputItem
```

See the [CLI documentation](../cli/README.md) for the recommended way to run evaluations.

### S3 Utilities

Helpers for uploading and downloading files from S3-compatible storage.

```python
from rakam_systems_tools.utils.s3 import S3Client
```

### Logging, Metrics & Tracing

- `rakam_systems_tools.utils.logging` — structured logging
- `rakam_systems_tools.utils.metrics` — metrics collection
- `rakam_systems_tools.utils.tracing` — tracing utilities

## Documentation

See the [official documentation](https://rakam-ai.github.io/rakam-systems-docs/) for full usage guides.

## License

Apache 2.0
