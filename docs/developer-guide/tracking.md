---
title: Track and evaluate
---

# Track and evaluate

## Tracking system

Built-in input/output tracking for debugging and evaluation:

```python
from rakam_systems_core.tracking import TrackingManager, track_method, TrackingMixin

class MyAgent(TrackingMixin, BaseAgent):
    @track_method()
    async def arun(self, input_data, deps=None):
        return await super().arun(input_data, deps)

# Enable tracking
agent.enable_tracking(output_dir="./tracking")

# Export tracking data
agent.export_tracking_data(format='csv')
agent.export_tracking_data(format='json')

# Get statistics
stats = agent.get_tracking_statistics()
```

## Evaluation

For the evaluation CLI (`rakam eval`) and the evaluation SDK (`DeepEvalClient`), see:

- [User Guide — Evaluation](../user-guide/evaluation.md)
- [CLI reference](../cli/index.md)
