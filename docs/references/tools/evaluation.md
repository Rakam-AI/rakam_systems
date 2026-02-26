---
title: Evaluation SDK
---

# Rakam Eval SDK Documentation

This document provides a comprehensive guide to using the `rakam-system-tools`, a Python client for interacting with the DeepEval API. This SDK allows you to run evaluations on your text and schema data, either synchronously or in the background.

## Installation

To get started, install the `rakam-system-tools` package using pip:

```bash
pip install rakam-system-tools
```

## Configuration

The `DeepEvalClient` is the main entry point for using the SDK. To use it, you need to configure the client with your API endpoint and token.

### Initializing the Client

You can initialize the `DeepEvalClient` by providing the `base_url` and `api_token` directly:

```python
from rakam_systems_tools.evaluation import DeepEvalClient

client = DeepEvalClient(
    base_url="http://your-deepeval-api-url.com",
    api_token="your-api-token"
)
```

### Configuration Options

The client can be configured in three ways, in the following order of precedence:

1.  **Directly in the constructor:** As shown in the example above.
2.  **Using a settings module:** You can pass a settings module to the client. The client will look for `EVALFRAMEWORK_URL` and `EVALFRAMEWORK_API_KEY` attributes in the module.

    ```python
    # settings.py
    EVALFRAMEWORK_URL = "http://your-deepeval-api-url.com"
    EVALFRAMEWORK_API_KEY = "your-api-token"

    # your_app.py
    from rakam_systems_tools.evaluation import DeepEvalClient
    import settings # for django can be something like: from django.conf import settings

    client = DeepEvalClient(settings_module=settings)
    ```

3.  **Using environment variables:** The client will automatically pick up the `EVALFRAMEWORK_URL` and `EVALFRAMEWORK_API_KEY` environment variables if they are set.

    ```bash
    export EVALFRAMEWORK_URL="http://your-deepeval-api-url.com"
    export EVALFRAMEWORK_API_KEY="your-api-token"
    ```

    ```python
    from rakam_systems_tools.evaluation import DeepEvalClient

    client = DeepEvalClient()
    ```

If no `base_url` is provided, it defaults to `http://localhost:8080`.

## Usage

The SDK provides methods for evaluating text and schema data. For each type of evaluation, there are synchronous and background (asynchronous) methods.

### Text Evaluation

Text evaluation is used for tasks like checking correctness, relevancy, faithfulness, and toxicity of text data.

#### Data and Metrics

- **`TextInputItem`**: Represents a single item for text evaluation. It includes `id`, `input`, `output`, `expected_output` (optional), and `retrieval_context` (optional).
- **`MetricConfig`**: Defines the metrics to be used for evaluation. Available text evaluation metrics are:
  - `CorrectnessConfig`
  - `AnswerRelevancyConfig`
  - `FaithfulnessConfig`
  - `ToxicityConfig`

#### Methods

- `text_eval()`: Runs a synchronous text evaluation.
- `text_eval_background()`: Runs a background text evaluation.
- `maybe_text_eval()`: Randomly runs `text_eval` based on a given probability.
- `maybe_text_eval_background()`: Randomly runs `text_eval_background` based on a given probability.

#### Example

```python
from rakam_systems_tools.evaluation import DeepEvalClient, TextInputItem, CorrectnessConfig

client = DeepEvalClient()

data = [
    TextInputItem(
        input="What is the capital of France?",
        output="Paris",
        expected_output="The capital of France is Paris."
    )
]

metrics = [
    CorrectnessConfig(model="gpt-4.1", steps=["Check if the output correctly answers the input."])
]

result = client.text_eval(data=data, metrics=metrics, component="faq-bot")
print(result)
```

### Schema Evaluation

Schema evaluation is used for tasks that involve structured data, such as JSON. It can be used to check for JSON correctness and the presence of specific fields.

#### Data and Metrics

- **`SchemaInputItem`**: Represents a single item for schema evaluation. It includes `input` and `output`.
- **`SchemaMetricConfig`**: Defines the metrics for schema evaluation. Available metrics are:
  - `JsonCorrectnessConfig`
  - `FieldsPresenceConfig`

#### Methods

- `schema_eval()`: Runs a synchronous schema evaluation.
- `schema_eval_background()`: Runs a background schema evaluation.
- `maybe_schema_eval()`: Randomly runs `schema_eval` based on a given probability.
- `maybe_schema_eval_background()`: Randomly runs `schema_eval_background` based on a given probability.

#### Example

```python
from rakam_systems_tools.evaluation import DeepEvalClient, SchemaInputItem, JsonCorrectnessConfig

client = DeepEvalClient()

data = [
    SchemaInputItem(
        input="Generate a JSON object with name and age.",
        output='{"name": "John", "age": 30}'
    )
]

metrics = [
    JsonCorrectnessConfig(
        excpected_schema={"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}  # Note: parameter name is misspelled in the SDK
    )
]

result = client.schema_eval(data=data, metrics=metrics, component="json-generator")
print(result)
```

### Probabilistic Evaluation

The `maybe_*` methods provide a way to run evaluations probabilistically. This can be useful for reducing the load on the evaluation service or for sampling evaluations. The `chance` parameter is a float between 0 and 1 that determines the probability of the evaluation running.

```python
# This evaluation will run approximately 10% of the time.
client.maybe_text_eval(data=data, metrics=metrics, chance=0.1)
```

### Error Handling

By default, the client methods will not raise exceptions on network errors or non-2xx HTTP responses. Instead, they will return a dictionary with an "error" key. To change this behavior and have exceptions raised, you can set `raise_exception=True`.

```python
try:
    result = client.text_eval(data=data, metrics=metrics, raise_exception=True)
except requests.RequestException as e:
    print(f"An error occurred: {e}")
```

### Client-Side Metrics

In addition to server-side evaluations, the SDK supports logging metrics that are calculated on the client side. This is useful when you have your own evaluation methods or want to log scores from other systems. These client-side metrics are sent along with the input data and are logged without any server-side evaluation.

#### `ClientSideMetricConfig`

The `ClientSideMetricConfig` class allows you to define your own metrics. The fields are:

- `name` (str): The name of the metric.
- `score` (float): The score of the metric.
- `success` (Optional[int]): Whether the evaluation was successful (1 for success, 0 for failure). Defaults to 1.
- `evaluation_cost` (Optional[float]): The cost of the evaluation. Defaults to 0.
- `reason` (Optional[str]): An optional reason or explanation for the score.
- `threshold` (Optional[float]): An optional threshold for the metric. Defaults to 0.

#### Example

You can include a list of `ClientSideMetricConfig` objects in the `metrics` field of each `TextInputItem` or `SchemaInputItem`.

```python
from rakam_systems_tools.evaluation import DeepEvalClient, TextInputItem, ClientSideMetricConfig

client = DeepEvalClient()

# Assume you have a custom function to evaluate sentiment
def calculate_sentiment_score(text):
    # Your custom logic here
    if "happy" in text:
        return 1.0
    return 0.2

output_text = "I am happy with this product."
sentiment_score = calculate_sentiment_score(output_text)

data = [
    TextInputItem(
        input="User review",
        output=output_text,
        metrics=[
            ClientSideMetricConfig(
                name="sentiment",
                score=sentiment_score,
                reason="The user expressed a positive sentiment."
            )
        ]
    )
]

# You can send client-side metrics without any server-side metrics
result = client.text_eval(data=data, metrics=[], component="review-analyzer")
print(result)
```

> **Note:** When you send client-side metrics, you can pass an empty list to the `metrics` parameter of the `text_eval` or `schema_eval` methods if you don't want to run any server-side evaluations.
