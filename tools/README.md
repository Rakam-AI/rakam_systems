# DeepEvalClient

A lightweight Python client for interacting with the **Evaluation API**.
It provides convenient wrappers for text and schema evaluation endpoints, with support for background jobs and probabilistic execution.

---

## Features

- 🔹 **Text Evaluation** – Run evaluations on plain text inputs.
- 🔹 **Schema Evaluation** – Evaluate structured inputs against schema-based metrics.
- 🔹 **Background Jobs** – Submit jobs asynchronously and process later.
- 🔹 **Probabilistic Execution** – Run evaluations with a configurable chance (e.g., A/B testing scenarios).
- 🔹 **Robust Error Handling** – Handles network errors and invalid JSON gracefully.
- 🔹 **Configurable** – Configure via constructor args, environment variables, or external settings module.

---

## Installation

```bash
pip install rakam-eval-sdk
```

Usage

1. Basic Setup

```python
from deepeval.client import DeepEvalClient
from deepeval.schema import TextInputItem, MetricConfig

client = DeepEvalClient(
    base_url="http://localhost:8080",
    api_token="your-api-key"
)

```

2. Text Evaluation

```python

    client.maybe_text_eval_background(
                component="ocr",
                data=[
                    TextInputItem(

                        id="runtime evaluation", # identifiar (that can be unique). use same id in case you want to follow performance over time
                        input="...", # input given to ai component
                        output="...", # output of the ai component
                        # optional args/ condtional based on metrics passed
                        expected_output=["..."],
                        retrieval_context=[
                            ["..."]
                        ]

                    )
                ],
                metrics=[
                    ToxicityConfig(
                        # model="gpt-4.1",
                        threshold=0.2,
                        include_reason=False
                    ),
                    CorrectnessConfig(
                        steps=[
                            "You are evaluating text extracted from resumes and job descriptions using OCR.",
                            "1. Verify that the extracted text is coherent and free of major corruption (e.g., broken words, random characters).",
                            "2. Check whether key resume/job-related fields are preserved correctly (e.g., name, job title, skills, education, experience, company name, job requirements).",
                            "3. Ensure that important details are not missing or replaced with irrelevant content.",
                            "4. Ignore minor formatting issues (line breaks, spacing) as long as the information is readable and accurate.",
                            "5. Consider the output correct if it faithfully represents the resume or job description’s main information."
                        ],
                        params=["actual_output"],

                    )
                ],
                chance=.3
            )

```

3. Schema Evaluation

```python

    client.maybe_text_eval_background(
                component="ocr",
                data=[
                    TextInputItem(

                        id="runtime evaluation", # identifiar (that can be unique). use same id in case you want to follow performance over time
                        input="...", # input given to ai component
                        output="...", # output of the ai component
                        # optional args/ condtional based on metrics passed
                        expected_output=["..."],
                        retrieval_context=[
                            ["..."]
                        ]

                    )
                ],
                metrics=[
                    ToxicityConfig(
                        # model="gpt-4.1",
                        threshold=0.2,
                        include_reason=False
                    ),
                    CorrectnessConfig(
                        steps=[
                            "You are evaluating text extracted from resumes and job descriptions using OCR.",
                            "1. Verify that the extracted text is coherent and free of major corruption (e.g., broken words, random characters).",
                            "2. Check whether key resume/job-related fields are preserved correctly (e.g., name, job title, skills, education, experience, company name, job requirements).",
                            "3. Ensure that important details are not missing or replaced with irrelevant content.",
                            "4. Ignore minor formatting issues (line breaks, spacing) as long as the information is readable and accurate.",
                            "5. Consider the output correct if it faithfully represents the resume or job description’s main information."
                        ],
                        params=["actual_output"],

                    )
                ],
                chance=.3
            )

```

## Configuration

The client can be configured in multiple ways:

### Directly via constructor arguments

```python
DeepEvalClient(base_url="http://api", api_token="123")
```

### Environment variables

```bash
export EVALFRAMEWORK_URL=http://api
export EVALFRAMWORK_API_KEY=123
```

### Settings module

```python
import settings # it can be django settings e.g.: from django.conf import settings
client = DeepEvalClient(settings_module=settings)
```

<!-- uv publish --index testpypi
twine upload --repository testpypi dist/\*
uv add twine build --dev

uv build -->
