# EarlyAI Test Integration in OpenLLmetry

## Exeutive Summary

This document outlines the integration of **EarlyAI-generated tests** into the OpenLLmetry monorepo. These tests improve test coverage and ensure instrumentation correctness while keeping the existing test flow intact.

Here is a summay of the test generated for the utils folder on the following projects:

| Project                                     | Total Tests | Passed  | Failed |
| ------------------------------------------- | ----------- | ------- | ------ |
| **opentelemetry-instrumentation-anthropic** | 48          | 47      | 1      |
| **opentelemetry-instrumentation-haystack**  | 20          | 20      | 0      |
| **opentelemetry-instrumentation-pinecone**  | 18          | 17      | 1      |
| **opentelemetry-instrumentation-groq**      | 29          | 29      | 0      |
| **Total**                                   | **115**     | **113** | **2**  |

## Failure Details

### opentelemetry-instrumentation-pinecone

**TestSetSpanAttribute.test_set_attribute_with_none_name_and_valid_value** failed.

- **Assertion failed:** Expected `set_attribute` to not be called, but it was called once.

### opentelemetry-instrumentation-anthropic

**TestSharedMetricsAttributes.test_shared_metrics_attributes_with_none_response** failed.

- Assertion failed: Expected a structured response, but `None` was returned

## Key Additions

### 1. Test Configuration:

- Updated **nx.json** to define `test:early` as a target for running EarlyAI tests across projects.
- Updated **package.json** to include scripts for running EarlyAI tests.
- Added a global **pytest.ini** file to manage test markers and configurations centrally.

### 2. Test Execution Support:

- Tests can be executed across the **entire monorepo** or **per project**.
- EarlyAI tests displayed in the **Early** VS Code extension.

## How to Run EarlyAI Tests

### Run All EarlyAI Tests Across All Projects

```bash
npm run test:early
```

This command runs all EarlyAI tests across the monorepo.

### Run EarlyAI Tests for a Specific Project

```bash
nx run <project-name>:test:early
```

Replace `<project-name>` with the relevant project (e.g., `opentelemetry-instrumentation-openai`).

---

## Technical Changes

### 1. Updated `nx.json`

We added a **global target** for EarlyAI test execution:

```json
"test:early": {
          "executor": "@nxlv/python:run-commands",
          "options": {
            "command": ". .venv/Scripts/activate && poetry run pytest source/test_early_utils/",
            "cwd": "{projectRoot}"
          }
        }
```

### 2. Updated `package.json`

Added a global script for running EarlyAI tests:

```json
"scripts": {
  "test:early": "nx run-many --target=test:early"
}
```

### 3. Added a Global `pytest.ini`

Instead of managing individual `pytest.ini` files per project, we added a **global pytest.ini**:

```ini
[tool.pytest.ini_options]
markers = [
    "describe: Custom marker for test groups",
    "happy_path: Tests the 'happy path' of a function",
    "edge_case: Tests edge cases of a function"
]
```

### 4. Added `test:early` Target in Each Project

Each project where EarlyAI tests were added includes the following target in its `project.json`:

```json
"test:early": {
  "executor": "@nxlv/python:run-commands",
  "outputs": [
    "{workspaceRoot}/reports/packages/opentelemetry-instrumentation-anthropic/unittests/early",
    "{workspaceRoot}/coverage/packages/opentelemetry-instrumentation-anthropic/early"
  ],
  "options": {
    "command": "poetry run pytest opentelemetry/instrumentation/anthropic/test_early_utils/",
    "cwd": "packages/opentelemetry-instrumentation-anthropic"
  }
}
```

(Each project follows a similar structure, replacing **anthropic** with the respective project name.)

[Early-Ai for Vscode](vscode:extension/Early-AI.EarlyAI)

[Early-Ai for Cursor](cursor:extension/Early-AI.EarlyAI)
