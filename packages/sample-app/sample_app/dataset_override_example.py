"""
Example script demonstrating the Traceloop Dataset override functionality.

Creates a dataset and then overrides its columns and rows.
"""

from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import ColumnType
from traceloop.sdk.datasets.model import (
    ColumnDefinition,
    CreateDatasetRequest,
    OverrideDatasetRequest,
)

# Initialize Traceloop
client = Traceloop.init(
    endpoint_is_traceloop=True
)


def main():
    slug = "sdk-override-example"

    print("=== Create Initial Dataset ===")
    create_request = CreateDatasetRequest(
        slug=slug,
        name="Original Dataset",
        description="Initial dataset before override",
        columns=[
            ColumnDefinition(name="Question", slug="question", type=ColumnType.STRING),
            ColumnDefinition(name="Answer", slug="answer", type=ColumnType.STRING),
        ],
        rows=[
            {"question": "What is Python?", "answer": "A programming language"},
            {"question": "What is OpenTelemetry?", "answer": "An observability framework"},
        ],
    )

    dataset = client.datasets.create(create_request)
    print(
        f"Created dataset '{dataset.name}' with "
        f"{len(dataset.columns)} columns and {len(dataset.rows)} rows"
    )

    # print("\n=== Override Dataset ===")
    # override_request = OverrideDatasetRequest(
    #     name="Updated Dataset Name",
    #     description="Updated description",
    #     columns=[
    #         ColumnDefinition(name="Input", slug="input", type=ColumnType.STRING),
    #         ColumnDefinition(
    #             name="Expected Output", slug="expected-output", type=ColumnType.STRING
    #         ),
    #     ],
    #     rows=[
    #         {"input": "hello", "expected-output": "world"},
    #         {"input": "foo", "expected-output": "bar"},
    #     ],
    # )

    # updated_dataset = client.datasets.override(slug, override_request)
    # print(
    #     f"Overridden dataset '{updated_dataset.name}' with "
    #     f"{len(updated_dataset.columns)} columns and {len(updated_dataset.rows)} rows"
    # )

    # # Clean up
    # print("\n=== Cleanup ===")
    # client.datasets.delete_by_slug(slug)
    # print("Dataset deleted")


if __name__ == "__main__":
    main()
