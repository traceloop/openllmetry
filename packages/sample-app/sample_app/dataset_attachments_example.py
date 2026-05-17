#!/usr/bin/env python3
"""
Sample application demonstrating how to use the attachment feature in Traceloop SDK.

This example shows:
1. Creating datasets with external URL attachments (YouTube videos, Google Docs)
2. Creating datasets with file uploads (local images, PDFs)
3. Creating datasets with in-memory data attachments
4. Mixed attachment types in a single dataset
"""

import os
import tempfile

from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import (
    Attachment,
    Datasets,
    ExternalAttachment,
    FileCellType,
)
from traceloop.sdk.datasets.model import (
    ColumnDefinition,
    ColumnType,
    CreateDatasetRequest,
)


def example_external_attachments():
    """Example: Creating a dataset with external URL attachments."""
    print("\n=== Example 1: External URL Attachments ===")

    # Initialize Traceloop
    Traceloop.init(app_name="attachment-demo")
    datasets = Datasets()

    # Create a product catalog with external media
    dataset_request = CreateDatasetRequest(
        slug="product-catalog-with-media",
        name="Product Catalog with Media",
        description="Product catalog with videos and documentation links",
        columns=[
            ColumnDefinition(
                slug="product_name", name="Product Name", type=ColumnType.STRING
            ),
            ColumnDefinition(slug="price", name="Price", type=ColumnType.NUMBER),
            ColumnDefinition(
                slug="demo_video", name="Demo Video", type=ColumnType.FILE
            ),
            ColumnDefinition(
                slug="user_manual", name="User Manual", type=ColumnType.FILE
            ),
        ],
        rows=[
            {
                "product_name": "Smart Widget Pro",
                "price": 299.99,
                "demo_video": ExternalAttachment(
                    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    file_type=FileCellType.VIDEO,
                    metadata={
                        "title": "Smart Widget Pro Demo",
                        "duration": "5:32",
                        "resolution": "1080p",
                    },
                ),
                "user_manual": ExternalAttachment(
                    url="https://docs.google.com/document/d/example-manual-id",
                    file_type=FileCellType.FILE,
                    metadata={"pages": 45, "format": "Google Docs", "version": "2.1"},
                ),
            },
            {
                "product_name": "EcoGadget Plus",
                "price": 199.99,
                "demo_video": ExternalAttachment(
                    url="https://vimeo.com/123456789",
                    file_type=FileCellType.VIDEO,
                    metadata={"title": "EcoGadget Plus Overview", "duration": "3:15"},
                ),
                "user_manual": ExternalAttachment(
                    url="https://example.com/manuals/ecogadget-plus.pdf",
                    file_type=FileCellType.FILE,
                    metadata={"pages": 30, "format": "PDF"},
                ),
            },
        ],
    )

    # Create the dataset
    dataset = datasets.create(dataset_request)
    print(f"Created dataset: {dataset.slug}")
    print(f"Total rows: {len(dataset.rows)}")

    # Access the attachment information
    for row in dataset.rows:
        print(f"\nProduct: {row.values['product_name']}")
        video = row.values.get("demo_video")
        if video:
            print(f"  Video URL: {video.get('url')}")
            print(f"  Video Type: {video.get('type')}")


def example_file_uploads():
    """Example: Creating a dataset with file uploads."""
    print("\n=== Example 2: File Upload Attachments ===")

    # Create temporary test files
    # In a real application, these would be actual files
    image_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image_file.write(b"fake image data for demo")
    image_file.close()

    pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf_file.write(b"fake pdf data for demo")
    pdf_file.close()

    try:
        datasets = Datasets()

        # Create a dataset with file uploads
        dataset_request = CreateDatasetRequest(
            slug="employee-records-with-photos",
            name="Employee Records with Photos",
            description="Employee database with profile photos and resumes",
            columns=[
                ColumnDefinition(
                    slug="employee_id", name="Employee ID", type=ColumnType.STRING
                ),
                ColumnDefinition(slug="name", name="Full Name", type=ColumnType.STRING),
                ColumnDefinition(
                    slug="profile_photo", name="Profile Photo", type=ColumnType.FILE
                ),
                ColumnDefinition(slug="resume", name="Resume", type=ColumnType.FILE),
            ],
            rows=[
                {
                    "employee_id": "EMP001",
                    "name": "Alice Johnson",
                    "profile_photo": Attachment(
                        file_path=image_file.name,
                        file_type=FileCellType.IMAGE,
                        metadata={
                            "alt_text": "Alice Johnson profile photo",
                            "photographer": "Company Photo Services",
                            "date_taken": "2024-01-15",
                        },
                    ),
                    "resume": Attachment(
                        file_path=pdf_file.name,
                        file_type=FileCellType.FILE,
                        content_type="application/pdf",
                        metadata={
                            "version": "3.0",
                            "last_updated": "2024-03-01",
                            "pages": 2,
                        },
                    ),
                },
            ],
        )

        # Create the dataset (uploads will happen automatically)
        dataset = datasets.create(dataset_request)
        print(f"Created dataset: {dataset.slug}")

        # Check upload status
        for row in dataset.rows:
            print(f"\nEmployee: {row.values['name']}")
            photo = row.values.get("profile_photo")
            if photo:
                print(f"  Photo Status: {photo.get('status')}")
                print(f"  Storage Type: {photo.get('storage')}")

            resume = row.values.get("resume")
            if resume:
                print(f"  Resume Status: {resume.get('status')}")
                print(f"  Storage Type: {resume.get('storage')}")

    finally:
        # Clean up temporary files
        os.unlink(image_file.name)
        os.unlink(pdf_file.name)


def example_in_memory_attachments():
    """Example: Creating a dataset with in-memory data attachments."""
    datasets = Datasets()

    # Generate some in-memory data
    # This could be data generated by your application
    csv_data = b"name,score\nAlice,95\nBob,87\nCarol,92"
    json_data = b'{"config": "example", "version": "1.0"}'

    # Create dataset with in-memory attachments
    dataset_request = CreateDatasetRequest(
        slug="analysis-results",
        name="Analysis Results",
        description="Results from data analysis with generated reports",
        columns=[
            ColumnDefinition(
                slug="analysis_id", name="Analysis ID", type=ColumnType.STRING
            ),
            ColumnDefinition(
                slug="dataset_name", name="Dataset Name", type=ColumnType.STRING
            ),
            ColumnDefinition(
                slug="results_csv", name="Results CSV", type=ColumnType.FILE
            ),
            ColumnDefinition(
                slug="config_json", name="Configuration", type=ColumnType.FILE
            ),
        ],
        rows=[
            {
                "analysis_id": "ANA001",
                "dataset_name": "Q4 Sales Analysis",
                "results_csv": Attachment(
                    data=csv_data,
                    filename="q4_results.csv",
                    content_type="text/csv",
                    file_type=FileCellType.FILE,
                    metadata={"rows": 3, "columns": 2, "analysis_date": "2024-01-10"},
                ),
                "config_json": Attachment(
                    data=json_data,
                    filename="analysis_config.json",
                    content_type="application/json",
                    file_type=FileCellType.FILE,
                    metadata={"version": "1.0", "algorithm": "regression"},
                ),
            },
        ],
    )

    dataset = datasets.create(dataset_request)
    print(f"Created dataset: {dataset.slug}")
    print("Attachments uploaded from memory")


def example_mixed_attachments():
    """Example: Creating a dataset with mixed attachment types."""
    print("\n=== Example 4: Mixed Attachment Types ===")

    # Create a temporary file for local attachment
    local_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    local_file.write(b"Sample report content")
    local_file.close()

    try:
        datasets = Datasets()

        # Create dataset with different attachment types
        dataset_request = CreateDatasetRequest(
            slug="project-documentation",
            name="Project Documentation",
            description="Project docs with various attachment types",
            columns=[
                ColumnDefinition(
                    slug="doc_id", name="Document ID", type=ColumnType.STRING
                ),
                ColumnDefinition(slug="title", name="Title", type=ColumnType.STRING),
                ColumnDefinition(
                    slug="attachment", name="Attachment", type=ColumnType.FILE
                ),
                ColumnDefinition(
                    slug="reference_video", name="Reference Video", type=ColumnType.FILE
                ),
            ],
            rows=[
                {
                    "doc_id": "DOC001",
                    "title": "Project Overview",
                    "attachment": Attachment(
                        file_path=local_file.name,
                        file_type=FileCellType.FILE,
                        metadata={"author": "Team Lead"},
                    ),
                    "reference_video": ExternalAttachment(
                        url="https://www.youtube.com/watch?v=example",
                        file_type=FileCellType.VIDEO,
                        metadata={"relevance": "high"},
                    ),
                },
                {
                    "doc_id": "DOC002",
                    "title": "Technical Specs",
                    "attachment": Attachment(
                        data=b"Technical specification details...",
                        filename="tech_specs.txt",
                        file_type=FileCellType.FILE,
                    ),
                    "reference_video": None,  # No video for this document
                },
                {
                    "doc_id": "DOC003",
                    "title": "External Resources",
                    "attachment": ExternalAttachment(
                        url="https://github.com/example/repo/blob/main/README.md",
                        file_type=FileCellType.FILE,
                        metadata={"type": "markdown"},
                    ),
                    "reference_video": ExternalAttachment(
                        url="https://vimeo.com/example-tutorial",
                        file_type=FileCellType.VIDEO,
                    ),
                },
            ],
        )

        dataset = datasets.create(dataset_request)
        print(f"Created dataset: {dataset.slug}")

        # Show the different storage types
        for row in dataset.rows:
            print(f"\nDocument: {row.values['title']}")
            attachment = row.values.get("attachment")
            if attachment:
                print(f"  Attachment Storage: {attachment.get('storage')}")
                if attachment.get("storage") == "external":
                    print(f"  URL: {attachment.get('url')}")

    finally:
        os.unlink(local_file.name)


def main():
    """Run all examples."""
    print("=" * 60)
    print("Traceloop SDK Attachment Feature Examples")
    print("=" * 60)

    # Set your API key
    # os.environ["TRACELOOP_API_KEY"] = "your-api-key-here"

    # Note: These examples use mock data and won't actually upload to S3
    # In production, real files would be uploaded to S3 storage

    try:
        # Run examples
        example_external_attachments()
        example_file_uploads()
        example_in_memory_attachments()
        example_mixed_attachments()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to set TRACELOOP_API_KEY environment variable")


if __name__ == "__main__":
    main()
