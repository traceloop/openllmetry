"""
Example script demonstrating the Traceloop Dataset functionality
"""

import os
import tempfile
from typing import Optional
from datetime import datetime
from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import Dataset, ColumnType, Column, Row
import pandas as pd
import openai


# Initialize Traceloop
client = Traceloop.init()


def create_sample_csv():
    """Create a sample CSV file for demonstration"""
    csv_content = """Name,Age,City,Salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,85000
Bob Johnson,35,Chicago,65000
Alice Brown,28,Seattle,90000
Charlie Wilson,32,Boston,70000"""

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        return f.name


def dataset_from_csv_example(slug: str) -> Optional[Dataset]:
    """Demonstrate creating a dataset from a CSV file"""
    print("=== Dataset from CSV Example ===")

    # Create sample CSV
    csv_file = create_sample_csv()
    print(f"Created sample CSV file: {csv_file}")

    try:
        # Create dataset from CSV
        dataset = client.datasets.from_csv(
            file_path=csv_file,
            slug=slug,
            name="Employee Dataset",
            description="Sample employee data for demonstration",
        )

        print("Created dataset from CSV successfully")
        return dataset

    except Exception as e:
        print(f"Error creating dataset from CSV: {e}")
        return None
    finally:
        # Clean up temporary file
        os.unlink(csv_file)


def dataset_from_dataframe_example(slug: str) -> Optional[Dataset]:
    """Demonstrate creating a dataset from a pandas DataFrame"""
    print("\n=== Dataset from DataFrame Example ===")

    try:
        # Create sample DataFrame
        data = {
            "product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
            "price": [999.99, 29.99, 79.99, 299.99],
            "in_stock": [True, True, False, True],
            "category": ["Electronics", "Accessories", "Accessories", "Electronics"],
        }
        df = pd.DataFrame(data)

        # Create dataset from DataFrame
        dataset = client.datasets.from_dataframe(
            df=df,
            slug=slug,
            name="Product Inventory",
            description="Sample product inventory data",
        )

        print("Created dataset from DataFrame successfully")
        return dataset

    except Exception as e:
        print(f"Error creating dataset from DataFrame: {e}")
        return None


def add_column_example(dataset: Dataset) -> Optional[Column]:
    """Demonstrate adding a column to a dataset"""
    print("\n=== Add Column Example ===")

    try:
        num_columns = len(dataset.columns)
        new_column = dataset.add_column(
            slug="department",
            name="department",
            col_type=ColumnType.STRING,
        )
        print(f"Added column: {new_column.name} (Slug: {new_column.slug})")

        assert len(dataset.columns) == num_columns + 1
        return new_column

    except Exception as e:
        print(f"Error adding column: {e}")
        return None


def update_column_example(dataset: Dataset, column: Column):
    """Demonstrate updating a column in a dataset"""
    print("\n=== Update Column Example ===")

    try:
        updated_name = "updated_name"
        column.update(name=updated_name, type=ColumnType.NUMBER)
        print(f"Updated column: {column.name} (Slug: {column.slug})")

        updated_col = next((c for c in dataset.columns if c.slug == column.slug), None)
        assert updated_col is not None
        assert updated_col.name == updated_name
        assert updated_col.type == ColumnType.NUMBER

    except Exception as e:
        print(f"Error updating column: {e}")


def delete_column_example(dataset: Dataset, column: Column):
    """Demonstrate deleting a column from a dataset"""
    print("\n=== Delete Column Example ===")

    try:
        column.delete()
        print(f"Deleted column: {column.name} (Slug: {column.slug})")
        assert column.slug not in [c.slug for c in dataset.columns]

    except Exception as e:
        print(f"Error deleting column: {e}")


def add_row_example(dataset: Dataset) -> Optional[Row]:
    """Demonstrate adding a row to a dataset"""
    print("\n=== Add Row Example ===")

    try:
        num_rows = len(dataset.rows)

        # Create row data
        row_data = {}
        for column in dataset.columns:
            if column.name == "product":
                row_data[column.slug] = "Updated Product"
            elif column.name == "price":
                row_data[column.slug] = 28.0
            elif column.name == "in_stock":
                row_data[column.slug] = True
            elif column.name == "category":
                row_data[column.slug] = "Marketing"

        dataset.add_rows([row_data])

        if dataset.rows and dataset.rows[0]:
            new_row = dataset.rows[0]
            print(f"Added row with ID: {new_row.id}")
            assert len(dataset.rows) == num_rows + 1
            assert new_row.id in [r.id for r in dataset.rows]
            return new_row
        else:
            print("No row was added")
            return None

    except Exception as e:
        print(f"Error adding row: {e}")
        return None


def update_row_example(dataset: Dataset):
    """Demonstrate updating a row in a dataset"""
    print("\n=== Update Row Example ===")

    try:
        if not dataset.rows:
            print("No rows to update")
            return

        # Get the first row
        row = dataset.rows[0]
        print(f"Updating row: {row.id}")

        # Create update data
        updates = {}
        for column in dataset.columns:
            if column.name == "product":
                updates[column.slug] = "Updated Product"
            elif column.name == "price":
                updates[column.slug] = 28.0
            elif column.name == "in_stock":
                updates[column.slug] = True
            elif column.name == "category":
                updates[column.slug] = "Marketing"

        # Update the row
        row.update(updates)
        updated_row = next((r for r in dataset.rows if r.id == row.id), None)
        assert updated_row is not None
        print(f"Updated row: ID={updated_row.id}, values={updated_row.values}")

    except Exception as e:
        print(f"Error updating row: {e}")


def delete_row_example(dataset: Dataset):
    """Demonstrate deleting a row from a dataset"""
    print("\n=== Delete Row Example ===")

    try:
        num_rows = len(dataset.rows)
        if not dataset.rows:
            print("No rows to delete")
            return

        # Get the first row
        row = dataset.rows[0]
        row_id = row.id
        print(f"Deleting row: {row_id}")

        # Delete the row
        row.delete()
        print(f"Deleted row: {row_id}")

        # Verify row is no longer in dataset
        assert row_id not in [r.id for r in dataset.rows]
        assert len(dataset.rows) == num_rows - 1

    except Exception as e:
        print(f"Error deleting row: {e}")


def publish_dataset_example(dataset: Dataset) -> Optional[str]:
    """Demonstrate publishing a dataset"""
    print("\n=== Publish Dataset Example ===")

    try:
        published_version = dataset.publish()
        print(f"Published dataset: {dataset.slug}, version: {published_version}")
        return published_version

    except Exception as e:
        print(f"Error publishing dataset: {e}")
        return None


def get_dataset_by_version_example(slug: str, version: str):
    """Demonstrate getting a dataset by version"""
    print("\n=== Get Dataset by Version Example ===")

    try:
        dataset_csv = client.datasets.get_version_csv(slug=slug, version=version)
        print(f"Retrieved dataset:\n{dataset_csv}")

    except Exception as e:
        print(f"Error getting dataset by version: {e}")


def get_dataset_by_slug_example(slug: str):
    """Demonstrate getting a dataset by slug"""
    print("\n=== Get Dataset by Slug Example ===")
    try:
        dataset = client.datasets.get_by_slug(slug)
        print(f"Retrieved dataset: {dataset}")
    except Exception as e:
        print(f"Error getting dataset by slug: {e}")


def delete_dataset_example(slug: str):
    """Demonstrate deleting a dataset"""
    print("\n=== Delete Dataset Example ===")
    try:
        client.datasets.delete_by_slug(slug)
        print("Dataset deleted")
    except Exception as e:
        print(f"Error deleting dataset: {e}")


def create_customer_support_dataset(slug: str) -> Optional[Dataset]:
    """Create a dataset with real OpenAI customer support interactions"""
    print("\n=== Customer Support Dataset with OpenAI ===")

    try:
        # Sample customer queries
        queries = [
            "How do I reset my password?",
            "My order hasn't arrived yet, what should I do?",
            "Can I return this item after 30 days?",
            "Do you offer international shipping?",
            "What's your refund policy?",
        ]

        data = []
        for query in queries:
            try:
                # Make OpenAI call for customer support response
                client_openai = openai.OpenAI()
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful customer support agent for an e-commerce company.",
                        },
                        {"role": "user", "content": query},
                    ],
                    max_tokens=150,
                    temperature=0.7,
                )

                ai_response = response.choices[0].message.content

                data.append(
                    {
                        "customer_query": query,
                        "ai_response": ai_response,
                        "timestamp": datetime.now().isoformat(),
                        "query_category": "general_support",
                        "resolved": True,
                    }
                )
                print(f"Generated response for: {query[:50]}...")

            except Exception as e:
                print(f"Error generating response for query '{query}': {e}")
                # Add fallback data
                data.append(
                    {
                        "customer_query": query,
                        "ai_response": """I apologize, but I'm unable to process your request at the moment.
                            Please contact our support team directly.""",
                        "timestamp": datetime.now().isoformat(),
                        "query_category": "general_support",
                        "resolved": False,
                    }
                )

        # Create DataFrame and dataset
        df = pd.DataFrame(data)

        dataset = client.datasets.from_dataframe(
            df=df,
            slug=slug,
            name="Customer Support Interactions",
            description="Dataset of customer queries and AI-generated support responses",
        )

        print(f"Created customer support dataset with {len(data)} interactions")
        return dataset

    except Exception as e:
        print(f"Error creating customer support dataset: {e}")
        return None


def create_translation_dataset(slug: str) -> Optional[Dataset]:
    """Create a dataset with OpenAI translation examples"""
    print("\n=== Translation Dataset with OpenAI ===")

    try:
        # Sample phrases to translate
        phrases = [
            "Hello, how are you today?",
            "Thank you for your help.",
            "Where is the nearest restaurant?",
            "I would like to book a room.",
            "What time does the store close?",
        ]

        languages = ["Spanish", "French", "German", "Italian"]

        data = []
        for phrase in phrases:
            for lang in languages:
                try:
                    # Use OpenAI for translation
                    client_openai = openai.OpenAI()
                    response = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are a professional translator.
                                Translate the following English text to {lang}.
                                Provide only the translation.""",
                            },
                            {"role": "user", "content": phrase},
                        ],
                        max_tokens=100,
                        temperature=0.3,
                    )

                    translation = response.choices[0].message.content.strip()

                    data.append(
                        {
                            "source_text": phrase,
                            "target_language": lang,
                            "translation": translation,
                            "timestamp": datetime.now().isoformat(),
                            "confidence_score": 0.95,  # Mock confidence score
                        }
                    )
                    print(f"Translated '{phrase}' to {lang}")

                except Exception as e:
                    print(f"Error translating '{phrase}' to {lang}: {e}")
                    data.append(
                        {
                            "source_text": phrase,
                            "target_language": lang,
                            "translation": f"[Translation error for {lang}]",
                            "timestamp": datetime.now().isoformat(),
                            "confidence_score": 0.0,
                        }
                    )

        # Create DataFrame and dataset
        df = pd.DataFrame(data)

        dataset = client.datasets.from_dataframe(
            df=df,
            slug=slug,
            name="Translation Results",
            description="Dataset of text translations generated using AI",
        )

        print(f"Created translation dataset with {len(data)} translations")
        return dataset

    except Exception as e:
        print(f"Error creating translation dataset: {e}")
        return None


def main():
    print("Traceloop Dataset Examples")
    print("=" * 50)

    ds1 = dataset_from_csv_example("sdk-example-1")
    if ds1:
        column = add_column_example(ds1)
        if column:
            update_column_example(ds1, column)
        published_version = publish_dataset_example(ds1)
        delete_row_example(ds1)
        if column:
            delete_column_example(ds1, column)
        get_dataset_by_slug_example(slug="sdk-example-1")
        if published_version:
            get_dataset_by_version_example(
                slug="sdk-example-1", version=published_version
            )
        delete_dataset_example(ds1.slug)

    ds2 = dataset_from_dataframe_example("sdk-example-2")
    if ds2:
        column = add_column_example(ds2)
        if column:
            update_column_example(ds2, column)
        add_row_example(ds2)
        update_row_example(ds2)
        if column:
            delete_column_example(ds2, column)
        published_version = publish_dataset_example(ds2)
        if published_version:
            get_dataset_by_version_example(
                slug="sdk-example-2", version=published_version
            )
        delete_dataset_example(ds2.slug)

    # OpenAI examples
    print("\n" + "=" * 50)
    print("OpenAI Integration Examples")
    print("=" * 50)

    # Customer Support Dataset
    support_ds = create_customer_support_dataset("openai-support-example")
    if support_ds:
        published_version = publish_dataset_example(support_ds)
        if published_version:
            get_dataset_by_version_example(
                slug="openai-support-example", version=published_version
            )
        delete_dataset_example(support_ds.slug)

    # Translation Dataset
    translation_ds = create_translation_dataset("openai-translation-example")
    if translation_ds:
        published_version = publish_dataset_example(translation_ds)
        if published_version:
            get_dataset_by_version_example(
                slug="openai-translation-example", version=published_version
            )
        delete_dataset_example(translation_ds.slug)

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
