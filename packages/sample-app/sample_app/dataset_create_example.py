"""
Example script demonstrating the Traceloop Dataset functionality
"""
import os
import tempfile
from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import Dataset, ColumnType, Column, Row
import pandas as pd


# Initialize Traceloop
Traceloop.init()


def create_sample_csv():
    """Create a sample CSV file for demonstration"""
    csv_content = """name,age,city,salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,85000
Bob Johnson,35,Chicago,65000
Alice Brown,28,Seattle,90000
Charlie Wilson,32,Boston,70000"""

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        return f.name


def dataset_from_csv_example(slug: str) -> Dataset:
    """Demonstrate creating a dataset from a CSV file"""
    print("=== Dataset from CSV Example ===")

    # Create sample CSV
    csv_file = create_sample_csv()
    print(f"Created sample CSV file: {csv_file}")

    try:
        # Create dataset from CSV
        dataset = Dataset.from_csv(
            file_path=csv_file,
            slug=slug,
            name="Employee Dataset",
            description="Sample employee data for demonstration"
        )

        print(f"Created dataset from CSV successfully")
        return dataset

    except Exception as e:
        print(f"Error creating dataset from CSV: {e}")
        return None
    finally:
        # Clean up temporary file
        os.unlink(csv_file)


def dataset_from_dataframe_example(slug: str) -> Dataset:
    """Demonstrate creating a dataset from a pandas DataFrame"""
    print("\n=== Dataset from DataFrame Example ===")

    try:
        # Create sample DataFrame
        data = {
            'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
            'price': [999.99, 29.99, 79.99, 299.99],
            'in_stock': [True, True, False, True],
            'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics']
        }
        df = pd.DataFrame(data)

        # Create dataset from DataFrame
        dataset = Dataset.from_dataframe(
            df=df,
            slug=slug,
            name="Product Inventory",
            description="Sample product inventory data"
        )

        print(f"Created dataset from DataFrame successfully")
        return dataset

    except Exception as e:
        print(f"Error creating dataset from DataFrame: {e}")
        return None


def add_column_example(dataset: Dataset) -> Column:
    """Demonstrate adding a column to a dataset"""
    print("\n=== Add Column Example ===")
    
    try:
        num_columns = len(dataset.columns)
        new_column = dataset.add_column(
            name="department",
            col_type=ColumnType.STRING,
            config={"description": "Employee department"}
        )
        print(f"Added column: {new_column.name} (ID: {new_column.id})")
    
        assert len(dataset.columns) == num_columns + 1
        return new_column
        
    except Exception as e:
        print(f"Error adding column: {e}")


def update_column_example(dataset: Dataset, column: Column):
    """Demonstrate updating a column in a dataset"""
    print("\n=== Update Column Example ===")
    
    try:
        column.update(name=f"{column.name}_updated", type=ColumnType.NUMBER)
        print(f"Updated column: {column.name} (ID: {column.id})")
        
        updated_col = dataset.columns.find(lambda c: c.id == column.id)
        assert updated_col is not None
        assert updated_col.name == f"{column.name}_updated"
        assert updated_col.type == ColumnType.NUMBER
            
    except Exception as e:
        print(f"Error updating column: {e}")


def delete_column_example(dataset: Dataset, column: Column):
    """Demonstrate deleting a column from a dataset"""
    print("\n=== Delete Column Example ===")
    
    try:
        column.delete()
        print(f"Deleted column: {column.name} (ID: {column.id})")
        assert column.id not in [c.id for c in dataset.columns]
            
    except Exception as e:
        print(f"Error deleting column: {e}")


def add_row_example(dataset: Dataset) -> Row:
    """Demonstrate adding a row to a dataset"""
    print("\n=== Add Row Example ===")
    
    try:
        num_rows = len(dataset.rows)
        
        # Create row data using column IDs
        row_data = {}
        for column in dataset.columns:
            if column.name == "name":
                row_data[column.id] = "New Employee"
            elif column.name == "age":
                row_data[column.id] = 27
            elif column.name == "city":
                row_data[column.id] = "Austin"
            elif column.name == "salary":
                row_data[column.id] = 80000
            elif column.name == "department":
                row_data[column.id] = "Engineering"
            else:
                row_data[column.id] = "Default Value"
        
        # Add row via API
        rows_response = dataset.add_rows([row_data])
        
        if rows_response.rows:
            new_row = rows_response.rows[0]
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
                updates[column.id] = "Updated Product"
            elif column.name == "price":
                updates[column.id] = 28.0
            elif column.name == "in_stock":
                updates[column.id] = True
            elif column.name == "category":
                updates[column.id] = "Marketing"
        
        # Update the row
        row.update(updates)
        updated_row = dataset.rows.find(lambda r: r.id == row.id)
        assert updated_row is not None
        print(f"Updated row: {updated_row.model_dump_json()}")
        
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


def delete_dataset_example(slug: str):
    """Demonstrate deleting a dataset"""
    print("\n=== Delete Dataset Example ===")
    Dataset.delete_by_slug(slug)
    print("Dataset deleted")


def main():
    print("Traceloop Dataset Examples")
    print("=" * 50)

    ds1 = dataset_from_csv_example("example-1")
    column = add_column_example(ds1)
    update_column_example(ds1, column)
    add_row_example(ds1)
    delete_row_example(ds1)
    delete_column_example(ds1, column)
    delete_dataset_example(ds1.slug)


    ds2 = dataset_from_dataframe_example("example-2")
    column = add_column_example(ds2)
    update_column_example(ds2, column)
    update_row_example(ds2)
    delete_column_example(ds2, column)
    delete_dataset_example(ds2.slug)

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
