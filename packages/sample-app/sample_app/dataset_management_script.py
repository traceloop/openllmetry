#!/usr/bin/env python3
"""
Dataset Management Script

This script provides comprehensive CRUD operations for Traceloop datasets:
- Get dataset by slug
- Add/update/delete rows
- Add/update/delete columns

Usage:
    python dataset_management_script.py
"""

from typing import Dict, Any, List, Optional
from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import Dataset, Column, Row, ColumnType

# Initialize Traceloop
Traceloop.init()

def list_all_datasets() -> List[Dataset]:
    """
    List all datasets using the new class method approach.
    """
    print("Listing all datasets...")
    
    try:
        # Use the new class method - no temporary instance needed!
        datasets = Dataset.get_all_datasets()
        print(f"Found {len(datasets)} datasets")
        for dataset_data in datasets:
            print(f"  - {dataset_data.name} (slug: {dataset_data.slug}, ID: {dataset_data.id})")
        return datasets
        
    except Exception as e:
        print(f"Error listing datasets: {e}")
        raise

def get_dataset_by_slug(slug: str) -> Dataset:
    """
    Get dataset by slug using the new class method approach.
    """
    print(f"Looking for dataset with slug: {slug}")
    
    try:
        # Use the new class method - no temporary instance needed!
        dataset = Dataset.get_by_slug(slug)
        print(f"Found dataset: {dataset.name} (ID: {dataset.id})")
        return dataset
        
    except Exception as e:
        print(f"Error retrieving dataset: {e}")
        raise

def add_row(dataset: Dataset, row_data: Dict[str, Any]) -> Row:
    """Add a new row to the dataset"""
    print(f"Adding row: {row_data}")
    
    try:
        # Convert column names to column IDs
        row_with_ids = {}
        for col_name, value in row_data.items():
            column = next((col for col in dataset.columns if col.name == col_name), None)
            if column:
                row_with_ids[column.id] = value
            else:
                print(f"Warning: Column '{col_name}' not found in dataset")
        
        # Add row via API
        rows_response = dataset.add_rows(dataset.slug, [row_with_ids])
        
        # Create Row object and add to dataset
        if rows_response.rows:
            row_obj = rows_response.rows[0]
            new_row = Row(
                id=row_obj.id,
                index=len(dataset.rows),
                values=row_obj.values,
                dataset_id=dataset.id,
                _client=dataset
            )
            dataset.rows.append(new_row)
            print(f"Successfully added row with ID: {new_row.id}")
            return new_row
        
    except Exception as e:
        print(f"Error adding row: {e}")
        raise

def update_row(dataset: Dataset, row_id: str, updates: Dict[str, Any]) -> None:
    """Update an existing row"""
    print(f"Updating row {row_id} with: {updates}")
    
    try:
        # Find the row
        row = next((r for r in dataset.rows if r.id == row_id), None)
        if not row:
            raise ValueError(f"Row with ID '{row_id}' not found")
        
        # Convert column names to column IDs if needed
        updates_with_ids = {}
        for key, value in updates.items():
            # Check if key is already a column ID
            if any(col.id == key for col in dataset.columns):
                updates_with_ids[key] = value
            else:
                # Try to find column by name
                column = next((col for col in dataset.columns if col.name == key), None)
                if column:
                    updates_with_ids[column.id] = value
                else:
                    print(f"Warning: Column '{key}' not found in dataset")
        
        # Update via Row object method
        row.update(updates_with_ids)
        print(f"Successfully updated row {row_id}")
        
    except Exception as e:
        print(f"Error updating row: {e}")
        raise

def delete_row(dataset: Dataset, row_id: str) -> None:
    """Delete a row from the dataset"""
    print(f"Deleting row {row_id}")
    
    try:
        # Find the row
        row = next((r for r in dataset.rows if r.id == row_id), None)
        if not row:
            raise ValueError(f"Row with ID '{row_id}' not found")
        
        # Delete via Row object method
        row.delete()
        
        # Remove from local list
        dataset.rows = [r for r in dataset.rows if r.id != row_id]
        print(f"Successfully deleted row {row_id}")
        
    except Exception as e:
        print(f"Error deleting row: {e}")
        raise

def add_column(dataset: Dataset, name: str, col_type: ColumnType, config: Optional[Dict[str, Any]] = None) -> Column:
    """Add a new column to the dataset"""
    print(f"Adding column: {name} ({col_type})")
    
    try:
        # Add column via Dataset method
        new_column = dataset.add_column(name, col_type, config)
        print(f"Successfully added column '{name}' with ID: {new_column.id}")
        return new_column
        
    except Exception as e:
        print(f"Error adding column: {e}")
        raise

def update_column(dataset: Dataset, column_id: str, name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None) -> None:
    """Update an existing column"""
    print(f"Updating column {column_id}")
    
    try:
        # Find the column
        column = next((col for col in dataset.columns if col.id == column_id), None)
        if not column:
            raise ValueError(f"Column with ID '{column_id}' not found")
        
        # Update via Column object method
        column.update(name=name, config=config)
        print(f"Successfully updated column {column_id}")
        
    except Exception as e:
        print(f"Error updating column: {e}")
        raise

def delete_column(dataset: Dataset, column_id: str) -> None:
    """Delete a column from the dataset"""
    print(f"Deleting column {column_id}")
    
    try:
        # Find the column
        column = next((col for col in dataset.columns if col.id == column_id), None)
        if not column:
            raise ValueError(f"Column with ID '{column_id}' not found")
        
        # Delete via Column object method
        column.delete()
        
        # Remove from local list
        dataset.columns = [col for col in dataset.columns if col.id != column_id]
        print(f"Successfully deleted column {column_id}")
        
    except Exception as e:
        print(f"Error deleting column: {e}")
        raise

def print_dataset_info(dataset: Dataset) -> None:
    """Print comprehensive dataset information"""
    print(f"\n=== Dataset Information ===")
    print(f"ID: {dataset.id}")
    print(f"Name: {dataset.name}")
    print(f"Slug: {dataset.slug}")
    print(f"Description: {dataset.description or 'None'}")
    print(f"Created: {dataset.created_at}")
    print(f"Updated: {dataset.updated_at}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Rows: {len(dataset.rows)}")
    
    print(f"Dataset has {len(dataset.columns)} columns:")
    for column in dataset.columns:
        print(f"  Column {column.id}: {column.name} ({column.type})")
    
    print(f"Dataset has {len(dataset.rows)} rows:")
    for row in dataset.rows:
        print(f"  Row {row.id}: {row.values}")

def main():
    """Simple example usage of the dataset management functions"""
    slug = "employee-data-1"
    
    try:
        # First, list all available datasets using the new class method
        print("=== All Available Datasets ===")
        list_all_datasets()
        
        # # Get dataset by slug using the improved approach
        # print(f"\n=== Getting Dataset by Slug ===")
        # dataset = get_dataset_by_slug(slug)
        # print_dataset_info(dataset)
        
        # # Example row operations
        # print("\n=== Example Operations ===")
        
        # # Add a new row
        # new_row_data = {"name": "New Employee", "age": 25, "city": "Austin", "salary": 80000}
        # add_row(dataset, new_row_data)
        
        # # Add a new column
        # add_column(dataset, "department", ColumnType.STRING)
        
        # # Show updated dataset
        # print("\n=== Updated Dataset ===")
        # print_dataset_info(dataset)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()