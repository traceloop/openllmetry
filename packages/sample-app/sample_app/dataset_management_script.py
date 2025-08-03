#!/usr/bin/env python3
"""
Dataset Management Script

This script provides comprehensive CRUD operations for Traceloop datasets:
- Get dataset by slug
- Add/update/delete rows
- Add/update/delete columns

Usage:
    python dataset_management_script.py <dataset_slug> [options]
"""

import argparse
import sys
from typing import Dict, Any, List, Optional
from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import Dataset, Column, Row, ColumnType


class DatasetManager:
    """Manager class for dataset CRUD operations"""
    
    def __init__(self):
        """Initialize Traceloop SDK"""
        Traceloop.init()
        self.dataset: Optional[Dataset] = None
    
    def get_dataset_by_slug(self, slug: str) -> Dataset:
        """
        Get dataset by slug. Since there's no direct get_by_slug method,
        we'll search through all datasets to find the one with matching slug.
        """
        print(f"Looking for dataset with slug: {slug}")
        
        # Create a temporary dataset instance to access API methods
        temp_dataset = Dataset(name="temp", slug="temp")
        
        try:
            # Get all datasets
            all_datasets = temp_dataset.get_all_datasets_api()
            
            # Find dataset with matching slug
            for dataset_data in all_datasets:
                if dataset_data.get("slug") == slug:
                    print(f"Found dataset: {dataset_data['name']} (ID: {dataset_data['id']})")
                    
                    # Get full dataset details
                    full_dataset = temp_dataset.get_dataset_api(dataset_data["id"])
                    
                    # Create Dataset instance with full data
                    dataset = Dataset(
                        id=full_dataset["id"],
                        name=full_dataset["name"],
                        slug=full_dataset["slug"],
                        description=full_dataset.get("description"),
                        created_at=full_dataset.get("created_at"),
                        updated_at=full_dataset.get("updated_at")
                    )
                    
                    # Populate columns and rows if available
                    if "columns" in full_dataset:
                        for col_id, col_data in full_dataset["columns"].items():
                            column = Column(
                                id=col_id,
                                name=col_data["name"],
                                type=ColumnType(col_data["type"]),
                                dataset_id=dataset.id,
                                _client=dataset
                            )
                            dataset.columns.append(column)
                    
                    if "rows" in full_dataset:
                        for idx, row_data in enumerate(full_dataset["rows"]):
                            row = Row(
                                id=row_data["id"],
                                index=idx,
                                values=row_data["values"],
                                dataset_id=dataset.id,
                                _client=dataset
                            )
                            dataset.rows.append(row)
                    
                    self.dataset = dataset
                    return dataset
            
            raise ValueError(f"Dataset with slug '{slug}' not found")
            
        except Exception as e:
            print(f"Error retrieving dataset: {e}")
            raise
    
    def add_row(self, row_data: Dict[str, Any]) -> Row:
        """Add a new row to the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Adding row: {row_data}")
        
        try:
            # Convert column names to column IDs
            row_with_ids = {}
            for col_name, value in row_data.items():
                column = next((col for col in self.dataset.columns if col.name == col_name), None)
                if column:
                    row_with_ids[column.id] = value
                else:
                    print(f"Warning: Column '{col_name}' not found in dataset")
            
            # Add row via API
            rows_response = self.dataset.add_rows(self.dataset.slug, [row_with_ids])
            
            # Create Row object and add to dataset
            if rows_response.rows:
                row_obj = rows_response.rows[0]
                new_row = Row(
                    id=row_obj.id,
                    index=len(self.dataset.rows),
                    values=row_obj.values,
                    dataset_id=self.dataset.id,
                    _client=self.dataset
                )
                self.dataset.rows.append(new_row)
                print(f"Successfully added row with ID: {new_row.id}")
                return new_row
            
        except Exception as e:
            print(f"Error adding row: {e}")
            raise
    
    def update_row(self, row_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing row"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Updating row {row_id} with: {updates}")
        
        try:
            # Find the row
            row = next((r for r in self.dataset.rows if r.id == row_id), None)
            if not row:
                raise ValueError(f"Row with ID '{row_id}' not found")
            
            # Convert column names to column IDs if needed
            updates_with_ids = {}
            for key, value in updates.items():
                # Check if key is already a column ID
                if any(col.id == key for col in self.dataset.columns):
                    updates_with_ids[key] = value
                else:
                    # Try to find column by name
                    column = next((col for col in self.dataset.columns if col.name == key), None)
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
    
    def delete_row(self, row_id: str) -> None:
        """Delete a row from the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Deleting row {row_id}")
        
        try:
            # Find the row
            row = next((r for r in self.dataset.rows if r.id == row_id), None)
            if not row:
                raise ValueError(f"Row with ID '{row_id}' not found")
            
            # Delete via Row object method
            row.delete()
            
            # Remove from local list
            self.dataset.rows = [r for r in self.dataset.rows if r.id != row_id]
            print(f"Successfully deleted row {row_id}")
            
        except Exception as e:
            print(f"Error deleting row: {e}")
            raise
    
    def add_column(self, name: str, col_type: ColumnType, config: Optional[Dict[str, Any]] = None) -> Column:
        """Add a new column to the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Adding column: {name} ({col_type})")
        
        try:
            # Add column via Dataset method
            new_column = self.dataset.add_column(name, col_type, config)
            print(f"Successfully added column '{name}' with ID: {new_column.id}")
            return new_column
            
        except Exception as e:
            print(f"Error adding column: {e}")
            raise
    
    def update_column(self, column_id: str, name: Optional[str] = None, 
                     config: Optional[Dict[str, Any]] = None) -> None:
        """Update an existing column"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Updating column {column_id}")
        
        try:
            # Find the column
            column = next((col for col in self.dataset.columns if col.id == column_id), None)
            if not column:
                raise ValueError(f"Column with ID '{column_id}' not found")
            
            # Update via Column object method
            column.update(name=name, config=config)
            print(f"Successfully updated column {column_id}")
            
        except Exception as e:
            print(f"Error updating column: {e}")
            raise
    
    def delete_column(self, column_id: str) -> None:
        """Delete a column from the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Deleting column {column_id}")
        
        try:
            # Find the column
            column = next((col for col in self.dataset.columns if col.id == column_id), None)
            if not column:
                raise ValueError(f"Column with ID '{column_id}' not found")
            
            # Delete via Column object method
            column.delete()
            
            # Remove from local list
            self.dataset.columns = [col for col in self.dataset.columns if col.id != column_id]
            print(f"Successfully deleted column {column_id}")
            
        except Exception as e:
            print(f"Error deleting column: {e}")
            raise
    
    def list_rows(self) -> List[Row]:
        """List all rows in the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Dataset has {len(self.dataset.rows)} rows:")
        for row in self.dataset.rows:
            print(f"  Row {row.id}: {row.values}")
        
        return self.dataset.rows
    
    def list_columns(self) -> List[Column]:
        """List all columns in the dataset"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"Dataset has {len(self.dataset.columns)} columns:")
        for column in self.dataset.columns:
            print(f"  Column {column.id}: {column.name} ({column.type})")
        
        return self.dataset.columns
    
    def print_dataset_info(self) -> None:
        """Print comprehensive dataset information"""
        if not self.dataset:
            raise ValueError("No dataset loaded. Use get_dataset_by_slug first.")
        
        print(f"\n=== Dataset Information ===")
        print(f"ID: {self.dataset.id}")
        print(f"Name: {self.dataset.name}")
        print(f"Slug: {self.dataset.slug}")
        print(f"Description: {self.dataset.description or 'None'}")
        print(f"Created: {self.dataset.created_at}")
        print(f"Updated: {self.dataset.updated_at}")
        print(f"Columns: {len(self.dataset.columns)}")
        print(f"Rows: {len(self.dataset.rows)}")
        
        self.list_columns()
        print()
        self.list_rows()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Manage Traceloop datasets')
    parser.add_argument('slug', help='Dataset slug')
    
    # Operation subcommands
    subparsers = parser.add_subparsers(dest='operation', help='Available operations')
    
    # Info command
    subparsers.add_parser('info', help='Display dataset information')
    
    # Row operations
    row_parser = subparsers.add_parser('row', help='Row operations')
    row_subparsers = row_parser.add_subparsers(dest='row_action')
    
    row_add = row_subparsers.add_parser('add', help='Add a new row')
    row_add.add_argument('--data', type=str, required=True, 
                        help='Row data as JSON string, e.g., \'{"name": "John", "age": 30}\'')
    
    row_update = row_subparsers.add_parser('update', help='Update an existing row')
    row_update.add_argument('--id', type=str, required=True, help='Row ID')
    row_update.add_argument('--data', type=str, required=True, 
                           help='Update data as JSON string')
    
    row_delete = row_subparsers.add_parser('delete', help='Delete a row')
    row_delete.add_argument('--id', type=str, required=True, help='Row ID')
    
    # Column operations
    col_parser = subparsers.add_parser('column', help='Column operations')
    col_subparsers = col_parser.add_subparsers(dest='col_action')
    
    col_add = col_subparsers.add_parser('add', help='Add a new column')
    col_add.add_argument('--name', type=str, required=True, help='Column name')
    col_add.add_argument('--type', type=str, required=True, 
                        choices=['string', 'number', 'boolean', 'json'],
                        help='Column type')
    
    col_update = col_subparsers.add_parser('update', help='Update an existing column')
    col_update.add_argument('--id', type=str, required=True, help='Column ID')
    col_update.add_argument('--name', type=str, help='New column name')
    
    col_delete = col_subparsers.add_parser('delete', help='Delete a column')
    col_delete.add_argument('--id', type=str, required=True, help='Column ID')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Initialize manager
    manager = DatasetManager()
    
    try:
        # Load dataset
        manager.get_dataset_by_slug(args.slug)
        
        if not args.operation or args.operation == 'info':
            manager.print_dataset_info()
            return
        
        # Handle row operations
        if args.operation == 'row':
            if args.row_action == 'add':
                import json
                row_data = json.loads(args.data)
                manager.add_row(row_data)
            elif args.row_action == 'update':
                import json
                update_data = json.loads(args.data)
                manager.update_row(args.id, update_data)
            elif args.row_action == 'delete':
                manager.delete_row(args.id)
        
        # Handle column operations
        elif args.operation == 'column':
            if args.col_action == 'add':
                col_type = ColumnType(args.type)
                manager.add_column(args.name, col_type)
            elif args.col_action == 'update':
                manager.update_column(args.id, name=args.name)
            elif args.col_action == 'delete':
                manager.delete_column(args.id)
        
        # Show updated dataset info
        print("\n" + "="*50)
        manager.print_dataset_info()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()