"""
Example script demonstrating the Traceloop Dataset functionality
"""
import os
import tempfile
from traceloop.sdk import Traceloop
from traceloop.sdk.datasets import Dataset, ColumnType
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

def dataset_from_csv_example():
    """Demonstrate creating a dataset from a CSV file"""
    print("=== Dataset from CSV Example ===")
    
    # Create sample CSV
    csv_file = create_sample_csv()
    print(f"Created sample CSV file: {csv_file}")
    
    try:
        # Create dataset from CSV
        dataset = Dataset.from_csv(
            file_path=csv_file,
            slug="employee-data-1",
            name="Employee Dataset",
            description="Sample employee data for demonstration"
        )
        
        print(f"Create dataset response: {dataset.model_dump_json()}")
            
    except Exception as e:
        print(f"Error creating dataset from CSV: {e}")
    finally:
        # Clean up temporary file
        os.unlink(csv_file)

def dataset_from_dataframe_example():
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
            slug="product-inventory",
            name="Product Inventory",
            description="Sample product inventory data"
        )
        
        print(f"Created dataset: {dataset.name}")
        print(f"Dataset ID: {dataset.id}")
        print(f"Number of columns: {len(dataset.columns)}")
        print(f"Number of rows: {len(dataset.rows)}")
        
        # Display column information with inferred types
        print("\nColumns with inferred types:")
        for col in dataset.columns:
            print(f"  - {col.name} ({col.type})")
        
        # Display all rows
        print("\nAll rows:")
        for i, row in enumerate(dataset.rows):
            print(f"  Row {i+1}: {row.values}")
            
    except ImportError:
        print("pandas not available. Install with: pip install pandas")
    except Exception as e:
        print(f"Error creating dataset from DataFrame: {e}")

def manual_dataset_example():
    """Demonstrate creating a dataset manually and adding columns/rows"""
    print("\n=== Manual Dataset Creation Example ===")
    
    try:
        # This would require a more complete implementation
        # For now, we'll show the basic structure
        print("Manual dataset creation would involve:")
        print("1. Creating a Dataset instance")
        print("2. Adding columns using add_column()")
        print("3. Adding rows programmatically")
        print("4. This requires additional API methods not shown in the current implementation")
        
    except Exception as e:
        print(f"Error with manual dataset creation: {e}")

def main():
    print("Traceloop Dataset Examples")
    print("=" * 50)
    
    # Run CSV example
    dataset_from_csv_example()
    
    # # Run DataFrame example
    # dataset_from_dataframe_example()
    
    # # Run manual example
    # manual_dataset_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")

if __name__ == "__main__":
    main()