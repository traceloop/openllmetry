# Datasets Feature Plan

## Overview

The Datasets feature will enable users to define, manage, and manipulate simple datasets within the traceloop-sdk package. This feature will provide a lean, Pythonic interface for dataset operations while integrating with the existing Traceloop API infrastructure.

## Architecture

Following the established pattern from `traceloop/sdk/prompts/`, the datasets feature will be structured as:

```
traceloop/sdk/datasets/
├── __init__.py          # Main API exports
├── client.py            # DatasetClient for API communication  
├── model.py             # Pydantic models (Dataset, Row, Column)
└── registry.py          # Optional: Local dataset registry/cache
```

## Core Classes

### 1. Dataset
Primary interface for dataset operations.

**Attributes:**
- `id`: str - Dataset ID from API
- `name`: str - Human-readable name
- `slug`: str - URL-friendly identifier 
- `description`: str - Dataset description
- `columns`: List[Column] - Dataset columns
- `rows`: List[Row] - Dataset rows (lazy-loaded)
- `created_at`: datetime
- `updated_at`: datetime

**Key Methods:**
- `from_csv(file_path, slug (mandatory),name (optoinal), description (optional))` - Class method to create from CSV
- `from_dataframe(df, slug (mandatory),name (optoinal), description (optional))` - Class method to create from pandas DataFrame
- `to_dataframe()` - Export to pandas DataFrame
- `add_rows_from_dataframe(df)` - Bulk add rows from DataFrame
- `add_rows_from_dict(data)` - Add rows from dictionary/list of dictionaries
- `add_column(name, col_type, config=None)` - Add new column (returns Column object)
- `publish_version()` - Create new version snapshot
- `get_version(version_id)` - Get specific version as DataFrame
- `get_all_versions()` - List all versions

### 2. Column
Represents a dataset column with type and configuration, and provides methods for column manipulation.

**Attributes:**
- `id`: str - Column ID
- `name`: str - Column name`
- `type`: ColumnType - Column type enum
- `config`: dict - Type-specific configuration
- `dataset_id`: str - Parent dataset ID
- `_client`: DatasetClient - Reference to client for API calls

**Methods:**
- `delete()` - Remove this column from dataset
- `update(name=None, config=None)` - Update this column's properties

### 3. Row
Represents a dataset row with values and provides methods for row manipulation.

**Attributes:**
- `id`: str - Row ID
- `index`: int - Row index in dataset
- `values`: dict - Column ID -> value mapping
- `dataset_id`: str - Parent dataset ID
- `_client`: DatasetClient - Reference to client for API calls

**Methods:**
- `delete()` - Remove this row from dataset
- `update(values)` - Update this row's values

## Column Types (ColumnType Enum)

Based on the API documentation, supported column types:
- `STRING` - text input
- `BOOLEAN` - True/false values
- `NUMBER` - Numerical metrics
- `JSON` - json string representation

## Client Architecture

### DatasetClient
Singleton client following the `PromptRegistryClient` pattern for API communication.

**Methods:**
- `create_dataset(name, description=None)` - Create new dataset
- `get_dataset(dataset_id)` - Retrieve dataset by ID
- `get_all_datasets()` - List all datasets
- `delete_dataset(dataset_id)` - Delete dataset
- `add_column(dataset_id, name, col_type, config=None)` - Add column
- `update_column(dataset_id, column_id, **kwargs)` - Update column
- `delete_column(dataset_id, column_id)` - Delete column
- `add_row(dataset_id, values)` - Add single row
- `delete_row(dataset_id, row_id)` - Delete row
- `update_cells(dataset_id, updates)` - Bulk update cells
- `update_column_order(dataset_id, column_ids)` - Reorder columns

## Implementation Plan

### Phase 1: Core Models and Client (Week 1)
1. **Create base structure** (`traceloop/sdk/datasets/`)
2. **Implement Pydantic models** in `model.py`
   - Column, Row, Dataset classes with proper typing
   - ColumnType enum
   - Request/response models for API
3. **Build DatasetClient** in `client.py`
   - Singleton pattern like PromptRegistryClient
   - HTTP client for API calls
   - Error handling and response parsing
4. **Create main exports** in `__init__.py`

### Phase 2: Dataset Core Operations (Week 2)
1. **Dataset creation methods**
   - `Dataset.from_csv()` with CSV parsing
   - `Dataset.from_dataframe()` with pandas integration
2. **Basic CRUD operations**
   - Create, read, update, delete for datasets
   - Column management (add, update, delete)
   - Row management (add, update, delete)
3. **Data export**
   - `Dataset.to_dataframe()` method
   - Proper data type conversion

### Phase 3: Bulk Operations & Advanced Features (Week 3)
1. **Bulk data operations**
   - `add_rows_from_dataframe()`
   - `add_rows_from_dict()`
   - Batch cell updates
2. **Version management**
   - `publish_version()` implementation
   - `get_version()` and `get_all_versions()`
3. **Data validation**
   - Column type validation
   - Required field checks

### Phase 4: Testing & Documentation (Week 4)
1. **Unit tests**
   - Model validation tests
   - Client API interaction tests with VCR.py
   - Dataset operation tests
2. **Integration tests**
   - End-to-end workflow tests
   - CSV/DataFrame import/export tests
3. **Documentation**
   - API reference documentation
   - Usage examples
   - Migration guide

## API Integration

The implementation will use the existing Traceloop API endpoints:
- `GET/POST /v2/datasets` - Dataset CRUD
- `POST/PUT/DELETE /v2/datasets/{id}/columns` - Column management
- `POST/DELETE /v2/datasets/{id}/rows` - Row management  
- `PUT /v2/datasets/{id}/cells` - Cell updates
- `PUT /v2/datasets/{id}/column-order` - Column ordering

## Usage Examples

### Basic Usage
```python
from traceloop.sdk.datasets import Dataset

# Create from CSV
dataset = Dataset.from_csv(
    "data.csv", 
    name="Customer Data", 
    slug="customer-data",
    description="Customer information dataset"
)

# Add a new column (returns Column object)
status_column = dataset.add_column(
    "status", 
    ColumnType.SELECT,
    config={"options": ["active", "inactive"]}
)

# Add rows from dictionary
dataset.add_rows_from_dict([
    {"name": "John Doe", "email": "john@example.com", "status": "active"},
    {"name": "Jane Smith", "email": "jane@example.com", "status": "inactive"}
])

# Export to DataFrame
df = dataset.to_dataframe()
```

### Advanced Operations
```python
# Create from existing DataFrame
import pandas as pd
df = pd.read_csv("large_dataset.csv")
dataset = Dataset.from_dataframe(
    df,
    name="Large Dataset",
    slug="large-dataset", 
    description="Imported from CSV"
)

# Update specific row using row object
row = dataset.rows[0]  # or get specific row by ID
row.update(values={"status": "active"})

# Delete a row using row object
row.delete()

# Update a column using column object
column = dataset.columns[0]  # or get by name/ID
column.update(name="updated_name", config={"new": "config"})

# Delete a column using column object
column.delete()

# Publish a version
version = dataset.publish_version()

# Get historical version
old_df = dataset.get_version(version_id="v1")
```

## Dependencies

### New Dependencies
- `pandas` - DataFrame operations (optional, with graceful fallback)
- `requests` - HTTP client (or reuse existing HTTP infrastructure)

### Existing Dependencies  
- `pydantic` - Model validation (already in use)
- `typing` - Type hints (already in use)

## Error Handling

Following the established pattern, implement proper error handling:
- Custom exception classes (`DatasetError`, `ColumnError`, `RowError`)
- API error response mapping
- Validation error messages
- Graceful handling of missing optional dependencies

## Testing Strategy

### Unit Tests
- Model validation and serialization
- Client method functionality
- Data type conversion accuracy

### Integration Tests
- API interaction with VCR.py cassettes
- CSV/DataFrame import/export workflows
- End-to-end dataset management scenarios

### Performance Tests
- Large dataset handling
- Bulk operations efficiency
- Memory usage for large DataFrames

## Security Considerations

- Input validation for all user data
- API authentication using existing Traceloop credentials
- Sanitization of CSV/DataFrame imports
- Rate limiting consideration for bulk operations

## Backward Compatibility

The datasets feature will be additive:
- No changes to existing SDK functionality
- Optional import (`from traceloop.sdk.datasets import Dataset`)
- Graceful handling of missing pandas dependency
- Version compatibility with existing API endpoints

## Success Metrics

1. **Functionality**: All planned features implemented and tested
2. **Performance**: Handle datasets up to 10k rows efficiently
3. **Usability**: Simple 3-line setup for basic use cases
4. **Integration**: Seamless pandas DataFrame interoperability
5. **Reliability**: 95%+ test coverage with comprehensive error handling

## Future Enhancements (Post-MVP)

- Dataset templates and schemas
- Data validation rules and constraints
- Real-time collaboration features
- Dataset sharing and permissions
- Advanced export formats (JSON, Parquet, etc.)
- Data transformation pipelines
- Integration with popular ML frameworks