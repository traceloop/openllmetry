import csv
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
from pydantic import Field, PrivateAttr

from .base import DatasetBaseModel, ColumnType
from .column import Column
from .row import Row

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None
    from .client import DatasetClient


class Dataset(DatasetBaseModel):
    id: Optional[str] = None
    name: str
    slug: str
    description: Optional[str] = None
    columns: List[Column] = Field(default_factory=list)
    rows: List[Row] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _client: Optional["DatasetClient"] = PrivateAttr(default=None)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "Dataset":
        """Class method to create dataset from CSV file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        if name is None:
            name = path.stem
        
        # Read CSV and infer column types
        columns = []
        rows_data = []
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            # Create columns based on CSV headers
            for i, field_name in enumerate(reader.fieldnames):
                columns.append(Column(
                    id=f"col_{i}",
                    name=field_name,
                    type=ColumnType.STRING,  # Default to STRING, could be enhanced with type inference
                    dataset_id="temp"  # Will be set after dataset creation
                ))
            
            # Read all rows
            for row_idx, row_data in enumerate(reader):
                rows_data.append({
                    "id": f"row_{row_idx}",
                    "index": row_idx,
                    "values": dict(row_data),
                    "dataset_id": "temp"  # Will be set after dataset creation
                })
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns=columns,
        )
        
        # Create dataset via API and get ID
        from .client import DatasetClient
        client = DatasetClient()
        api_dataset = client.create_dataset(name, description)
        dataset.id = api_dataset["id"]
        dataset.created_at = datetime.fromisoformat(api_dataset["created_at"])
        dataset.updated_at = datetime.fromisoformat(api_dataset["updated_at"])
        
        # Update column and row dataset_ids
        for col in dataset.columns:
            col.dataset_id = dataset.id
            col._client = client
        
        # Create columns via API
        for col in dataset.columns:
            api_col = client.add_column(dataset.id, col.name, col.type, col.config)
            col.id = api_col["id"]
        
        # Create rows
        for row_data in rows_data:
            row_data["dataset_id"] = dataset.id
            row = Row(**row_data)
            row._client = client
            dataset.rows.append(row)
            
            # Add row via API
            client.add_row(dataset.id, row.values)
        
        dataset._client = client
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "Dataset":
        """Class method to create dataset from pandas DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for from_dataframe(). Install with: pip install pandas")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")
        
        if name is None:
            name = f"Dataset from DataFrame"
        
        # Create columns from DataFrame
        columns = []
        for i, col_name in enumerate(df.columns):
            # Infer column type from pandas dtype
            dtype = df[col_name].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = ColumnType.NUMBER
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = ColumnType.BOOLEAN
            else:
                col_type = ColumnType.STRING
            
            columns.append(Column(
                id=f"col_{i}",
                name=str(col_name),
                type=col_type,
                dataset_id="temp"  # Will be set after dataset creation
            ))
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns=columns,
        )
        
        # Create dataset via API and get ID
        from .client import DatasetClient
        client = DatasetClient()
        api_dataset = client.create_dataset(name, description)
        dataset.id = api_dataset["id"]
        dataset.created_at = datetime.fromisoformat(api_dataset["created_at"])
        dataset.updated_at = datetime.fromisoformat(api_dataset["updated_at"])
        
        # Update column dataset_ids and create via API
        for col in dataset.columns:
            col.dataset_id = dataset.id
            col._client = client
            api_col = client.add_column(dataset.id, col.name, col.type, col.config)
            col.id = api_col["id"]
        
        # Create rows from DataFrame
        for row_idx, (_, pandas_row) in enumerate(df.iterrows()):
            # Convert pandas row to dict, handling NaN values
            row_values = {}
            for col_name in df.columns:
                value = pandas_row[col_name]
                if pd.isna(value):
                    row_values[col_name] = None
                else:
                    row_values[col_name] = value
            
            row = Row(
                id=f"row_{row_idx}",
                index=row_idx,
                values=row_values,
                dataset_id=dataset.id,
                _client=client
            )
            dataset.rows.append(row)
            
            # Add row via API
            client.add_row(dataset.id, row.values)
        
        dataset._client = client
        return dataset

    def add_column(self, name: str, col_type: ColumnType, config: Optional[Dict[str, Any]] = None) -> Column:
        """Add new column (returns Column object)"""
        if self._client is None:
            from .client import DatasetClient
            self._client = DatasetClient()
        
        api_col = self._client.add_column(self.id, name, col_type, config)
        
        column = Column(
            id=api_col["id"],
            name=name,
            type=col_type,
            config=config,
            dataset_id=self.id,
            _client=self._client
        )
        self.columns.append(column)
        return column