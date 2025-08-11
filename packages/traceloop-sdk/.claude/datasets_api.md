# Datasets API Documentation

## Overview

The Datasets API provides a comprehensive set of endpoints for managing datasets, columns, rows, and cells. This RESTful API supports full CRUD operations for dataset management with flexible column types and real-time data manipulation capabilities.

### API Information
- **Title**: Datasets API
- **Version**: 1.0
- **Base URL**: `https://api.traceloop.com.`
- **Description**: Dataset management API for creating, managing, and manipulating structured data collections.

## Authentication

The API uses authentication middleware to secure all endpoints. All requests require proper authentication tokens passed via the Authorization header.

```
Authorization: Bearer YOUR_TOKEN
```

## API Endpoints

### Datasets

#### Get All Datasets
**GET** `/v2/datasets`

Retrieve all datasets for the authenticated user.

**Request:**
- **Method**: GET
- **Content-Type**: application/json

**Response:**
- **200 OK**: Returns an array of datasets
  ```json
  [
    {
      "id": "string",
      "name": "string",
      "orgID": "string",
      "env": "string",
      "createdAt": "string",
      "updatedAt": "string",
      "columnOrder": ["string"],
      "columns": [DatasetColumn],
      "rows": [DatasetRow]
    }
  ]
  ```
- **500 Internal Server Error**: Server error

#### Create Dataset
**POST** `/v2/datasets`

Create a new dataset for the authenticated user.

**Request:**
- **Method**: POST
- **Content-Type**: application/json
- **Parameters**:
  - `name` (formData, required): Dataset name

**Response:**
- **201 Created**: Returns the created dataset
- **400 Bad Request**: Invalid request data
- **500 Internal Server Error**: Server error

#### Get Specific Dataset
**GET** `/v2/datasets/{dataset_id}`

Retrieve a specific dataset by ID.

**Request:**
- **Method**: GET
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID

**Response:**
- **200 OK**: Returns the dataset object
- **404 Not Found**: Dataset not found
- **500 Internal Server Error**: Server error

#### Delete Dataset
**DELETE** `/v2/datasets/{dataset_id}`

Delete a specific dataset by ID.

**Request:**
- **Method**: DELETE
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID

**Response:**
- **204 No Content**: Dataset successfully deleted
- **500 Internal Server Error**: Server error

### Dataset Columns

#### Create Column
**POST** `/v2/datasets/{dataset_id}/columns`

Create a new column in a specific dataset.

**Request:**
- **Method**: POST
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
- **Request Body**:
  - `name` (required): Column name
  - `type` (required): Column type (custom-text, select, multi-select, boolean, metric, prompt, api)
  - `config` (optional): Column configuration object

**Response:**
- **201 Created**: Returns the created column
  ```json
  {
    "id": "string",
    "name": "string",
    "type": "string",
    "config": {},
    "datasetId": "string",
    "env": "string",
    "createdAt": "string",
    "updatedAt": "string"
  }
  ```
- **400 Bad Request**: Invalid request data
- **404 Not Found**: Dataset not found
- **500 Internal Server Error**: Server error

#### Update Column
**PUT** `/v2/datasets/{dataset_id}/columns/{column_id}`

Update a specific column in a dataset.

**Request:**
- **Method**: PUT
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
  - `column_id` (required): Column ID
- **Request Body**: UpdateDatasetColumnInput object
  ```json
  {
    "id": "string",
    "columnID": "string",
    "name": "string",
    "type": "ColumnType",
    "config": {},
    "orgID": "string",
    "env": "string"
  }
  ```

**Response:**
- **200 OK**: Column updated successfully
- **400 Bad Request**: Invalid request data
- **500 Internal Server Error**: Server error

#### Delete Column
**DELETE** `/v2/datasets/{dataset_id}/columns/{column_id}`

Delete a specific column from a dataset.

**Request:**
- **Method**: DELETE
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
  - `column_id` (required): Column ID

**Response:**
- **204 No Content**: Column successfully deleted
- **500 Internal Server Error**: Server error

### Dataset Rows

#### Create Row
**POST** `/v2/datasets/{dataset_id}/rows`

Create a new row in a specific dataset.

**Request:**
- **Method**: POST
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
- **Request Body**: CreateDatasetRowInput object
  ```json
  {
    "id": "string",
    "orgID": "string",
    "env": "string",
    "values": {}
  }
  ```

**Response:**
- **201 Created**: Returns the created row
  ```json
  {
    "id": "string",
    "datasetID": "string",
    "orgID": "string",
    "env": "string",
    "index": 0,
    "values": [0],
    "createdAt": "string",
    "updatedAt": "string",
    "dataset": "Dataset object"
  }
  ```
- **400 Bad Request**: Invalid request data
- **500 Internal Server Error**: Server error

#### Delete Row
**DELETE** `/v2/datasets/{dataset_id}/rows/{row_id}`

Delete a specific row from a dataset.

**Request:**
- **Method**: DELETE
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
  - `row_id` (required): Row ID

**Response:**
- **204 No Content**: Row successfully deleted
- **500 Internal Server Error**: Server error

### Dataset Cells

#### Update Cell Values
**PUT** `/v2/datasets/{dataset_id}/cells`

Update specific cell values in a dataset.

**Request:**
- **Method**: PUT
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
- **Request Body**: UpdateDatasetCellInput object
  ```json
  {
    "updates": [
      {
        "columnId": "string",
        "rowId": "string",
        "value": {}
      }
    ]
  }
  ```

**Response:**
- **200 OK**: Cells updated successfully
- **400 Bad Request**: Invalid request data
- **404 Not Found**: Dataset not found
- **500 Internal Server Error**: Server error

### Column Order Management

#### Update Column Order
**PUT** `/v2/datasets/{dataset_id}/column-order`

Update the order of columns in a specific dataset.

**Request:**
- **Method**: PUT
- **Path Parameters**:
  - `dataset_id` (required): Dataset ID
- **Request Body**: Array of column IDs in desired order
  ```json
  ["column_id_1", "column_id_2", "column_id_3"]
  ```

**Response:**
- **200 OK**: Column order updated successfully
- **400 Bad Request**: Invalid request data
- **500 Internal Server Error**: Server error

## Data Models

### Column Types
Supported column types for datasets:
- `custom-text`: Custom text input
- `select`: Single selection
- `multi-select`: Multiple selection
- `boolean`: True/false values
- `metric`: Numerical metrics
- `prompt`: AI prompt columns
- `api`: API integration columns

### Dataset
```json
{
  "id": "string",
  "name": "string",
  "orgID": "string",
  "env": "string",
  "createdAt": "string",
  "updatedAt": "string",
  "columnOrder": ["string"],
  "columns": [DatasetColumn],
  "rows": [DatasetRow]
}
```

### DatasetColumn
```json
{
  "id": "string",
  "name": "string",
  "type": "ColumnType",
  "config": [0],
  "datasetID": "string",
  "orgID": "string",
  "env": "string",
  "createdAt": "string",
  "updatedAt": "string",
  "dataset": "Dataset object"
}
```

### DatasetRow
```json
{
  "id": "string",
  "datasetID": "string",
  "orgID": "string",
  "env": "string",
  "index": 0,
  "values": [0],
  "createdAt": "string",
  "updatedAt": "string",
  "dataset": "Dataset object"
}
```

### Request Models

#### CreateDatasetRowInput
```json
{
  "id": "string",
  "orgID": "string",
  "env": "string",
  "values": {}
}
```

#### UpdateDatasetColumnInput
```json
{
  "id": "string",
  "columnID": "string",
  "name": "string",
  "type": "ColumnType",
  "config": {},
  "orgID": "string",
  "env": "string"
}
```

#### UpdateDatasetCellInput
```json
{
  "updates": [
    {
      "columnId": "string",
      "rowId": "string",
      "value": {}
    }
  ]
}
```

#### CellUpdate
```json
{
  "columnId": "string",
  "rowId": "string",
  "value": {}
}
```

### Response Models

#### DatasetColumnResponse
```json
{
  "id": "string",
  "name": "string",
  "type": "ColumnType",
  "config": {},
  "datasetId": "string",
  "env": "string",
  "createdAt": "string",
  "updatedAt": "string"
}
```

## Error Responses

The API returns standard HTTP status codes:

- **200 OK**: Successful GET, PUT requests
- **201 Created**: Successful POST requests
- **204 No Content**: Successful DELETE requests
- **400 Bad Request**: Invalid request data or parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server-side errors

Error responses typically include a JSON object with error details.

## Usage Examples

### Creating a Dataset
```bash
curl -X POST "https://api.traceloop.com.v2/datasets" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "name=My New Dataset"
```

### Adding a Column to a Dataset
```bash
curl -X POST "https://api.traceloop.com.v2/datasets/dataset123/columns" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Status",
    "type": "select",
    "config": {
      "options": ["active", "inactive", "pending"]
    }
  }'
```

### Creating a Row in a Dataset
```bash
curl -X POST "https://api.traceloop.com.v2/datasets/dataset123/rows" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "values": {
      "column1": "value1",
      "column2": "value2"
    }
  }'
```

### Updating Cell Values
```bash
curl -X PUT "https://api.traceloop.com.v2/datasets/dataset123/cells" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "updates": [
      {
        "columnId": "col456",
        "rowId": "row789",
        "value": "updated value"
      }
    ]
  }'
```

### Updating Column Order
```bash
curl -X PUT "https://api.traceloop.com.v2/datasets/dataset123/column-order" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '["col1", "col3", "col2"]'
```

### Getting All Datasets
```bash
curl -X GET "https://api.traceloop.com.v2/datasets" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Getting a Specific Dataset
```bash
curl -X GET "https://api.traceloop.com.v2/datasets/dataset123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Deleting a Dataset
```bash
curl -X DELETE "https://api.traceloop.com.v2/datasets/dataset123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Deleting a Column
```bash
curl -X DELETE "https://api.traceloop.com.v2/datasets/dataset123/columns/col456" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Deleting a Row
```bash
curl -X DELETE "https://api.traceloop.com.v2/datasets/dataset123/rows/row789" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Notes

- All endpoints require proper authentication via Bearer token
- Dataset operations are scoped to the authenticated user's organization
- Column types determine the validation and behavior of data stored in those columns
- The `values` field in rows contains key-value pairs where keys are column IDs
- Column order can be customized using the column order management endpoint
- All timestamps are returned in ISO 8601 format
- Dataset, column, and row IDs are typically generated using CUID format