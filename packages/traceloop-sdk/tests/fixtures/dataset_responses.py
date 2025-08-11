"""Centralized dataset API response fixtures"""
import pytest


def create_dataset_response(price_type="string", in_stock_type="string"):
    """Generate dataset response with configurable column types"""
    return {
        "id": "cmdvei5dd000g01vvyftz2zv1",
        "org_id": "c108269c-cf1e-4ac6-a7e4-5a456cc9fdb7",
        "project_id": "cm9v2g95l0011z613sv851kwd",
        "slug": "test-dataset",
        "name": "Dataset",
        "description": "Dataset Description",
        "columns": {
            "name": {"name": "Name", "type": "string"},
            "price": {"name": "Price", "type": price_type},
            "in-stock": {"name": "In Stock", "type": in_stock_type},
        },
        "last_version": None,
        "created_at": "2025-08-03T08:09:53.329521779Z",
        "updated_at": "2025-08-03T08:09:53.329522049Z",
    }


CREATE_ROWS_RESPONSE = {
    "rows": [
        {
            "id": "row_1_id",
            "row_index": 1,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Laptop",
                "cmdvei5dd000e01vvz0eb5kz8": 999.99,
                "cmdvei5dd000f01vv7aazk674": True,
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z",
        },
        {
            "id": "row_2_id",
            "row_index": 2,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Mouse",
                "cmdvei5dd000e01vvz0eb5kz8": 29.99,
                "cmdvei5dd000f01vv7aazk674": False,
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z",
        },
    ],
    "total": 2,
}

GET_DATASET_BY_SLUG_RESPONSE = {
    "id": "cmdvki9zv003c01vvj7is4p80",
    "slug": "product-inventory-2",
    "name": "Product Inventory",
    "description": "Sample product inventory data",
    "columns": {
        "product": {"name": "product", "type": "string"},
        "price": {"name": "price", "type": "number"},
        "in-stock": {"name": "in_stock", "type": "boolean"},
        "category": {"name": "category", "type": "string"},
    },
    "created_at": "2025-08-03T10:57:57.019Z",
    "updated_at": "2025-08-03T10:57:57.019Z",
    "rows": [
        {
            "id": "cmdvkieye003d01vv1zlmkjrg",
            "row_index": 1,
            "values": {
                "product": "Laptop",
                "price": 999.99,
                "in-stock": True,
                "category": "Electronics",
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z",
        },
        {
            "id": "cmdvkieye003e01vvs4onq0sq",
            "row_index": 2,
            "values": {
                "product": "Mouse",
                "price": 29.99,
                "in-stock": True,
                "category": "Accessories",
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z",
        },
        {
            "id": "cmdvkieye003f01vvkc6jxgad",
            "row_index": 3,
            "values": {
                "product": "Keyboard",
                "price": 79.99,
                "in-stock": False,
                "category": "Accessories",
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z",
        },
        {
            "id": "cmdvkieye003g01vvcuze8p1z",
            "row_index": 4,
            "values": {
                "product": "Monitor",
                "price": 299.99,
                "in-stock": True,
                "category": "Electronics",
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z",
        },
    ],
}

ADD_ROWS_RESPONSE = {
    "rows": [
        {
            "id": "row_add_1",
            "row_index": 0,
            "values": {"name": "Gal", "age": 8, "is-active": True},
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z",
        },
        {
            "id": "row_add_2",
            "row_index": 1,
            "values": {"name": "Nir", "age": 70, "is-active": False},
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z",
        },
        {
            "id": "row_add_3",
            "row_index": 2,
            "values": {"name": "Nina", "age": 52, "is-active": True},
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z",
        },
        {
            "id": "row_add_4",
            "row_index": 3,
            "values": {"name": "Aviv", "age": 52, "is-active": False},
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z",
        },
    ],
    "total": 4,
}

SINGLE_ROW_RESPONSE = {
    "rows": [
        {
            "id": "single_row_id",
            "row_index": 1,
            "values": {"name": "single"},
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z",
        }
    ],
    "total": 1,
}

GET_ALL_DATASETS_RESPONSE = {
    "datasets": [
        {
            "id": "cmdwnop4y0004meitkf17oxtn",
            "slug": "product-inventory-3",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-04T08:14:41.602Z",
            "updated_at": "2025-08-04T08:14:41.602Z",
        },
        {
            "id": "cmdvk9hil000e2cp0088rqrud",
            "slug": "product-inventory",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-03T13:51:06.861Z",
            "updated_at": "2025-08-03T13:51:06.861Z",
        },
        {
            "id": "cmdvki9zv003c01vvj7is4p80",
            "slug": "product-inventory-2",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-03T10:57:57.019Z",
            "updated_at": "2025-08-03T10:57:57.019Z",
        },
        {
            "id": "cmdvkg5eg003301vv1n7jm0m9",
            "slug": "product-inventory-1",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-03T10:56:17.753Z",
            "updated_at": "2025-08-03T10:56:17.753Z",
        },
        {
            "id": "cmdvg6jcq001p01vv5v4ob09v",
            "slug": "employee-data-1",
            "name": "Employee Dataset",
            "description": "Sample employee data for demonstration",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-03T08:56:50.81Z",
            "updated_at": "2025-08-03T08:56:50.81Z",
        },
        {
            "id": "cmdvfm9ms001f01vvbe30fbuj",
            "slug": "employee-data",
            "name": "Employee Dataset",
            "description": "Sample employee data for demonstration",
            "last_version": None,
            "columns": {},
            "created_at": "2025-08-03T08:41:05.093Z",
            "updated_at": "2025-08-03T08:41:05.093Z",
        },
    ],
    "total": 6,
}

BASIC_DATASET_RESPONSE = {
    "id": "cmdpy2ah40000q9p0ait7vukf",
    "slug": "daatset-5",
    "name": "Data",
    "description": "Data for example",
    "columns": {
        "col-id-1": {"slug": "col-id-1", "name": "col_number", "type": "number"},
        "col-id-2": {"slug": "col-id-2", "name": "col_bool", "type": "boolean"},
    },
    "created_at": "2025-07-30T15:30:48.712Z",
    "updated_at": "2025-08-04T09:26:41.255Z",
}

PUBLISH_DATASET_RESPONSE = {"dataset_id": "cmdpy2ah40000q9p0ait7vukf", "version": "v1"}

GET_DATASET_BY_VERSION_CSV = """product,price,in_stock,category,New Column 1
Laptop,999.99,false,Electronics,
,0,true,,
Mouse,29.99,false,Accessories,
,0,false,,
Monitor,299.99,true,Electronics,
,0,true,,
Keyboard,79.99,true,Accessories,
"""

ADD_COLUMN_RESPONSE = {
    "slug": "new-column-id",
    "name": "Test Column",
    "type": "string",
}


@pytest.fixture
def dataset_responses():
    """Fixture providing all dataset API response mocks"""
    return {
        "create_dataset": create_dataset_response,
        "create_rows": CREATE_ROWS_RESPONSE,
        "get_dataset_by_slug": GET_DATASET_BY_SLUG_RESPONSE,
        "add_rows": ADD_ROWS_RESPONSE,
        "single_row": SINGLE_ROW_RESPONSE,
        "get_all_datasets": GET_ALL_DATASETS_RESPONSE,
        "basic_dataset": BASIC_DATASET_RESPONSE,
        "publish_dataset": PUBLISH_DATASET_RESPONSE,
        "get_dataset_by_version_csv": GET_DATASET_BY_VERSION_CSV,
        "add_column": ADD_COLUMN_RESPONSE,
    }