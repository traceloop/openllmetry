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
            "cmdvei5dd000d01vv2yvmp7vt": {"name": "Name", "type": "string"},
            "cmdvei5dd000e01vvz0eb5kz8": {"name": "Price", "type": price_type},
            "cmdvei5dd000f01vv7aazk674": {"name": "In Stock", "type": in_stock_type},
        },
        "last_version": None,
        "created_at": "2025-08-03T08:09:53.329521779Z",
        "updated_at": "2025-08-03T08:09:53.329522049Z",
    }


create_rows_response_json = """
{
    "rows": [
        {
            "id": "row_1_id",
            "row_index": 1,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Laptop",
                "cmdvei5dd000e01vvz0eb5kz8": 999.99,
                "cmdvei5dd000f01vv7aazk674": true
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z"
        },
        {
            "id": "row_2_id",
            "row_index": 2,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Mouse",
                "cmdvei5dd000e01vvz0eb5kz8": 29.99,
                "cmdvei5dd000f01vv7aazk674": false
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z"
        }
    ],
    "total": 2
}
"""

get_dataset_by_slug_json = """
{
    "id": "cmdvki9zv003c01vvj7is4p80",
    "slug": "product-inventory-2",
    "name": "Product Inventory",
    "description": "Sample product inventory data",
    "columns": {
        "cmdvki9zv003801vv1idaywus": {
            "name": "product",
            "type": "string"
        },
        "cmdvki9zv003901vv5zr5i24b": {
            "name": "price",
            "type": "number"
        },
        "cmdvki9zv003a01vvvqqlytpr": {
            "name": "in_stock",
            "type": "boolean"
        },
        "cmdvki9zv003b01vvmk3d22km": {
            "name": "category",
            "type": "string"
        }
    },
    "created_at": "2025-08-03T10:57:57.019Z",
    "updated_at": "2025-08-03T10:57:57.019Z",
    "rows": [
        {
            "id": "cmdvkieye003d01vv1zlmkjrg",
            "row_index": 1,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Laptop",
                "cmdvki9zv003901vv5zr5i24b": 999.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Electronics"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003e01vvs4onq0sq",
            "row_index": 2,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Mouse",
                "cmdvki9zv003901vv5zr5i24b": 29.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Accessories"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003f01vvkc6jxgad",
            "row_index": 3,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Keyboard",
                "cmdvki9zv003901vv5zr5i24b": 79.99,
                "cmdvki9zv003a01vvvqqlytpr": false,
                "cmdvki9zv003b01vvmk3d22km": "Accessories"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003g01vvcuze8p1z",
            "row_index": 4,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Monitor",
                "cmdvki9zv003901vv5zr5i24b": 299.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Electronics"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        }
    ]
}
"""

add_rows_response_json = """
{
    "rows": [
        {
            "id": "row_add_1",
            "row_index": 0,
            "values": {
                "cmdr3ce1s0003hmp0vqons5ey": "Gal",
                "cmdr3ce1s0004hmp0ies575jr": 8,
                "cmdr3ce1s0005hmp0bdln01js": true
            },
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z"
        },
        {
            "id": "row_add_2",
            "row_index": 1,
            "values": {
                "cmdr3ce1s0003hmp0vqons5ey": "Nir",
                "cmdr3ce1s0004hmp0ies575jr": 70,
                "cmdr3ce1s0005hmp0bdln01js": false
            },
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z"
        },
        {
            "id": "row_add_3",
            "row_index": 2,
            "values": {
                "cmdr3ce1s0003hmp0vqons5ey": "Nina",
                "cmdr3ce1s0004hmp0ies575jr": 52,
                "cmdr3ce1s0005hmp0bdln01js": true
            },
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z"
        },
        {
            "id": "row_add_4",
            "row_index": 3,
            "values": {
                "cmdr3ce1s0003hmp0vqons5ey": "Aviv",
                "cmdr3ce1s0004hmp0ies575jr": 52,
                "cmdr3ce1s0005hmp0bdln01js": false
            },
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z"
        }
    ],
    "total": 4
}
"""

single_row_response_json = """
{
    "rows": [{
        "id": "single_row_id",
        "row_index": 1,
        "values": {"cmdr3ce1s0003hmp0vqons5ey": "single"},
        "created_at": "2025-08-03T12:00:00.000Z",
        "updated_at": "2025-08-03T12:00:00.000Z"
    }],
    "total": 1
}
"""

get_all_datasets_json = """
{
    "datasets": [
        {
            "id": "cmdwnop4y0004meitkf17oxtn",
            "slug": "product-inventory-3",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-04T08:14:41.602Z",
            "updated_at": "2025-08-04T08:14:41.602Z"
        },
        {
            "id": "cmdvk9hil000e2cp0088rqrud",
            "slug": "product-inventory",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-03T13:51:06.861Z",
            "updated_at": "2025-08-03T13:51:06.861Z"
        },
        {
            "id": "cmdvki9zv003c01vvj7is4p80",
            "slug": "product-inventory-2",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-03T10:57:57.019Z",
            "updated_at": "2025-08-03T10:57:57.019Z"
        },
        {
            "id": "cmdvkg5eg003301vv1n7jm0m9",
            "slug": "product-inventory-1",
            "name": "Product Inventory",
            "description": "Sample product inventory data",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-03T10:56:17.753Z",
            "updated_at": "2025-08-03T10:56:17.753Z"
        },
        {
            "id": "cmdvg6jcq001p01vv5v4ob09v",
            "slug": "employee-data-1",
            "name": "Employee Dataset",
            "description": "Sample employee data for demonstration",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-03T08:56:50.81Z",
            "updated_at": "2025-08-03T08:56:50.81Z"
        },
        {
            "id": "cmdvfm9ms001f01vvbe30fbuj",
            "slug": "employee-data",
            "name": "Employee Dataset",
            "description": "Sample employee data for demonstration",
            "last_version": null,
            "columns": {},
            "created_at": "2025-08-03T08:41:05.093Z",
            "updated_at": "2025-08-03T08:41:05.093Z"
        }
    ],
    "total": 6
}
"""

basic_dataset_response_json = """
{
    "id": "cmdpy2ah40000q9p0ait7vukf",
    "slug": "daatset-5",
    "name": "Data",
    "description": "Data for example",
    "columns": {
        "column_id_1": {
            "name": "col_number",
            "type": "number"
        },
        "column_id_2": {
            "name": "col_bool",
            "type": "boolean"
        }
    },
    "created_at": "2025-07-30T15:30:48.712Z",
    "updated_at": "2025-08-04T09:26:41.255Z"
}
"""

publish_dataset_response_json = """
{
    "dataset_id": "cmdpy2ah40000q9p0ait7vukf",
    "version": "v1"
}
"""

get_dataset_by_version_json = """
product,price,in_stock,category,New Column 1
Laptop,999.99,false,Electronics,
,0,true,,
Mouse,29.99,false,Accessories,
,0,false,,
Monitor,299.99,true,Electronics,
,0,true,,
Keyboard,79.99,true,Accessories,
"""

add_column_response_json = {
    "id": "new_column_id",
    "name": "Test Column",
    "type": "string",
}
