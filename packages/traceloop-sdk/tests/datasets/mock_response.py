def create_dataset_response(price_type="string", in_stock_type="string"):
    """Generate dataset response with configurable column types"""
    return {
        "id": "cmdvei5dd000g01vvyftz2zv1",
        "orgId": "c108269c-cf1e-4ac6-a7e4-5a456cc9fdb7",
        "projectId": "cm9v2g95l0011z613sv851kwd",
        "slug": "daatset-12",
        "name": "Dataset",
        "description": "Dataset Description",
        "columns": {
            "cmdvei5dd000d01vv2yvmp7vt": {
                "name": "Name",
                "type": "string"
            },
            "cmdvei5dd000e01vvz0eb5kz8": {
                "name": "Price",
                "type": price_type
            },
            "cmdvei5dd000f01vv7aazk674": {
                "name": "In Stock",
                "type": in_stock_type
            }
        },
        "lastVersion": None,
        "createdAt": "2025-08-03T08:09:53.329521779Z",
        "updatedAt": "2025-08-03T08:09:53.329522049Z"
    }

create_rows_response_json = """
{
    "rows": [
        {
            "id": "row_1_id",
            "rowIndex": 1,
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
            "rowIndex": 2,
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
            "rowIndex": 1,
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
            "rowIndex": 2,
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
            "rowIndex": 3,
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
            "rowIndex": 4,
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
            "rowIndex": 0,
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
            "rowIndex": 1,
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
            "rowIndex": 2,
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
            "rowIndex": 3,
            "values": {
                "cmdr3ce1s0003hmp0vqons5ey": "Aviv",
                "cmdr3ce1s0004hmp0ies575jr": 52,
                "cmdr3ce1s0005hmp0bdln01js": true
            },
            "created_at": "2025-08-03T12:00:00.000Z",
            "updated_at": "2025-08-03T12:00:00.000Z"
        }
    ],
    "total": 4
}
"""
