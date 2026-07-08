"""
Test constants to reduce string repetition across dataset tests.
"""


class TestConstants:
    # API Configuration
    API_KEY = "test-api-key"

    # Dataset IDs and Identifiers
    DATASET_ID = "cmdvei5dd000g01vvyftz2zv1"
    DATASET_SLUG = "test-dataset"
    DATASET_NAME = "Dataset"
    DATASET_DESCRIPTION = "Dataset Description"

    # Test Dataset Slugs
    TEST_CSV_DATASET_SLUG = "test-csv-dataset"
    TEST_DF_DATASET_SLUG = "test-df-dataset"
    PRODUCT_INVENTORY_SLUG = "product-inventory-2"

    # Test Dataset Names
    TEST_CSV_DATASET_NAME = "Test CSV Dataset"
    TEST_DF_DATASET_NAME = "Test DataFrame Dataset"
    PRODUCT_INVENTORY_NAME = "Product Inventory"

    # Test Dataset Descriptions
    TEST_CSV_DATASET_DESC = "Dataset created from CSV"
    TEST_DF_DATASET_DESC = "Dataset created from DataFrame"
    PRODUCT_INVENTORY_DESC = "Sample product inventory data"

    # Column IDs
    NAME_COLUMN_ID = "cmdvei5dd000d01vv2yvmp7vt"
    PRICE_COLUMN_ID = "cmdvei5dd000e01vv8h7k3q2s"
    STOCK_COLUMN_ID = "cmdvei5dd000f01vvmn9x1p4w"

    # Column Names
    NAME_COLUMN = "Name"
    PRICE_COLUMN = "Price"
    STOCK_COLUMN = "In Stock"
    PRODUCT_COLUMN = "product"

    # Column Types
    STRING_TYPE = "string"
    NUMBER_TYPE = "number"
    BOOLEAN_TYPE = "boolean"

    # Test Data Values
    LAPTOP_VALUE = "Laptop"
    LAPTOP_PRICE = 999.99
    MOUSE_VALUE = "Mouse"
    MOUSE_PRICE = 29.99

    # CSV Test Content
    SAMPLE_CSV_CONTENT = """Name,Price,In Stock
Laptop,999.99,true
Mouse,29.99,false"""

    # Mock Dataset IDs for different tests
    MOCK_DATASET_ID = "mock-dataset-id"
    TEST_DATASET_ID = "test_dataset_id"

    # API Endpoints
    DEFAULT_PROJECT_PATH = "datasets"

    # Version Information
    VERSION_V1 = "v1"

    # Test Row Data Column IDs (from mock_objects.py)
    TEST_NAME_COL_ID = "cmdr3ce1s0003hmp0vqons5ey"
    TEST_VALUE_COL_ID = "cmdr3ce1s0004hmp0ies575jr"
    TEST_ACTIVE_COL_ID = "cmdr3ce1s0005hmp0bdln01js"

    # File Paths
    NON_EXISTENT_FILE_PATH = "/non_existent/file.csv"
