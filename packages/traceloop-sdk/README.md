# traceloop-sdk

Traceloop’s Python SDK allows you to easily start monitoring and debugging your LLM execution. Tracing is done in a non-intrusive way, built on top of OpenTelemetry. You can choose to export the traces to Traceloop, or to your existing observability stack.

```python
Traceloop.init(app_name="joke_generation_service")

@workflow(name="joke_creation")
def create_joke():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content
```

## Working with Attachments in Datasets

Datasets now support file attachments through a declarative API inspired by Braintrust. You can upload files to internal storage (S3) or link to external URLs using the new Attachment classes.

### File Types Supported

- `FileCellType.IMAGE` - Images (PNG, JPEG, GIF, etc.)
- `FileCellType.VIDEO` - Video files (MP4, AVI, etc.) or video URLs (YouTube, Vimeo)
- `FileCellType.AUDIO` - Audio files (MP3, WAV, etc.) or audio URLs (Spotify, SoundCloud)
- `FileCellType.FILE` - General files (PDF, TXT, DOC, etc.)

### Creating Datasets with Initial Attachments

You can now create datasets with attachments directly in the row values:

```python
from traceloop.sdk import Traceloop
from traceloop.sdk.dataset import Attachment, ExternalAttachment, FileCellType
from traceloop.sdk.dataset.model import CreateDatasetRequest, ColumnDefinition, ColumnType

Traceloop.init(api_key="your-api-key")
datasets = Traceloop.get_datasets()

# Create dataset request with attachments in row values
dataset_request = CreateDatasetRequest(
    slug="product-catalog",
    name="Product Catalog with Media",
    description="Products with images and videos",
    columns=[
        ColumnDefinition(slug="name", name="Product Name", type=ColumnType.STRING),
        ColumnDefinition(slug="price", name="Price", type=ColumnType.NUMBER),
        ColumnDefinition(slug="image", name="Product Image", type=ColumnType.FILE),
        ColumnDefinition(slug="video", name="Demo Video", type=ColumnType.FILE),
        ColumnDefinition(slug="manual", name="User Manual", type=ColumnType.FILE),
    ],
    rows=[
        {
            "name": "Smart Watch Pro",
            "price": 299.99,
            "image": Attachment(
                file_path="/path/to/watch.jpg",
                file_type=FileCellType.IMAGE,
                metadata={"alt_text": "Smart Watch Pro"}
            ),
            "video": ExternalAttachment(
                url="https://www.youtube.com/watch?v=demo123",
                file_type=FileCellType.VIDEO,
                metadata={"duration": "2:30"}
            ),
            "manual": Attachment(
                file_path="/path/to/manual.pdf",
                file_type=FileCellType.FILE
            ),
        },
        {
            "name": "Wireless Earbuds",
            "price": 149.99,
            "image": Attachment(
                data=image_bytes,  # From memory
                filename="earbuds.png",
                content_type="image/png",
                file_type=FileCellType.IMAGE
            ),
            "video": ExternalAttachment(
                url="https://vimeo.com/demo456",
                file_type=FileCellType.VIDEO
            ),
            "manual": None,  # No manual for this product
        },
    ]
)

# Create the dataset - attachments are automatically processed
dataset = datasets.create(dataset_request)
print(f"Created dataset: {dataset.slug}")

# The dataset is created with all attachments processed
for row in dataset.rows:
    print(f"Product: {row.values['name']}")
    if row.values.get('image'):
        print(f"  Image: {row.values['image']['status']}")  # 'success' or 'failed'
    if row.values.get('video'):
        print(f"  Video URL: {row.values['video']['url']}")
```

### Adding Attachments to Existing Datasets

For existing datasets, you can upload attachments to specific cells:

```python
from traceloop.sdk import Traceloop
from traceloop.sdk.dataset import Attachment, ExternalAttachment, FileCellType

Traceloop.init(api_key="your-api-key")
datasets = Traceloop.get_datasets()
dataset = datasets.get_by_slug("my-dataset")
row = dataset.rows[0]

# Upload a file attachment
attachment = Attachment(
    file_path="/path/to/document.pdf",
    file_type=FileCellType.FILE,
    metadata={"version": "1.0", "pages": 10}
)
ref = attachment.upload(datasets._http, dataset.slug, row.id, "document")

# Link an external URL
external = ExternalAttachment(
    url="https://docs.google.com/document/d/abc123",
    file_type=FileCellType.FILE,
    metadata={"source": "Google Docs"}
)
ref = external.attach(datasets._http, dataset.slug, row.id, "specifications")
```

### Working with In-Memory Data

You can create attachments from bytes data without saving to disk:

```python
# Generate or fetch data
image_data = generate_chart()  # Returns bytes
pdf_data = generate_report()   # Returns bytes

# Create attachments from memory
image_attachment = Attachment(
    data=image_data,
    filename="chart.png",
    content_type="image/png",
    file_type=FileCellType.IMAGE,
    metadata={"chart_type": "bar", "date": "2024-01-15"}
)

pdf_attachment = Attachment(
    data=pdf_data,
    filename="report.pdf",
    content_type="application/pdf",
    file_type=FileCellType.FILE,
    metadata={"report_type": "quarterly", "q": "Q4-2023"}
)

# Use in dataset creation
dataset_request = CreateDatasetRequest(
    slug="reports",
    name="Generated Reports",
    columns=[
        ColumnDefinition(slug="title", name="Title", type=ColumnType.STRING),
        ColumnDefinition(slug="chart", name="Chart", type=ColumnType.FILE),
        ColumnDefinition(slug="report", name="Report", type=ColumnType.FILE),
    ],
    rows=[{
        "title": "Q4 2023 Results",
        "chart": image_attachment,
        "report": pdf_attachment,
    }]
)

dataset = datasets.create(dataset_request)
```

### Attachment Validation

The Attachment class includes validation to ensure proper usage:

```python
# This will raise ValueError - can't provide both file_path and data
attachment = Attachment(
    file_path="/path/to/file.txt",
    data=b"test data"  # Error!
)

# This will raise ValueError - must provide either file_path or data
attachment = Attachment()  # Error!

# This will raise FileNotFoundError when uploading if file doesn't exist
attachment = Attachment(
    file_path="/nonexistent/file.txt"  # Error during upload!
)
```
