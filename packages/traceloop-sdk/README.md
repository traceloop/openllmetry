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

## Working with File Cells in Datasets

Datasets now support file columns that can store images, videos, audio files, and documents. You can either upload files to internal storage (S3) or link to external URLs.

### File Types Supported

- `IMAGE` - Images (PNG, JPEG, GIF, etc.)
- `VIDEO` - Video files (MP4, AVI, etc.) or video URLs (YouTube, Vimeo)
- `AUDIO` - Audio files (MP3, WAV, etc.) or audio URLs (Spotify, SoundCloud)
- `FILE` - General files (PDF, TXT, DOC, etc.)

### Upload a file to internal storage (S3):

```python
from traceloop.sdk import Traceloop
from traceloop.sdk.dataset.model import FileCellType

Traceloop.init(api_key="your-api-key")

# Get dataset and row
datasets = Traceloop.get_datasets()
dataset = datasets.get_by_slug("my-dataset")
row = dataset.rows[0]

# Upload an image file
row.set_file_cell(
    column_slug="screenshot",
    file_path="/path/to/image.png",
    file_type=FileCellType.IMAGE,
    content_type="image/png",
    metadata={"description": "Product screenshot"}
)

# Upload with a thumbnail
row.set_file_cell(
    column_slug="product_image",
    file_path="/path/to/large_image.jpg",
    file_type=FileCellType.IMAGE,
    content_type="image/jpeg",
    with_thumbnail=True,
    thumbnail_path="/path/to/thumbnail.jpg"
)

# Upload an audio file
row.set_file_cell(
    column_slug="podcast",
    file_path="/path/to/episode.mp3",
    file_type=FileCellType.AUDIO,
    content_type="audio/mpeg",
    metadata={"episode": "001", "duration": "45:00"}
)

# Upload a PDF document
row.set_file_cell(
    column_slug="manual",
    file_path="/path/to/manual.pdf",
    file_type=FileCellType.FILE,
    content_type="application/pdf"
)
```

### Link to external URLs:

```python
# Link to a YouTube video
row.set_file_cell(
    column_slug="demo_video",
    url="https://www.youtube.com/watch?v=abc123",
    file_type=FileCellType.VIDEO,
    metadata={"title": "Product Demo", "duration": "5:30"}
)

# Link to a Spotify track
row.set_file_cell(
    column_slug="theme_song",
    url="https://open.spotify.com/track/example",
    file_type=FileCellType.AUDIO,
    metadata={"artist": "Artist Name", "album": "Album Name"}
)

# Link to a Google Docs document
row.set_file_cell(
    column_slug="specifications",
    url="https://docs.google.com/document/d/example",
    file_type=FileCellType.FILE,
    metadata={"source": "Google Docs", "last_updated": "2024-01-15"}
)
```

### Error Handling

The `set_file_cell` method includes validation to ensure proper usage:

```python
# This will raise ValueError - can't provide both file_path and url
row.set_file_cell(
    column_slug="file",
    file_path="/path/to/file.txt",
    url="https://example.com/file.txt"  # Error!
)

# This will raise ValueError - must provide either file_path or url
row.set_file_cell(column_slug="file")  # Error!

# This will raise FileNotFoundError if the file doesn't exist
row.set_file_cell(
    column_slug="file",
    file_path="/nonexistent/file.txt"  # Error!
)
```
