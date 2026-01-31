# Azure AI Search Instrumentation - Span Attribute Extraction Guide

## Overview

This document explains how span attributes are extracted from Azure AI Search SDK method calls in the OpenTelemetry instrumentation, and provides a step-by-step methodology for adding support for new SDK methods and attributes.

## Table of Contents

1. [Current Implementation Architecture](#current-implementation-architecture)
2. [Span Attribute Extraction Methodology](#span-attribute-extraction-methodology)
3. [Step-by-Step Guide for Adding New Attributes](#step-by-step-guide-for-adding-new-attributes)
4. [Best Practices](#best-practices)
5. [Testing Strategy](#testing-strategy)

---

## Current Implementation Architecture

### File Structure

```
packages/opentelemetry-instrumentation-azure-search/
├── opentelemetry/instrumentation/azure_search/
│   ├── __init__.py          # Method definitions and instrumentor
│   ├── wrapper.py           # Span creation and attribute extraction
│   ├── utils.py             # Helper functions (@dont_throw decorator)
│   └── config.py            # Configuration
└── tests/
    ├── test_azure_search_instrumentation.py    # Unit tests (mocked)
    └── test_azure_search_integration.py        # Integration tests (VCR)

packages/opentelemetry-semantic-conventions-ai/
└── opentelemetry/semconv_ai/__init__.py  # Attribute constant definitions
```

### Key Components

#### 1. Semantic Conventions (`semconv_ai/__init__.py`)

Defines all span attribute constants:

```python
class SpanAttributes:
    # Azure Search specific attributes
    AZURE_SEARCH_INDEX_NAME = "azure_search.index_name"
    AZURE_SEARCH_SEARCH_TEXT = "azure_search.search.text"
    AZURE_SEARCH_SEARCH_TOP = "azure_search.search.top"
    # ... more attributes
```

**Naming Pattern:** `azure_search.<category>.<attribute>`

#### 2. Method Wrapper (`wrapper.py`)

Contains:

- `_wrap()`: Main wrapper that creates spans and routes to extraction functions
- `_set_<operation>_attributes()`: Specific extraction functions for each operation type
- `_set_span_attribute()`: Helper that only sets non-null, non-empty values

#### 3. Method Registry (`__init__.py`)

Defines which methods to instrument:

```python
SEARCH_CLIENT_METHODS = [
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "search",
        "span_name": "azure_search.search",
    },
    # ... more methods
]
```

---

## Span Attribute Extraction Methodology

### Core Techniques

#### 1. **Flexible Parameter Extraction**

Handle both positional and keyword arguments:

```python
# Example: search_text can be positional or keyword
search_text = kwargs.get("search_text") or (args[0] if args else None)
```

**How it works:**
- `kwargs.get("search_text")` - tries keyword arguments first
- `or (args[0] if args else None)` - falls back to positional argument
- `if args else None` - safe handling when no args provided

#### 2. **Instance Attribute Access**
Extract attributes stored on SDK client objects:

```python
# SearchClient stores index name in _index_name attribute
index_name = getattr(instance, "_index_name", None)
```

**When to use:**
- Index names, endpoints, or other client-level configuration
- Attributes that don't change per-call

#### 3. **Object Property Access**
Extract data from complex parameter objects:

```python
# For create_index(index=SearchIndex(...))
index = kwargs.get("index") or (args[0] if args else None)
if index:
    index_name = getattr(index, "name", None)
```

**When to use:**
- Parameters that are SDK-specific classes (not primitives)
- Need to extract nested properties

#### 4. **Type Detection and Handling**
Handle different parameter types gracefully:

```python
documents = kwargs.get("documents") or (args[0] if args else None)
if documents:
    if hasattr(documents, "__len__"):
        count = len(documents)
    else:
        # Handle generators or iterables
        try:
            docs_list = list(documents)
            count = len(docs_list)
        except (TypeError, ValueError):
            pass  # Can't determine count
```

#### 5. **Error Resilience**
Use `@dont_throw` decorator to ensure instrumentation never breaks the app:

```python
@dont_throw
def _set_search_attributes(span, args, kwargs):
    """Set attributes for search operations."""
    # Extraction logic here
```

### Current Extraction Functions

| Function | Purpose | Key Techniques |
|----------|---------|----------------|
| `_set_index_name_attribute()` | Extract index name from client instance | Instance attribute access |
| `_set_search_attributes()` | Extract search parameters | Positional/keyword handling |
| `_set_get_document_attributes()` | Extract document key | Positional/keyword handling |
| `_set_document_batch_attributes()` | Extract document count | Type detection, len() handling |
| `_set_index_documents_attributes()` | Extract batch size | Object property access |
| `_set_suggestion_attributes()` | Extract autocomplete/suggest params | Multi-arg positional handling |
| `_set_index_management_attributes()` | Extract index management params | Method-based routing |
| `_set_analyze_text_attributes()` | Extract analyzer info | Nested object property access |

---

## Step-by-Step Guide for Adding New Attributes

### Scenario: Azure SDK Adds New Method or Parameters

Let's walk through adding support for a new hypothetical method: `SearchClient.vector_search(query_vector, k=10, filter=None)`

### Step 1: Define Semantic Conventions

**File:** `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py`

```python
class SpanAttributes:
    # ... existing attributes ...

    # Add new Azure Search vector search attributes
    AZURE_SEARCH_QUERY_VECTOR_DIMENSIONS = "azure_search.query.vector_dimensions"
    AZURE_SEARCH_VECTOR_K = "azure_search.vector.k"
    AZURE_SEARCH_VECTOR_FILTER = "azure_search.vector.filter"
```

**Checklist:**
- [ ] Follow naming convention: `azure_search.<category>.<attribute>`
- [ ] Use descriptive, lowercase names with underscores
- [ ] Group related attributes together
- [ ] Add comments for clarity if needed

### Step 2: Register the Method

**File:** `packages/opentelemetry-instrumentation-azure-search/opentelemetry/instrumentation/azure_search/__init__.py`

Add to `SEARCH_CLIENT_METHODS` list:

```python
SEARCH_CLIENT_METHODS = [
    # ... existing methods ...
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "vector_search",
        "span_name": "azure_search.vector_search",
    },
]
```

**Checklist:**
- [ ] Correct module path
- [ ] Correct class name ("SearchClient" or "SearchIndexClient")
- [ ] Exact method name (case-sensitive)
- [ ] Descriptive span name following pattern

### Step 3: Create Extraction Function

**File:** `packages/opentelemetry-instrumentation-azure-search/opentelemetry/instrumentation/azure_search/wrapper.py`

```python
@dont_throw
def _set_vector_search_attributes(span, args, kwargs):
    """Set attributes for vector search operations."""
    # Extract query_vector (positional or keyword)
    query_vector = kwargs.get("query_vector") or (args[0] if args else None)
    if query_vector:
        # Extract dimensions if it's a list/array
        if hasattr(query_vector, "__len__"):
            dimensions = len(query_vector)
            _set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_QUERY_VECTOR_DIMENSIONS,
                dimensions
            )

    # Extract k parameter
    k = kwargs.get("k")
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_K, k)

    # Extract filter parameter
    filter_expr = kwargs.get("filter")
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_FILTER, filter_expr)

    # Also set as top_k for vector DB convention
    if k:
        _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, k)
```

**Checklist:**
- [ ] Use `@dont_throw` decorator
- [ ] Handle both positional and keyword arguments
- [ ] Use `_set_span_attribute()` helper (handles null/empty checks)
- [ ] Clear, descriptive docstring
- [ ] Handle edge cases (None, empty, generators, etc.)

### Step 4: Route to Extraction Function

**File:** `packages/opentelemetry-instrumentation-azure-search/opentelemetry/instrumentation/azure_search/wrapper.py`

In the `_wrap()` function, add routing logic:

```python
@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    # ... existing code ...

    with tracer.start_as_current_span(name, kind=SpanKind.CLIENT, attributes={...}) as span:
        _set_index_name_attribute(span, instance, args, kwargs)

        # Add new routing
        if method == "search":
            _set_search_attributes(span, args, kwargs)
        elif method == "vector_search":  # NEW
            _set_vector_search_attributes(span, args, kwargs)  # NEW
        elif method == "get_document":
            _set_get_document_attributes(span, args, kwargs)
        # ... rest of routing ...
```

**Checklist:**
- [ ] Add to appropriate location in if/elif chain
- [ ] Use exact method name from Step 2
- [ ] Call the extraction function created in Step 3

### Step 5: Add Unit Tests

**File:** `packages/opentelemetry-instrumentation-azure-search/tests/test_azure_search_instrumentation.py`

```python
class TestSearchClientInstrumentation:
    # ... existing tests ...

    def test_vector_search_creates_span(self, exporter):
        """Test that vector_search() creates a span with correct attributes."""
        from opentelemetry import trace

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.vector_search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
                SpanAttributes.AZURE_SEARCH_QUERY_VECTOR_DIMENSIONS: 5,
                SpanAttributes.AZURE_SEARCH_VECTOR_K: 10,
                SpanAttributes.AZURE_SEARCH_VECTOR_FILTER: "category eq 'electronics'",
                SpanAttributes.VECTOR_DB_QUERY_TOP_K: 10,
            }
        ):
            # Simulate method call
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "azure_search.vector_search"
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_QUERY_VECTOR_DIMENSIONS) == 5
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_VECTOR_K) == 10
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_VECTOR_FILTER) == "category eq 'electronics'"
        assert span.attributes.get(SpanAttributes.VECTOR_DB_QUERY_TOP_K) == 10
```

**Checklist:**
- [ ] Test all new attributes are captured
- [ ] Test with different parameter types (positional, keyword)
- [ ] Test edge cases (None, empty, missing parameters)
- [ ] Descriptive test name and docstring

### Step 6: Add Integration Tests

**File:** `packages/opentelemetry-instrumentation-azure-search/tests/test_azure_search_integration.py`

```python
class TestSearchClientIntegration:
    # ... existing tests ...

    @pytest.mark.vcr
    def test_vector_search(self, exporter, search_client):
        """Test that vector_search() creates a span with correct attributes."""
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = list(search_client.vector_search(
            query_vector=query_vector,
            k=10,
            filter="category eq 'electronics'",
        ))

        spans = exporter.get_finished_spans()
        vector_search_spans = [s for s in spans if s.name == "azure_search.vector_search"]
        assert len(vector_search_spans) == 1

        span = vector_search_spans[0]
        assert span.kind == SpanKind.CLIENT
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_QUERY_VECTOR_DIMENSIONS) == 5
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_VECTOR_K) == 10
        assert span.attributes.get(SpanAttributes.AZURE_SEARCH_VECTOR_FILTER) == "category eq 'electronics'"
```

**Checklist:**
- [ ] Use `@pytest.mark.vcr` decorator for VCR cassette recording
- [ ] Test with real SDK client (mocked via VCR)
- [ ] Filter for specific span name if nested spans exist
- [ ] Verify SpanKind.CLIENT
- [ ] Record cassettes with real Azure credentials

### Step 7: Update Mock Client (for unit tests)

**File:** `packages/opentelemetry-instrumentation-azure-search/tests/test_azure_search_instrumentation.py`

Update `MockSearchClient` class:

```python
class MockSearchClient:
    """Mock SearchClient for testing."""

    # ... existing methods ...

    def vector_search(self, query_vector=None, k=10, filter=None, **kwargs):
        """Mock vector search method."""
        return iter([{"id": "1", "score": 0.95, "name": "Test Document"}])
```

**Checklist:**
- [ ] Match SDK method signature
- [ ] Return appropriate mock data
- [ ] Include **kwargs for flexibility

---

## Best Practices

### 1. **Attribute Naming Conventions**

✅ **Good:**
```python
AZURE_SEARCH_SEARCH_TOP = "azure_search.search.top"
AZURE_SEARCH_VECTOR_K = "azure_search.vector.k"
```

❌ **Bad:**
```python
AZURE_SEARCH_TOP = "azure_search.top"  # Not specific enough
SEARCH_TOP = "search.top"  # Missing vendor prefix
```

**Rules:**
- Always start with `azure_search.`
- Use category grouping (e.g., `search.`, `vector.`, `document.`)
- Use lowercase with underscores
- Be descriptive but concise

### 2. **Parameter Extraction Safety**

Always handle missing/None values:

```python
# ✅ Good - handles all cases
param = kwargs.get("param") or (args[0] if args else None)
if param:
    _set_span_attribute(span, ATTR_NAME, param)

# ❌ Bad - can crash
param = args[0]  # IndexError if args is empty
_set_span_attribute(span, ATTR_NAME, param)  # None values set
```

### 3. **Type Handling**

Use duck typing to handle different types:

```python
# ✅ Good - flexible type handling
if hasattr(documents, "__len__"):
    count = len(documents)
else:
    try:
        count = len(list(documents))
    except (TypeError, ValueError):
        count = None

# ❌ Bad - assumes specific type
count = len(documents)  # Fails for generators
```

### 4. **Error Resilience**

Always use `@dont_throw` decorator:

```python
# ✅ Good - errors won't break instrumentation
@dont_throw
def _set_custom_attributes(span, args, kwargs):
    # Extraction logic

# ❌ Bad - unhandled exception breaks app
def _set_custom_attributes(span, args, kwargs):
    # Extraction logic without error handling
```

### 5. **Null Safety**

Use the `_set_span_attribute()` helper:

```python
# ✅ Good - only sets non-null, non-empty values
_set_span_attribute(span, ATTR_NAME, value)

# ❌ Bad - sets None and empty strings
span.set_attribute(ATTR_NAME, value)
```

Implementation:
```python
def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
```

---

## Testing Strategy

### Unit Tests (Mocked)
**Purpose:** Fast feedback, test extraction logic in isolation

**Approach:**
1. Create mock SDK objects
2. Manually create spans with expected attributes
3. Assert attributes are set correctly

**When to use:** Testing extraction logic, edge cases, error handling

### Integration Tests (VCR Cassettes)
**Purpose:** Test with real SDK behavior, verify attribute extraction from actual API calls

**Approach:**
1. Record interactions with real Azure Search service
2. Replay from cassettes in CI/CD
3. Verify spans match real SDK behavior

**When to use:** Testing against SDK changes, validating real-world scenarios

### VCR Cassette Recording

**First time (requires Azure credentials):**
```bash
AZURE_SEARCH_ENDPOINT="https://your-service.search.windows.net" \
AZURE_SEARCH_ADMIN_KEY="your-key" \
poetry run pytest tests/test_azure_search_integration.py --record-mode=all
```

**Subsequent runs (playback mode):**
```bash
poetry run pytest tests/test_azure_search_integration.py --record-mode=none
```

**Setup/Teardown Pattern:**
```python
@pytest.fixture(scope="class", autouse=True)
def setup_test_index(self, index_client_setup):
    """Set up the test index before all tests, tear down after."""
    # Skip in playback mode
    is_playback_mode = os.environ.get("AZURE_SEARCH_ADMIN_KEY") == "test-api-key"

    if not is_playback_mode:
        # Create fresh index
        index_client_setup.delete_index(INDEX_NAME)
        index_client_setup.create_index(index)

    yield

    if not is_playback_mode:
        # Cleanup
        index_client_setup.delete_index(INDEX_NAME)
```

---

## Common Patterns by SDK Method Type

### Pattern 1: Simple Query Methods
**Examples:** `search()`, `autocomplete()`, `suggest()`

**Characteristics:**
- Multiple optional parameters
- Parameters are primitives (strings, ints)
- First parameter often positional

**Template:**
```python
@dont_throw
def _set_<operation>_attributes(span, args, kwargs):
    # First param (positional or keyword)
    param1 = kwargs.get("param1") or (args[0] if args else None)
    _set_span_attribute(span, ATTR_PARAM1, param1)

    # Optional keyword params
    _set_span_attribute(span, ATTR_PARAM2, kwargs.get("param2"))
    _set_span_attribute(span, ATTR_PARAM3, kwargs.get("param3"))
```

### Pattern 2: Batch Document Methods
**Examples:** `upload_documents()`, `merge_documents()`, `delete_documents()`

**Characteristics:**
- First parameter is a collection
- Need to extract count
- May be list or generator

**Template:**
```python
@dont_throw
def _set_document_batch_attributes(span, args, kwargs):
    documents = kwargs.get("documents") or (args[0] if args else None)
    if documents:
        if hasattr(documents, "__len__"):
            count = len(documents)
        else:
            try:
                count = len(list(documents))
            except (TypeError, ValueError):
                return
        _set_span_attribute(span, ATTR_DOCUMENT_COUNT, count)
```

### Pattern 3: Index Management Methods
**Examples:** `create_index()`, `delete_index()`, `get_index()`

**Characteristics:**
- Parameter is SDK object or string
- Need to extract index name
- Multiple parameter formats

**Template:**
```python
@dont_throw
def _set_index_management_attributes(span, method, args, kwargs):
    if method in ["create_index", "create_or_update_index"]:
        # Parameter is SearchIndex object
        index = kwargs.get("index") or (args[0] if args else None)
        if index:
            index_name = getattr(index, "name", None)
            _set_span_attribute(span, ATTR_INDEX_NAME, index_name)
    elif method in ["delete_index", "get_index"]:
        # Parameter is string
        index_name = kwargs.get("index_name") or (args[0] if args else None)
        _set_span_attribute(span, ATTR_INDEX_NAME, index_name)
```

### Pattern 4: Complex Nested Parameters
**Examples:** `analyze_text()`

**Characteristics:**
- Parameters are SDK objects with nested properties
- Need to extract from nested structures
- May have multiple fallback paths

**Template:**
```python
@dont_throw
def _set_analyze_text_attributes(span, args, kwargs):
    # Extract from complex object
    analyze_request = kwargs.get("analyze_request") or (args[1] if len(args) > 1 else None)

    if analyze_request:
        # Try to get from object property
        analyzer_name = getattr(analyze_request, "analyzer_name", None)

    # Fallback to direct kwargs
    if not analyzer_name:
        analyzer_name = kwargs.get("analyzer_name")

    if analyzer_name:
        # Handle enum types
        if hasattr(analyzer_name, "value"):
            analyzer_name = analyzer_name.value
        _set_span_attribute(span, ATTR_ANALYZER_NAME, str(analyzer_name))
```

---

## Quick Reference Checklist

When adding support for a new SDK method:

- [ ] **Step 1:** Define semantic convention attributes in `semconv_ai/__init__.py`
- [ ] **Step 2:** Register method in `SEARCH_CLIENT_METHODS` or `SEARCH_INDEX_CLIENT_METHODS`
- [ ] **Step 3:** Create `_set_<operation>_attributes()` extraction function
- [ ] **Step 4:** Add routing logic in `_wrap()` function
- [ ] **Step 5:** Add unit test in `test_azure_search_instrumentation.py`
- [ ] **Step 6:** Add integration test in `test_azure_search_integration.py`
- [ ] **Step 7:** Update `MockSearchClient` with new method
- [ ] **Step 8:** Record VCR cassettes with real Azure credentials
- [ ] **Step 9:** Verify tests pass in playback mode
- [ ] **Step 10:** Run linting: `poetry run flake8`

---

## Troubleshooting

### Issue: Attribute not appearing in spans

**Check:**
1. Is the attribute defined in `semconv_ai/__init__.py`?
2. Is the method registered in `__init__.py`?
3. Is routing logic added to `_wrap()`?
4. Is `_set_span_attribute()` being called (not direct `span.set_attribute()`)?
5. Is the value actually non-None and non-empty?

**Debug:**
```python
# Add temporary logging
import logging
logger = logging.getLogger(__name__)

@dont_throw
def _set_<operation>_attributes(span, args, kwargs):
    param = kwargs.get("param")
    logger.info(f"Extracting param: {param}")  # Debug
    _set_span_attribute(span, ATTR_NAME, param)
```

### Issue: Tests failing with KeyError

**Cause:** Environment variables not set

**Solution:**
- Unit tests: Should work without real credentials (uses mocks)
- Integration tests with VCR: Need `AZURE_SEARCH_ENDPOINT` but not real key (uses cassettes)
- Integration tests recording: Need both real endpoint and key

### Issue: VCR cassette mismatch

**Cause:** Request doesn't match cassette

**Solutions:**
1. Re-record cassettes: `--record-mode=all`
2. Check VCR match configuration in `conftest.py`
3. Ensure environment variables match cassette recordings

---

## Additional Resources

- **OpenTelemetry Semantic Conventions:** https://opentelemetry.io/docs/specs/semconv/gen-ai/
- **Azure Search SDK Docs:** https://learn.microsoft.com/en-us/python/api/azure-search-documents/
- **VCR.py Documentation:** https://vcrpy.readthedocs.io/

---

## Document Metadata

- **Created:** 2024-12-15
- **Last Updated:** 2024-12-15
- **Maintainer:** OpenLLMetry Team
- **Related Packages:**
  - `opentelemetry-instrumentation-azure-search`
  - `opentelemetry-semantic-conventions-ai`
