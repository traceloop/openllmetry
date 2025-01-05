import pytest
from unittest.mock import Mock
from traceloop.sdk.annotation import Annotation
from traceloop.sdk.client.http import HTTPClient


@pytest.fixture
def mock_http():
    """Create a mock HTTP client"""
    http = Mock(spec=HTTPClient)
    http.post.return_value = {"status": "success"}
    return http


@pytest.fixture
def annotation(mock_http):
    """Create an Annotation instance with mock HTTP client"""
    return Annotation(mock_http, "test-app")


def test_annotation_initialization(mock_http):
    """Test annotation is properly initialized"""
    annotation = Annotation(mock_http, "test-app")

    assert annotation._http == mock_http
    assert annotation._app_name == "test-app"


def test_create_annotation_basic(annotation, mock_http):
    """Test creating an annotation with basic parameters"""
    annotation.create(
        annotation_task="task_123",
        entity_instance_id="instance_456",
        tags={"sentiment": "positive"},
    )

    mock_http.post.assert_called_once_with(
        "annotation-tasks/task_123/annotations",
        {
            "entity_instance_id": "instance_456",
            "tags": {"sentiment": "positive"},
            "source": "sdk",
            "flow": "user_feedback",
            "actor": {
                "type": "service",
                "id": "test-app",
            },
        },
    )


def test_create_annotation_complex_tags(annotation, mock_http):
    """Test creating an annotation with complex tags"""
    tags = {"sentiment": "positive", "relevance": 0.95, "tones": ["happy", "nice"]}

    annotation.create(annotation_task="task_123", entity_instance_id="instance_456", tags=tags)

    mock_http.post.assert_called_once_with(
        "annotation-tasks/task_123/annotations",
        {
            "entity_instance_id": "instance_456",
            "tags": tags,
            "source": "sdk",
            "flow": "user_feedback",
            "actor": {
                "type": "service",
                "id": "test-app",
            },
        },
    )
