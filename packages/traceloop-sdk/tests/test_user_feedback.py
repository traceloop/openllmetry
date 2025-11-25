"""
Tests for the UserFeedback class.

These tests verify:
1. Proper initialization of UserFeedback instances
2. Basic feedback submission with minimal parameters
3. Handling of complex tag structures
4. Proper API endpoint construction and payload formatting
"""

import pytest
from unittest.mock import Mock
from traceloop.sdk.annotation.user_feedback import UserFeedback
from traceloop.sdk.client.http import HTTPClient


@pytest.fixture
def mock_http():
    """Create a mock HTTP client"""
    http = Mock(spec=HTTPClient)
    http.post.return_value = {"status": "success"}
    return http


@pytest.fixture
def user_feedback(mock_http):
    """Create a UserFeedback instance with mock HTTP client"""
    return UserFeedback(mock_http, "test-app")


def test_user_feedback_initialization(mock_http):
    """Test UserFeedback is properly initialized"""
    feedback = UserFeedback(mock_http, "test-app")
    assert feedback._http == mock_http
    assert feedback._app_name == "test-app"


def test_create_basic_feedback(user_feedback: UserFeedback, mock_http: Mock):
    """Test creating basic user feedback"""
    user_feedback.create(
        annotation_task="task_123", entity_id="instance_456", tags={"sentiment": "positive"}
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


def test_create_feedback_complex_tags(user_feedback: UserFeedback, mock_http: Mock):
    """Test creating user feedback with complex tags"""
    tags = {"sentiment": "positive", "relevance": 0.95, "tones": ["happy", "nice"]}

    user_feedback.create(annotation_task="task_123", entity_id="instance_456", tags=tags)

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


def test_create_feedback_parameter_validation(user_feedback: UserFeedback):
    """Test parameter validation for feedback creation"""
    with pytest.raises(ValueError, match="annotation_task is required"):
        user_feedback.create(annotation_task="", entity_id="instance_456", tags={"sentiment": "positive"})

    with pytest.raises(ValueError, match="entity_id is required"):
        user_feedback.create(annotation_task="task_123", entity_id="", tags={"sentiment": "positive"})

    with pytest.raises(ValueError, match="tags cannot be empty"):
        user_feedback.create(annotation_task="task_123", entity_id="instance_456", tags={})
