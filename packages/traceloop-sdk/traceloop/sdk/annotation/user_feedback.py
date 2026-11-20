from typing import Any, Dict

from traceloop.sdk.client.http import HTTPClient
from .base_annotation import (
    AnnotationCreateResult,
    BaseAnnotation,
)


class UserFeedback(BaseAnnotation):
    def __init__(self, http: HTTPClient, app_name: str):
        super().__init__(http, app_name, "user_feedback")

    def create(
        self,
        annotation_task: str,
        entity_id: str,
        tags: Dict[str, Any],
    ) -> AnnotationCreateResult:
        """Create an annotation for a specific task.

        Args:
            annotation_task (str): The ID/slug of the annotation task to report to.
                Can be found at app.traceloop.com/annotation_tasks/:annotation_task_id
            entity_id (str): The ID of the specific entity being annotated, should be reported
                in the association properties
            tags (Dict[str, Any]): Dictionary containing the tags to be reported.
                Should match the tags defined in the annotation task

        Returns:
            AnnotationCreateResult: Result of annotation delivery.
                - DELIVERED: request succeeded and payload contains decoded response body.
                - REFUSED: server responded with a non-2xx status code.
                - UNREACHABLE: request failed before receiving an HTTP response.

        Example:
            ```python
            client = Client(api_key="your-key")
            client.annotation.create(
                annotation_task="task_123",
                entity_id="instance_456",
                tags={
                    "sentiment": "positive",
                    "relevance": 0.95,
                    "tones": ["happy", "nice"]
                },
            )
            ```
        """

        return BaseAnnotation.create(self, annotation_task, entity_id, tags)
