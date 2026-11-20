from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import requests

from ..client.http import HTTPClient


class AnnotationCreateStatus(str, Enum):
    DELIVERED = "delivered"
    REFUSED = "refused"
    UNREACHABLE = "unreachable"


@dataclass
class AnnotationCreateResult:
    status: AnnotationCreateStatus
    payload: Optional[Any] = None
    status_code: Optional[int] = None
    error: Optional[requests.exceptions.RequestException] = None


class BaseAnnotation:
    """
    Annotation class for creating annotations in Traceloop.

    This class provides functionality to create annotations for specific tasks.
    """

    _http: HTTPClient
    _app_name: str

    def __init__(self, http: HTTPClient, app_name: str, flow: str):
        self._http = http
        self._app_name = app_name
        self._flow = flow

    def create(
        self,
        annotation_task: str,
        entity_id: str,
        tags: Dict[str, Any],
    ) -> AnnotationCreateResult:
        """Create an user feedback annotation for a specific task.

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
                entity_id="456",
                tags={
                    "sentiment": "positive",
                    "relevance": 0.95,
                    "tones": ["happy", "nice"]
                },
            )
            ```
        """

        if not annotation_task:
            raise ValueError("annotation_task is required")
        if not entity_id:
            raise ValueError("entity_id is required")
        if not tags:
            raise ValueError("tags cannot be empty")

        try:
            payload = self._http.post(
                f"annotation-tasks/{annotation_task}/annotations",
                {
                    "entity_instance_id": entity_id,
                    "tags": tags,
                    "source": "sdk",
                    "flow": self._flow,
                    "actor": {
                        "type": "service",
                        "id": self._app_name,
                    },
                },
                raise_on_error=True,
            )
            return AnnotationCreateResult(
                status=AnnotationCreateStatus.DELIVERED, payload=payload
            )
        except requests.exceptions.HTTPError as error:
            status_code = (
                error.response.status_code if error.response is not None else None
            )
            return AnnotationCreateResult(
                status=AnnotationCreateStatus.REFUSED,
                payload=self._extract_response_payload(error.response),
                status_code=status_code,
                error=error,
            )
        except requests.exceptions.RequestException as error:
            return AnnotationCreateResult(
                status=AnnotationCreateStatus.UNREACHABLE,
                error=error,
            )

    @staticmethod
    def _extract_response_payload(response: Optional[requests.Response]) -> Optional[Any]:
        if response is None:
            return None

        try:
            return response.json()
        except ValueError:
            if response.text:
                return response.text
            return None
