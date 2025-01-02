from typing import Literal, Dict, Any

from ..client.http import HTTPClient


class Annotation():
    """
    Annotation class for creating annotations in Traceloop.
    
    This class provides functionality to create annotations for specific tasks,
    supporting both user feedback and LLM feedback flows.
    """

    _http: HTTPClient
    _app_name: str
    _VALID_FLOWS = Literal["user_feedback", "llm_feedback"]

    def __init__(self, http: HTTPClient, app_name: str):
        self._http = http
        self._app_name = app_name

    def create(
        self,
        annotation_task_id: str,
        entity_instance_id: str,
        tags: Dict[str, Any],
        flow: _VALID_FLOWS = "user_feedback",
    ) -> None:
        """Create an annotation for a specific task.

        Args:
            annotation_task_id (str): The ID of the annotation task to report to.
                Can be found at app.traceloop.com/annotation_tasks/:annotation_task_id
            entity_instance_id (str): The ID of the specific entity instance being labeled
            tags (Dict[str, Any]): Dictionary containing the tags to be reported.
                Should match the tags defined in the annotation task
            flow (str, optional): The flow type of the annotation.
                Must be either "user_feedback" or "llm_feedback". Defaults to "user_feedback"

        Raises:
            ValueError: If flow is not one of "user_feedback" or "llm_feedback"

        Example:
            ```python
            client = Client(api_key="your-key")
            client.annotation.create(
                annotation_task_id="task_123",
                entity_instance_id="instance_456",
                tags={
                    "sentiment": "positive",
                    "relevance": 0.95,
                    "tones": ["happy", "nice"]
                },
                flow="user_feedback"
            )
            ```
        """
        if flow not in ["user_feedback", "llm_feedback"]:
            raise ValueError('flow must be either "user_feedback" or "llm_feedback"')

        self._http.post(
            f"annotation-tasks/{annotation_task_id}/annotations",
            {
                "annotation_task_id": annotation_task_id,
                "entity_instance_id": entity_instance_id,
                "tags": tags,
                "source": "sdk",
                "flow": flow,
                "actor": {
                    "type": "service",
                    "id": self._app_name,
                },
            },
        )

