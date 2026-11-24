from unittest.mock import patch

import pytest
import requests

from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.column import Column
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import ColumnType
from traceloop.sdk.datasets.row import Row


def _failed_response(status_code: int = 503) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = b"{}"
    return response


@pytest.fixture
def dataset() -> Dataset:
    http = HTTPClient("https://example.invalid", "test", "test")
    dataset = Dataset(http)
    dataset.slug = "evaluation"
    column = Column(http, dataset, "question", "Question", ColumnType.STRING)
    row = Row(http, dataset, "row-1", {"question": "What is 2 + 2?"})
    dataset.columns = [column]
    dataset.rows = [row]
    return dataset


def test_column_delete_failure_keeps_local_state(dataset: Dataset) -> None:
    column = dataset.columns[0]

    with patch("requests.delete", return_value=_failed_response()):
        with pytest.raises(Exception, match="Failed to delete column question"):
            column.delete()

    assert dataset.columns == [column]
    assert column.deleted is False
    assert dataset.rows[0].values == {"question": "What is 2 + 2?"}


def test_row_delete_failure_keeps_local_state(dataset: Dataset) -> None:
    row = dataset.rows[0]

    with patch("requests.delete", return_value=_failed_response()):
        with pytest.raises(Exception, match="Failed to delete row row-1"):
            row.delete()

    assert dataset.rows == [row]
    assert row.deleted is False
