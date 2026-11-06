from unittest.mock import MagicMock, patch

import pytest

from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.datasets.model import ColumnType

pd = pytest.importorskip("pandas")


def _build_request(df):
    datasets = Datasets(MagicMock())
    with patch.object(Datasets, "_create_dataset") as create, patch(
        "traceloop.sdk.datasets.datasets.Dataset.from_create_dataset_response"
    ):
        datasets.from_dataframe(df=df, slug="no-header")
    return create.call_args.args[0]


def test_from_dataframe_with_integer_column_labels():
    # A DataFrame built from a list of rows has integer column labels 0, 1, ...
    df = pd.DataFrame([["What is 2+2?", 4], ["Capital of France?", 5]])

    request = _build_request(df)

    assert [(c.slug, c.name) for c in request.columns] == [("0", "0"), ("1", "1")]
    assert request.columns[1].type == ColumnType.NUMBER
    assert request.rows == [
        {"0": "What is 2+2?", "1": 4},
        {"0": "Capital of France?", "1": 5},
    ]


def test_from_dataframe_column_slugs_match_row_keys():
    df = pd.DataFrame({"Question Text": ["hi"], 2024: [1]})

    request = _build_request(df)

    column_slugs = [c.slug for c in request.columns]
    assert column_slugs == ["question-text", "2024"]
    assert list(request.rows[0].keys()) == column_slugs
