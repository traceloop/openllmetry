from unittest.mock import MagicMock, patch

from traceloop.sdk.datasets.datasets import Datasets


def _parse_csv(tmp_path, content):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(content, encoding="utf-8")

    datasets = Datasets(MagicMock())
    with patch.object(Datasets, "_create_dataset") as create, patch(
        "traceloop.sdk.datasets.datasets.Dataset.from_create_dataset_response"
    ):
        datasets.from_csv(file_path=str(csv_path), slug="single-column")
    return create.call_args.args[0]


def test_from_csv_single_column(tmp_path):
    request = _parse_csv(
        tmp_path,
        "question\nWhat is the capital of France?\nWho wrote Hamlet?\n",
    )

    assert [c.slug for c in request.columns] == ["question"]
    assert request.rows == [
        {"question": "What is the capital of France?"},
        {"question": "Who wrote Hamlet?"},
    ]


def test_from_csv_still_detects_semicolon_delimiter(tmp_path):
    request = _parse_csv(tmp_path, "name;price\nLaptop;999.99\nMouse;29.99\n")

    assert [c.slug for c in request.columns] == ["name", "price"]
    assert request.rows[0] == {"name": "Laptop", "price": "999.99"}
